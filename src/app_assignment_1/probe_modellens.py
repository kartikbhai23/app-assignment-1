"""Real activation extractor for the offline probe, built on ModelLens.

ModelLens (https://github.com/SebastianVillalobosAlva/ModelLens) wraps a
HuggingFace model and captures layer outputs with forward hooks. This module
captures one Qwen3-1.7B decoder layer with it and pools the layer with the
same function ModelLens's own linear probes use
(`modellens.analysis.probing`). It only adapts that to `ActivationExtractor`,
so `train_probe` and `run_gate` in `probe.py` work on real activations.

The text the model reads is rebuilt the way vLLM builds it for the gateway:
the gateway's system prompt, the customer message, the chat template's
generation prompt, then the reply exactly as the model generated it. Nothing
follows the reply, so `last` pooling lands on the reply's final token, not on
the template's closing tokens.

ModelLens, torch and transformers are in the optional `probe` extra and are
imported only by `ModelLensExtractor.from_pretrained`, so the gateway and the
test suite never load them. Nothing here runs in the gateway.
"""

import importlib
import math
from collections.abc import Callable
from typing import Any, Literal, get_args

from pydantic import TypeAdapter

from app_assignment_1.config import get_settings
from app_assignment_1.labels import load_labels
from app_assignment_1.schemas import PolicyCard
from app_assignment_1.vllm import build_system_prompt

Pooling = Literal["last", "mean"]
POOLINGS: tuple[str, ...] = get_args(Pooling)

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
# The revision vLLM serves on the GPU host (compose.gpu.yml).
DEFAULT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Qwen3ForCausalLM names its decoder layers model.layers.0 to model.layers.27.
LAYER_NAME = "model.layers.{layer}"

MISSING_EXTRA = (
    "ModelLensExtractor needs the probe extra: "
    "uv sync --extra dev --extra probe (see docs/PROBE.md)."
)


class ModelLensExtractor:
    """Pools one decoder layer's output into a probe feature vector.

    `lens` is a `modellens.ModelLens` around a causal language model and
    `tokenizer` its HuggingFace tokenizer. Use `from_pretrained` to build both.

    Pooling is done by ModelLens: `last` keeps the final token of the
    formatted prompt and reply, and `mean` averages over all tokens.
    docs/PROBE.md proposes the last intent-label token as the primary
    position; that needs the reply's token offsets and is still to build.
    """

    mocked = False

    def __init__(
        self,
        lens: Any,
        tokenizer: Any,
        layer: int,
        pooling: Pooling = "last",
        device: str | None = None,
        pool: Callable[[Any, str], Any] | None = None,
        system_prompt: str | None = None,
    ):
        """`system_prompt` is the system message the gateway sent with the
        message (see `gateway_system_prompt`); None leaves it out. `pool`
        defaults to ModelLens's probe pooling; tests pass a fake."""
        if layer < 0:
            raise ValueError("layer must be 0 or more.")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}.")
        self.lens = lens
        self.tokenizer = tokenizer
        self.layer = layer
        self.pooling: Pooling = pooling
        self.device = device
        self.pool = pool
        self.system_prompt = system_prompt
        self.layer_name = LAYER_NAME.format(layer=layer)
        if self.layer_name not in lens.layer_names():
            raise ValueError(f"Layer '{self.layer_name}' not found.")

    @classmethod
    def from_pretrained(
        cls,
        layer: int,
        pooling: Pooling = "last",
        model_name: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_REVISION,
        device: str = "cpu",
        system_prompt: str | None = None,
    ) -> "ModelLensExtractor":
        """Load the model and tokenizer from HuggingFace and wrap them.

        `revision` defaults to the one vLLM serves, so the probe reads the
        same weights that produced the replies. `system_prompt` defaults to
        the gateway's (`gateway_system_prompt`); pass "" to leave it out.
        """
        if system_prompt is None:
            system_prompt = gateway_system_prompt()
        try:
            modellens = importlib.import_module("modellens")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise ImportError(MISSING_EXTRA) from error
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, revision=revision
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype="auto"
        ).to(device)
        lens = modellens.ModelLens(model, backend="huggingface")
        lens.adapter.set_tokenizer(tokenizer)
        return cls(
            lens, tokenizer, layer, pooling, device, system_prompt=system_prompt or None
        )

    def format_text(self, prompt: str, reply: str) -> str:
        """The text the model read when it wrote `reply`.

        With a chat template, this is the template applied to the system and
        user messages with the generation prompt, followed by `reply`. Pass
        the reply exactly as the model generated it (the raw completion,
        including any thinking block), so the tokens match.
        """
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        if getattr(self.tokenizer, "chat_template", None):
            prefix: str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prefix + reply
        return "\n".join(message["content"] for message in messages) + "\n" + reply

    def extract(self, prompt: str, reply: str) -> list[float]:
        tokens = self.tokenizer(self.format_text(prompt, reply), return_tensors="pt")
        if self.device is not None:
            tokens = tokens.to(self.device)
        self.lens.attach_layers([self.layer_name])
        self.lens.run(tokens)
        activation = self.lens.get_layer_activation(self.layer_name)
        if activation is None:
            raise RuntimeError(f"No activation captured for {self.layer_name}.")
        pool = self.pool or _modellens_pool()
        # Batch of one: [1, tokens, hidden] pools to [1, hidden].
        vector: list[float] = pool(activation, self.pooling)[0].tolist()
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Activations must be finite numbers.")
        return vector


def gateway_system_prompt() -> str:
    """The system prompt the gateway sends to vLLM, built from the same
    labels and policy cards (paths from the gateway's settings)."""
    settings = get_settings()
    cards = TypeAdapter(dict[str, PolicyCard]).validate_json(
        settings.policy_cards_path.read_text()
    )
    return build_system_prompt(load_labels(settings.labels_path), cards)


def _modellens_pool() -> Callable[[Any, str], Any]:
    """ModelLens's pooling for its own probes, so features here match
    `modellens.analysis.probing.train_probe` on the same layer."""
    try:
        probing = importlib.import_module("modellens.analysis.probing")
    except ImportError as error:
        raise ImportError(MISSING_EXTRA) from error
    pool: Callable[[Any, str], Any] = probing._pool
    return pool
