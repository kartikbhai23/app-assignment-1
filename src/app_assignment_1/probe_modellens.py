"""Real activation extractor for the offline probe, built on ModelLens.

ModelLens (https://github.com/sv642-hub/DSAN-6600-ModelLens) wraps a
HuggingFace model and captures layer outputs with forward hooks. Its
`probe_features` reads one decoder layer of Qwen3-1.7B and pools it into one
vector. This module only adapts that to `ActivationExtractor`, so
`train_probe` and `run_gate` in `probe.py` work on real activations.

ModelLens, torch and transformers are in the optional `probe` extra and are
imported only by `ModelLensExtractor.from_pretrained`, so the gateway and the
test suite never load them. Nothing here runs in the gateway.
"""

import importlib
import math
from typing import Any, Literal, get_args

Pooling = Literal["last", "mean"]
POOLINGS: tuple[str, ...] = get_args(Pooling)

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

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
    ):
        if layer < 0:
            raise ValueError("layer must be 0 or more.")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}.")
        self.lens = lens
        self.tokenizer = tokenizer
        self.layer = layer
        self.pooling: Pooling = pooling
        self.device = device
        self.layer_name = LAYER_NAME.format(layer=layer)
        if self.layer_name not in lens.layer_names():
            raise ValueError(f"Layer '{self.layer_name}' not found.")

    @classmethod
    def from_pretrained(
        cls,
        layer: int,
        pooling: Pooling = "last",
        model_name: str = DEFAULT_MODEL,
        revision: str | None = None,
        device: str = "cpu",
    ) -> "ModelLensExtractor":
        """Load the model and tokenizer from HuggingFace and wrap them.

        Pin `revision` for a gate run, so the report can be repeated.
        """
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
        return cls(lens, tokenizer, layer, pooling, device)

    def format_text(self, prompt: str, reply: str) -> str:
        """The prompt and reply as the model saw them, using its chat
        template when the tokenizer has one."""
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ]
            text: str = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return text
        return f"{prompt}\n{reply}"

    def extract(self, prompt: str, reply: str) -> list[float]:
        tokens = self.tokenizer(self.format_text(prompt, reply), return_tensors="pt")
        if self.device is not None:
            tokens = tokens.to(self.device)
        features = self.lens.probe_features(tokens, self.layer_name, self.pooling)
        # Batch of one: shape [1, hidden].
        vector: list[float] = features[self.layer_name][0].tolist()
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Activations must be finite numbers.")
        return vector
