"""ModelLensExtractor with a fake lens and tokenizer: no torch needed."""

import importlib

import pytest

from app_assignment_1 import probe_modellens
from app_assignment_1.probe import (
    ActivationExtractor,
    ProbeExample,
    train_probe,
)
from app_assignment_1.probe_modellens import ModelLensExtractor, pool


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        assert index == 0
        return self

    def float(self):
        return self

    def tolist(self):
        return self.rows


class FakeTokens:
    def __init__(self, text):
        self.text = text
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize):
        assert tokenize is False
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        return FakeTokens(text)


class FakeLens:
    """Returns one row per character, so the reply changes the activations."""

    def __init__(self, layers=28):
        self.names = {f"model.layers.{i}" for i in range(layers)}
        self.attached = []
        self.last = None
        self.ran = None

    def attach_layers(self, names):
        for name in names:
            if name not in self.names:
                raise ValueError(f"Layer '{name}' not found.")
        self.attached = names
        return self

    def run(self, tokens):
        self.ran = tokens
        self.last = FakeTensor([[float(ord(c)), 1.0] for c in tokens.text])

    def get_layer_activation(self, name):
        return self.last if name in self.attached else None


def test_extractor_attaches_the_named_layer():
    lens = FakeLens()
    extractor = ModelLensExtractor(lens, FakeTokenizer(), layer=14)
    assert lens.attached == ["model.layers.14"]
    assert extractor.mocked is False


def test_extractor_fits_the_protocol():
    extractor: ActivationExtractor = ModelLensExtractor(
        FakeLens(), FakeTokenizer(), layer=0
    )
    assert extractor.mocked is False


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        ModelLensExtractor(FakeLens(layers=2), FakeTokenizer(), layer=5)


@pytest.mark.parametrize(
    ("layer", "pooling"), [(-1, "last"), (0, "max")], ids=["layer", "pooling"]
)
def test_bad_settings_are_rejected(layer, pooling):
    with pytest.raises(ValueError):
        ModelLensExtractor(FakeLens(), FakeTokenizer(), layer=layer, pooling=pooling)


def test_last_pooling_reads_the_final_token():
    extractor = ModelLensExtractor(FakeLens(), FakeTokenizer(), layer=3)
    assert extractor.extract("hi", "ok") == [float(ord("k")), 1.0]


def test_mean_pooling_averages_tokens():
    extractor = ModelLensExtractor(FakeLens(), FakeTokenizer(), 3, pooling="mean")
    # "a\nc": codes 97, 10, 99.
    assert extractor.extract("a", "c") == [(97 + 10 + 99) / 3, 1.0]


def test_chat_template_is_used_when_present():
    extractor = ModelLensExtractor(FakeLens(), FakeTokenizer("t"), layer=0)
    assert extractor.format_text("q", "a") == "user:q|assistant:a"


def test_tokens_move_to_the_device():
    lens = FakeLens()
    extractor = ModelLensExtractor(lens, FakeTokenizer(), layer=0, device="cuda")
    extractor.extract("q", "a")
    assert lens.ran.device == "cuda"


def test_missing_activation_raises():
    lens = FakeLens()
    extractor = ModelLensExtractor(lens, FakeTokenizer(), layer=0)
    lens.attached = []
    with pytest.raises(RuntimeError, match="No activation"):
        extractor.extract("q", "a")


def test_pool_rejects_empty_and_non_finite():
    with pytest.raises(ValueError, match="No tokens"):
        pool([], "last")
    with pytest.raises(ValueError, match="finite"):
        pool([[float("nan")]], "mean")
    with pytest.raises(ValueError, match="pooling"):
        pool([[1.0]], "max")  # type: ignore[arg-type]


def test_features_train_a_probe():
    extractor = ModelLensExtractor(FakeLens(), FakeTokenizer(), layer=0)
    examples = [
        ProbeExample(prompt=f"m{i}", reply=r, correct=r == "z", token_confidence=0.5)
        for i in range(4)
        for r in ("a", "z")
    ]
    features = [extractor.extract(e.prompt, e.reply) for e in examples]
    probe = train_probe(features, [e.correct for e in examples])
    assert len(probe.weights) == 2


def test_from_pretrained_explains_a_missing_extra(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, *args):
        if name in {"modellens", "transformers"}:
            raise ImportError(name)
        return real_import(name, *args)

    monkeypatch.setattr(probe_modellens.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match="--extra probe"):
        ModelLensExtractor.from_pretrained(layer=0)
