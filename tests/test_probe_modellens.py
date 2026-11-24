"""ModelLensExtractor with a fake lens and tokenizer: no torch needed."""

import importlib

import pytest

from app_assignment_1 import probe_modellens
from app_assignment_1.probe import (
    ActivationExtractor,
    ProbeExample,
    train_probe,
)
from app_assignment_1.probe_modellens import ModelLensExtractor


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


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
    """Pools like ModelLens: one row per character, so the reply matters."""

    def __init__(self, layers=28):
        self.names = [f"model.layers.{i}" for i in range(layers)]
        self.ran = None

    def layer_names(self):
        return self.names

    def probe_features(self, tokens, layer_name, pooling):
        self.ran = tokens
        rows = [[float(ord(c)), 1.0] for c in tokens.text]
        if pooling == "last":
            vector = rows[-1]
        else:
            vector = [sum(col) / len(rows) for col in zip(*rows, strict=True)]
        return {layer_name: [FakeVector(vector)]}


def test_extractor_reads_the_named_layer():
    extractor = ModelLensExtractor(FakeLens(), FakeTokenizer(), layer=14)
    assert extractor.layer_name == "model.layers.14"
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


def test_non_finite_activations_raise():
    lens = FakeLens()
    lens.probe_features = lambda tokens, name, pooling: {
        name: [FakeVector([float("nan")])]
    }
    extractor = ModelLensExtractor(lens, FakeTokenizer(), layer=0)
    with pytest.raises(ValueError, match="finite"):
        extractor.extract("q", "a")


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
