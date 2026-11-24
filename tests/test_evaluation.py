"""Offline evaluation: datasets, scoring, the oracle and the runner script."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from app_assignment_1.config import Settings
from app_assignment_1.evaluation import (
    gold_classes,
    oracle,
    parse_banking77,
    parse_labelled_csv,
    reply_format_errors,
    score,
    wilson,
)
from app_assignment_1.labels import load_labels
from app_assignment_1.router import MockJevRouter
from app_assignment_1.schemas import BackendConfig, BackendReply
from app_assignment_1.service import Gateway

ROOT = Path(__file__).resolve().parents[1]
LABELS = load_labels(ROOT / "config/labels.json")
CARDS = set(json.loads((ROOT / "config/policy_cards.json").read_text()))


def item(row, gold, predicted, **extra):
    return {"row": row, "gold": gold, "predicted": predicted, "error": None,
            "latency_ms": 100.0, "model": "qwen3-1.7b", **extra}  # fmt: skip


def test_banking77_names_map_to_our_labels():
    body = (
        b"text,category\n"
        b"a,card_arrival\n"
        b"b,Refund_not_showing_up\n"
        b"c,reverted_card_payment?\n"
    )
    messages = parse_banking77(body, LABELS)
    assert [(m.row, m.gold) for m in messages] == [
        (0, "card_arrival"),
        (1, "refund_not_showing_up"),
        (2, "reverted_card_payment"),
    ]
    with pytest.raises(ValueError, match="not a Banking77 intent"):
        parse_banking77(b"text,category\na,made_up\n", LABELS)


def test_labelled_csv_accepts_labels_banking77_names_and_out_of_scope():
    body = (
        b"text,intent,writer\n"
        b"a,card_arrival,x\n"
        b"b,Refund_not_showing_up,x\n"
        b"c,out_of_scope,x\n"
        b"d,unknown,x\n"
    )
    golds = [m.gold for m in parse_labelled_csv(body, LABELS)]
    assert golds == ["card_arrival", "refund_not_showing_up", "unknown", "unknown"]
    with pytest.raises(ValueError, match="not a label"):
        parse_labelled_csv(b"text,intent\na,made_up\n", LABELS)


def test_macro_f1_averages_over_the_77_intents():
    classes = gold_classes(LABELS)
    assert len(classes) == 77 and "unknown" not in classes
    items = [
        item(0, "card_arrival", "card_arrival"),
        item(1, "card_arrival", "unknown"),  # unknown is wrong, not a free pass
        item(2, "age_limit", "card_arrival"),  # a false positive for card_arrival
        {**item(3, "age_limit", None), "error": "BackendTimeoutError"},
    ]
    result = score(items, LABELS)
    assert result["accuracy"] == 0.25
    assert result["errors"] == {"BackendTimeoutError": 1}
    assert result["predicted_unknown"] == 1
    # card_arrival: precision 1/2, recall 1/2, F1 0.5; every other class 0.
    assert result["macro_f1"] == pytest.approx(0.5 / 77)
    assert result["per_class_recall"] == {"card_arrival": 0.5, "age_limit": 0.0}
    assert result["latency_ms_p50"] == 100.0


def test_wilson_interval():
    low, high = wilson(50, 100)
    assert low == pytest.approx(0.404, abs=0.001)
    assert high == pytest.approx(0.596, abs=0.001)
    assert wilson(0, 0) == (0.0, 0.0)


def test_oracle_pays_for_the_large_model_only_where_it_helps():
    small = [
        item(0, "card_arrival", "card_arrival"),  # small right
        item(1, "age_limit", "unknown"),  # small wrong, large right
        item(2, "atm_support", "unknown"),  # both wrong
    ]
    large = [
        item(0, "card_arrival", "card_arrival", model="qwen3-4b"),
        item(1, "age_limit", "age_limit", model="qwen3-4b"),
        item(2, "atm_support", "age_limit", model="qwen3-4b"),
    ]
    combined = oracle(small, large)
    assert [c["model"] for c in combined] == ["qwen3-1.7b", "qwen3-4b", "qwen3-1.7b"]
    assert score(combined, LABELS)["correct"] == 2
    with pytest.raises(ValueError, match="only in the large"):
        oracle(small[:2], large)


class FixedBackend:
    def __init__(self, name, route, reply):
        self.config = BackendConfig(name=name, route=route, latency_ms=0)
        self.reply = reply

    async def generate(self, message):
        return self.reply


@pytest.mark.parametrize(
    "reply",
    [
        BackendReply(intent="card_arrival", reply="It arrives soon.", policy_facts=[]),
        BackendReply(intent="card_arrival", reply="", policy_facts=[]),
        BackendReply(intent="card_arrival", reply="Hi {name}", policy_facts=[]),
        BackendReply(intent="made_up", reply="Hi.", policy_facts=[]),
        BackendReply(intent="unknown", reply="Please describe it.", policy_facts=[]),
        BackendReply(intent="unknown", reply="Please.", policy_facts=["a fact"]),
        BackendReply(intent="age_limit", reply="", policy_facts=[]),
        BackendReply(intent="age_limit", reply="", policy_facts=["a fact"]),
    ],
)
def test_format_errors_agree_with_the_gateway(reply):
    settings = Settings(_env_file=None)
    backends = [
        FixedBackend("qwen3-1.7b", "standard", reply),
        FixedBackend("qwen3-4b", "complex", reply),
    ]
    gateway = Gateway(settings, MockJevRouter(settings.intents_path), backends)
    trace = asyncio.run(gateway.chat("card arrival")).trace
    assert reply_format_errors(reply, LABELS, CARDS) == trace.format_errors


def load_script():
    spec = importlib.util.spec_from_file_location(
        "run_eval", ROOT / "scripts/run_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_on_a_csv_with_the_mocks(tmp_path, capsys):
    run_eval = load_script()
    texts = ["When will my new card arrive?", "How old do I need to be?"]
    data = tmp_path / "fresh.csv"
    data.write_text(f"text,intent\n{texts[0]},card_arrival\n{texts[1]},age_limit\n")
    small, large = tmp_path / "small.jsonl", tmp_path / "large.jsonl"
    for target, out in (("small", small), ("large", large)):
        args = ["run", "--target", target, "--mock", "--dataset-csv", str(data)]
        assert run_eval.main([*args, "--out", str(out)]) == 0

    header, items = run_eval.read_results(small)
    assert header["dataset"] == "fresh" and header["mocked"] is True
    assert header["dataset_size"] == 2 and len(items) == 2
    assert items[0]["gold"] == "card_arrival" and items[0]["error"] is None
    # No message text anywhere in the results.
    assert not any(text in small.read_text() for text in texts)

    # Running again skips the rows already done and adds nothing.
    assert run_eval.main([*args, "--out", str(large)]) == 0
    assert len(large.read_text().splitlines()) == 3

    # A results file from another run is refused, not mixed.
    with pytest.raises(SystemExit, match="different run"):
        run_eval.main(["run", "--target", "small", "--mock", "--dataset-csv",
                       str(data), "--out", str(large)])  # fmt: skip

    assert run_eval.main(["oracle", str(small), str(large)]) == 0
    assert "large model on 0/2" in capsys.readouterr().out


def test_runner_refuses_a_remote_gateway(tmp_path):
    run_eval = load_script()
    data = tmp_path / "fresh.csv"
    data.write_text("text,intent\nhello,card_arrival\n")
    with pytest.raises(SystemExit, match="Refusing example.org"):
        run_eval.main(["run", "--target", "gateway", "--base-url",
                       "http://example.org:8080", "--dataset-csv", str(data),
                       "--out", str(tmp_path / "g.jsonl")])  # fmt: skip
    assert not (tmp_path / "g.jsonl").exists()


def test_runner_needs_a_key_for_a_real_model(tmp_path, monkeypatch):
    run_eval = load_script()
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    data = tmp_path / "fresh.csv"
    data.write_text("text,intent\nhello,card_arrival\n")
    with pytest.raises(SystemExit, match="VLLM_API_KEY"):
        run_eval.main(["run", "--target", "small", "--base-url",
                       "http://10.0.0.1:8001", "--dataset-csv", str(data),
                       "--out", str(tmp_path / "s.jsonl")])  # fmt: skip
