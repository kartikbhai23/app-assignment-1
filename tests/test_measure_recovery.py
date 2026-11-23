"""scripts/measure_recovery.py: the measurements, and every scenario run
end to end against the app in process, with a 1-second cooldown."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app_assignment_1.api import create_app
from app_assignment_1.config import Settings

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_recovery.py"
TOKEN = "t" * 32


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_recovery", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mr = load()


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    backends = json.loads(Path("config/backends.json").read_text())
    for backend in backends:
        backend["latency_ms"] = 0
    path = tmp_path / "backends.json"
    path.write_text(json.dumps(backends))
    values = {
        "backends_path": path,
        "operator_token": TOKEN,
        "controller_cooldown_seconds": 1,
        "semantic_actions_enabled": True,
        "semantic_baseline_disagreement_rate": 0.0,
    } | overrides
    return Settings(_env_file=None, **values)


def transport(client: TestClient):
    def send(method: str, path: str, payload: Any, headers: dict[str, str]):
        response = client.request(method, path, json=payload, headers=headers)
        return response.status_code, response.json()

    return send


def reply(at: float, phase: str, **fields: Any):
    defaults = {
        "route": "standard",
        "status": 200,
        "model": mr.SMALL,
        "intent": "card_arrival",
        "format_check": "passed",
        "expected_intent": "card_arrival",
        "latency_ms": 10.0,
    }
    return mr.Reply(at=at, phase=phase, **(defaults | fields))


def test_summarize():
    disabled = f"disabled {mr.SMALL}: format failure rate 50% over last 10 calls"
    replies = [
        reply(0.0, "normal"),
        reply(0.5, "normal", route="complex", model=mr.LARGE),
        reply(1.0, "fault", format_check="failed", intent="unknown"),
        reply(1.5, "fault", route="complex", model=mr.LARGE, latency_ms=30.0),
        reply(
            2.0,
            "fault",
            format_check="failed",
            intent="unknown",
            controller_action=disabled,
        ),
        reply(2.5, "fault", model=mr.LARGE),
        reply(3.0, "cleared", model=mr.LARGE),
        reply(3.5, "cleared", status=503, model=None, intent=None, format_check=None),
        reply(5.0, "cleared"),
    ]
    result = mr.summarize("bad_format", replies, fault_at=0.8)
    assert result.detected
    assert result.detect_seconds == 1.2
    # Two standard requests up to and including the one that disabled it.
    assert result.requests_to_detect == 2
    assert result.bad_replies_before_detect == 2
    assert result.unavailable == 1
    assert result.false_interventions == 0
    assert result.restored
    assert result.restore_seconds == 2.0
    assert result.latency_ms_normal == 10.0
    assert result.latency_ms_fault == 10.0


def test_a_controller_action_before_the_fault_is_a_false_intervention():
    replies = [
        reply(0.0, "normal", controller_action="disabled qwen3-4b: ..."),
        reply(1.0, "fault"),
    ]
    result = mr.summarize("outage", replies, fault_at=0.5)
    assert result.false_interventions == 1
    assert not result.detected
    assert not result.restored


@pytest.mark.parametrize("scenario", ["outage", "bad_format", "label_swap"])
def test_each_scenario_is_detected_and_recovers(tmp_path, scenario):
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        gateway = mr.Gateway(transport(client), TOKEN)
        result = mr.run_scenario(gateway, scenario, interval=0, timeout=10)
        health = client.get("/v1/health").json()
    assert result.detected, scenario
    assert result.restored, scenario
    assert result.false_interventions == 0
    assert result.unavailable == 0
    # 1 s cooldown: the model serves again about a second after the fault ends.
    assert 0 < result.restore_seconds < 3
    if scenario == "outage":
        # The large model answers every request the small one fails.
        assert result.bad_replies_before_detect == 0
    else:
        assert result.bad_replies_before_detect > 0
    # Every fault is cleared afterwards.
    assert all(
        b["error_rate"] == b["fault_rate"] == b["label_swap_rate"] == 0
        for b in health["backends"]
    )


def test_the_next_scenario_waits_out_a_cooldown(tmp_path):
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        gateway = mr.Gateway(transport(client), TOKEN)
        first = mr.run_scenario(gateway, "outage", interval=0, timeout=10)
        second = mr.run_scenario(gateway, "bad_format", interval=0, timeout=10)
    assert first.detected and second.detected
    assert second.false_interventions == 0


def test_without_the_token_it_stops_with_a_clear_error(tmp_path):
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        gateway = mr.Gateway(transport(client), None)
        with pytest.raises(RuntimeError, match="APP_OPERATOR_TOKEN"):
            mr.run_scenario(gateway, "outage", interval=0, timeout=10)


def test_label_swap_needs_semantic_actions_and_a_baseline(tmp_path):
    for overrides, ready in [
        ({}, True),
        ({"semantic_actions_enabled": False}, False),
        ({"semantic_baseline_disagreement_rate": None}, False),
    ]:
        with TestClient(create_app(settings(tmp_path, **overrides))) as client:
            assert mr.label_swap_ready(client.get("/v1/health").json()) is ready


def test_refuses_a_remote_host(capsys):
    assert mr.main(["http://gateway.example.org:8080"]) == 2
    assert "Refusing gateway.example.org" in capsys.readouterr().err
