"""scripts/chaos_demo.py really drives all three faults against the gateway."""

import importlib.util
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app_assignment_1.api import create_app
from app_assignment_1.config import Settings

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "chaos_demo.py"
spec = importlib.util.spec_from_file_location("chaos_demo", SCRIPT)
assert spec and spec.loader
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class ClientApi:
    """Same call shape as demo.Api, but in process."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def call(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        if payload is None:
            return self.client.get(path).json()
        headers = {"Authorization": "Bearer " + demo.DEMO_TOKEN}
        response = self.client.post(path, json=payload, headers=headers)
        assert response.status_code == 200, response.text
        return response.json()


def demo_settings() -> Settings:
    # Mirrors compose.demo.yml, with a short cooldown so the test is quick.
    return Settings(
        _env_file=None,
        operator_token=demo.DEMO_TOKEN,
        semantic_baseline_disagreement_rate=0,
        semantic_actions_enabled=True,
        controller_cooldown_seconds=0.5,
    )


@pytest.mark.parametrize("name", list(demo.SCENARIOS))
def test_each_scenario_disables_and_recovers(name: str) -> None:
    with TestClient(create_app(demo_settings())) as client:
        lines: list[str] = []
        result = demo.run_scenario(
            ClientApi(client), name, out=lines.append, sleep=lambda _s: time.sleep(0.2)
        )
        assert result["messages_to_disable"] >= 1
        assert result["seconds_to_recover"] > 0
        text = "\n".join(lines)
        assert "CONTROLLER: " in text
        # The demo's own messages are the only text it sends.
        assert demo.STANDARD not in text and demo.COMPLEX not in text


def test_scenario_names_match_the_fault_kinds() -> None:
    assert {fault for fault, _ in demo.SCENARIOS.values()} == {
        "bad_format",
        "outage",
        "label_swap",
    }


def test_demo_override_matches_the_token() -> None:
    override = (SCRIPT.parent.parent / "compose.demo.yml").read_text()
    assert demo.DEMO_TOKEN in override
