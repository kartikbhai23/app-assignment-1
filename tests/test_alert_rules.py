"""Every gateway metric an alert uses is one the gateway serves, so a renamed
metric can't silently turn an alert off. promtool tests the alert logic in
prometheus/alerts_test.yml."""

import re
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from app_assignment_1.api import create_app
from app_assignment_1.config import Settings

RULES = Path("prometheus/alerts.yml")
RUNBOOK = Path("docs/MONITORING.md")
RUNBOOK_URL = (
    "https://github.com/sv642-hub/app-assignment-1/blob/main/docs/MONITORING.md#"
)
SUFFIXES = ("_total", "_bucket", "_sum", "_count", "_created")


def served_metric_names() -> set[str]:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        client.post("/v1/chat", json={"message": "my card has not arrived"})
        text = client.get("/metrics").text
    return {
        line.split("{")[0].split(" ")[0]
        for line in text.splitlines()
        if line and not line.startswith("#")
    }


def test_alert_rules_only_use_served_gateway_metrics():
    rules = yaml.safe_load(RULES.read_text())["groups"][0]["rules"]
    used = {
        name
        for rule in rules
        for name in re.findall(r"\bgateway_[a-z_]+", rule["expr"])
    }
    assert used, "the rules should use gateway metrics"
    served = served_metric_names()
    missing = sorted(
        name
        for name in used
        if name not in served and not any(name + s in served for s in SUFFIXES)
    )
    assert missing == []


def test_every_alert_has_a_severity_and_a_summary():
    for rule in yaml.safe_load(RULES.read_text())["groups"][0]["rules"]:
        assert rule["labels"]["severity"] in ("warning", "critical"), rule["alert"]
        assert rule["annotations"]["summary"], rule["alert"]


def test_prometheus_loads_the_rules_file():
    config = yaml.safe_load(Path("prometheus/prometheus.yml").read_text())
    assert config["rule_files"] == ["alerts.yml"]
    compose = Path("compose.yml").read_text()
    assert "./prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro" in compose


def test_every_alert_links_to_its_runbook_section():
    # GitHub turns the heading "### GatewayDown" into the anchor #gatewaydown.
    alerts = [
        rule["alert"]
        for rule in yaml.safe_load(RULES.read_text())["groups"][0]["rules"]
    ]
    text = RUNBOOK.read_text()
    alerts_part = text.split("\n## Alerts\n", 1)[1].split("\n## ", 1)[0]
    headings = re.findall(r"^### (\S+)$", alerts_part, re.M)
    assert headings == alerts
    for rule in yaml.safe_load(RULES.read_text())["groups"][0]["rules"]:
        url = rule["annotations"]["runbook_url"]
        assert url == RUNBOOK_URL + rule["alert"].lower()
