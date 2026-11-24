"""The start-at-boot units in deploy/autostart/ (README, Start at boot)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "deploy/autostart"
GPU = (
    "/usr/bin/docker compose -f compose.gpu.yml --env-file env/gpu.env "
    "-p smart-gateway-gpu"
)
CPU = "/usr/bin/docker compose -p PROJECT_NAME"


def service(name: str) -> dict[str, list[str]]:
    """The [Service] section's keys, each with every value in order."""
    values: dict[str, list[str]] = {}
    section = None
    for line in (UNITS / name).read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
        elif section == "[Service]" and "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values.setdefault(key, []).append(value)
    return values


@pytest.mark.parametrize(
    ("name", "compose"),
    [("smart-gateway-gpu.service", GPU), ("smart-gateway-cpu.service", CPU)],
)
def test_units_stop_then_start_the_stack_in_order(name, compose):
    unit = service(name)
    assert unit["WorkingDirectory"] == ["CHECKOUT_DIR"]
    # `down` first, so containers Docker restarted on its own at boot are
    # replaced by an ordered start that waits for health.
    assert unit["ExecStartPre"][-1] == f"{compose} down"
    assert unit["ExecStart"] == [f"{compose} up -d --wait"]
    assert unit["ExecStop"] == [f"{compose} down"]
    assert unit["Restart"] == ["on-failure"]
    # A boot start uses the images of the last deploy; it never builds.
    assert "--build" not in " ".join(unit["ExecStart"])


def test_gpu_unit_matches_the_readme_commands():
    readme = (ROOT / "README.md").read_text()
    assert GPU.removeprefix("/usr/bin/") + " up -d" in readme


def test_gpu_unit_checks_the_host_first():
    unit = service("smart-gateway-gpu.service")
    assert unit["ExecStartPre"][0] == "CHECKOUT_DIR/deploy/autostart/gpu-preflight.sh"
    script = UNITS / "gpu-preflight.sh"
    assert script.stat().st_mode & 0o111, "gpu-preflight.sh must be executable"
    text = script.read_text()
    # It reads only these two settings, so the API key never reaches the log.
    assert "VLLM_API_KEY" not in text
    for check in ("VLLM_BIND_IP", "MODEL_CACHE_DIR", "nvidia-smi"):
        assert check in text


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_gpu_preflight_is_valid_shell():
    subprocess.run(["sh", "-n", str(UNITS / "gpu-preflight.sh")], check=True)
