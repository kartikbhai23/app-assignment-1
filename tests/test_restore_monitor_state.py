"""After a restart, the gateway rebuilds its monitor windows and running
cooldowns from the store."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app_assignment_1.config import Settings
from app_assignment_1.monitor import Controller
from app_assignment_1.router import MockJevRouter
from app_assignment_1.schemas import Intervention, MonitorSample
from app_assignment_1.service import Gateway
from app_assignment_1.store import MemoryStore

STANDARD = "my card has not arrived"
SMALL, LARGE = "qwen3-1.7b", "qwen3-4b"
NOW = datetime(2026, 11, 22, 12, tzinfo=UTC)


def gateway(settings: Settings, store: MemoryStore) -> Gateway:
    return Gateway(settings, MockJevRouter(settings.intents_path), store=store)


def chat(g: Gateway, times: int, message: str = STANDARD) -> None:
    async def go() -> None:
        for _ in range(times):
            await g.chat(message)

    asyncio.run(go())


def failure(at: datetime, outcome: str = "ok", model: str = SMALL) -> MonitorSample:
    return MonitorSample(at=at, model=model, monitor="failure", outcome=outcome)


def event(at: datetime, action: str, model: str = SMALL) -> Intervention:
    return Intervention(at=at, backend=model, action=action, reason="test")


def test_a_restart_rebuilds_windows_and_resumes_the_cooldown():
    # The chaos config corrupts every small-model reply, so 5 standard
    # messages disable it. A second gateway on the same store is a restart.
    settings = Settings(
        _env_file=None, backends_path=Path("config/backends.chaos.json")
    )
    store = MemoryStore(settings.request_store_limit)
    first = gateway(settings, store)
    chat(first, 5)
    assert first.controller.is_disabled(SMALL)
    before = {m.name: m for m in first.controller.stats()}
    semantic_before = first.semantic.stats()

    second = gateway(settings, store)
    asyncio.run(second.restore())
    after = {m.name: m for m in second.controller.stats()}
    assert after[SMALL].disabled_by_controller
    assert after[SMALL].disabled_for_seconds is not None
    assert 55 < after[SMALL].disabled_for_seconds <= 60
    assert after[SMALL].format_failures == before[SMALL].format_failures == 5
    assert after[LARGE].samples == before[LARGE].samples
    assert second.semantic.stats() == semantic_before
    # Restored samples are not saved a second time.
    assert second.controller.unsaved_samples == []
    assert second.semantic.unsaved == []


def test_only_samples_after_the_last_re_enable_come_back(tmp_path):
    c = Controller([SMALL, LARGE], Settings(_env_file=None), clock=lambda: 0.0)
    samples = [
        failure(NOW - timedelta(minutes=10), "format_failure"),
        failure(NOW - timedelta(minutes=9), "format_failure"),
        failure(NOW - timedelta(minutes=2), "ok"),
    ]
    events = [
        event(NOW - timedelta(minutes=8), "disabled"),
        event(NOW - timedelta(minutes=7), "re-enabled"),
    ]
    c.restore(events, samples, NOW)
    assert list(c.windows[SMALL]) == ["ok"]
    assert not c.is_disabled(SMALL)


def test_a_cooldown_that_ended_while_stopped_starts_an_empty_window():
    c = Controller([SMALL, LARGE], Settings(_env_file=None), clock=lambda: 0.0)
    samples = [failure(NOW - timedelta(minutes=5), "format_failure")] * 5
    # Disabled 2 minutes ago with a 60 s cooldown: it would have been
    # re-enabled, which clears the window.
    c.restore([event(NOW - timedelta(minutes=2), "disabled")], samples, NOW)
    assert not c.is_disabled(SMALL)
    assert list(c.windows[SMALL]) == []


def test_a_running_cooldown_keeps_the_window_from_before_the_disable():
    clock = [100.0]
    c = Controller([SMALL, LARGE], Settings(_env_file=None), clock=lambda: clock[0])
    samples = [failure(NOW - timedelta(seconds=40), "format_failure")] * 5
    c.restore([event(NOW - timedelta(seconds=20), "disabled")], samples, NOW)
    assert c.disabled_until[SMALL] == 140.0  # 40 s of the 60 s cooldown left
    assert list(c.windows[SMALL]) == ["format_failure"] * 5
    clock[0] = 141.0
    assert not c.is_disabled(SMALL)


def test_an_operator_hold_comes_back_after_a_restart():
    c = Controller([SMALL, LARGE], Settings(_env_file=None), clock=lambda: 0.0)
    events = [
        event(NOW - timedelta(hours=2), "operator_disabled"),
        event(NOW - timedelta(hours=1), "operator_enabled"),
        event(NOW - timedelta(minutes=30), "operator_disabled"),
        event(NOW - timedelta(minutes=20), "operator_disabled", LARGE),
        event(NOW - timedelta(minutes=10), "operator_enabled", LARGE),
    ]
    c.restore(events, [], NOW)
    assert c.held == {SMALL}
    assert c.is_disabled(SMALL)
    assert not c.is_disabled(LARGE)
    # Restoring logs nothing new.
    assert c.unsaved == []


def test_an_operator_enable_ends_the_cooldown_and_clears_the_window():
    c = Controller([SMALL, LARGE], Settings(_env_file=None), clock=lambda: 0.0)
    samples = [
        failure(NOW - timedelta(seconds=40), "format_failure"),
        failure(NOW - timedelta(seconds=5), "ok"),
    ]
    events = [
        event(NOW - timedelta(seconds=30), "disabled"),
        event(NOW - timedelta(seconds=20), "operator_enabled"),
    ]
    c.restore(events, samples, NOW)
    assert not c.is_disabled(SMALL)
    assert list(c.windows[SMALL]) == ["ok"]


def test_a_disable_after_an_operator_enable_keeps_the_newer_samples():
    clock = [100.0]
    c = Controller([SMALL, LARGE], Settings(_env_file=None), clock=lambda: clock[0])
    samples = [
        failure(NOW - timedelta(minutes=5), "call_error"),
        failure(NOW - timedelta(seconds=30), "format_failure"),
    ]
    events = [
        event(NOW - timedelta(minutes=4), "operator_enabled"),
        event(NOW - timedelta(seconds=10), "disabled"),
    ]
    c.restore(events, samples, NOW)
    assert c.disabled_until[SMALL] == 150.0
    assert list(c.windows[SMALL]) == ["format_failure"]


def test_a_hold_survives_a_gateway_restart():
    settings = Settings(_env_file=None)
    store = MemoryStore(settings.request_store_limit)
    first = gateway(settings, store)
    first.operator_disable(SMALL, "maintenance")
    asyncio.run(first.save_interventions())
    second = gateway(settings, store)
    asyncio.run(second.restore())
    assert {m.name: m for m in second.controller.stats()}[SMALL].disabled_by_operator


def test_each_window_keeps_only_its_newest_samples():
    settings = Settings(_env_file=None, monitor_window=3, monitor_min_samples=3)
    c = Controller([SMALL, LARGE], settings, clock=lambda: 0.0)
    outcomes = ["call_error", "ok", "ok", "format_failure"]
    samples = [
        failure(NOW - timedelta(seconds=10 - i), o) for i, o in enumerate(outcomes)
    ]
    c.restore([], samples, NOW)
    assert list(c.windows[SMALL]) == ["ok", "ok", "format_failure"]


def test_a_store_problem_leaves_the_windows_empty(caplog):
    class Broken(MemoryStore):
        async def recent_monitor_samples(self, per_window):
            raise ConnectionError("down")

    settings = Settings(_env_file=None)
    g = gateway(settings, Broken(settings.request_store_limit))
    asyncio.run(g.restore())
    assert all(m.samples == 0 for m in g.controller.stats())
    assert "Could not restore monitor state (ConnectionError)" in caplog.text


def test_a_failed_sample_save_does_not_fail_the_request(caplog):
    class NoSamples(MemoryStore):
        async def save_monitor_samples(self, items):
            raise ConnectionError("down")

    settings = Settings(_env_file=None)
    g = gateway(settings, NoSamples(settings.request_store_limit))
    chat(g, 2)
    assert len(g.store.requests) == 2  # type: ignore[attr-defined]
    assert "Could not save monitor samples (ConnectionError)" in caplog.text
    assert g.controller.unsaved_samples == [] and g.semantic.unsaved == []


def test_a_restart_rebuilds_the_semantic_windows():
    settings = Settings(_env_file=None)
    store = MemoryStore(settings.request_store_limit)
    first = gateway(settings, store)
    chat(first, 3)
    chat(first, 2, "my card was stolen")  # the large model's route
    before = first.semantic.stats()
    assert [m.samples for m in before] == [3, 2]

    second = gateway(settings, store)
    asyncio.run(second.restore())
    assert second.semantic.stats() == before


def test_a_cancelled_intervention_save_keeps_the_interventions():
    # /v1/health calls save_interventions directly, outside the shielded chat
    # save, so its client leaving can cancel the save (review of #92).
    class Slow(MemoryStore):
        async def save_interventions(self, items):
            await asyncio.sleep(10)

    settings = Settings(_env_file=None)
    g = gateway(settings, Slow(settings.request_store_limit))
    for _ in range(5):
        g.controller.record_error(SMALL, can_disable=True)
    pending = list(g.controller.unsaved)
    assert pending

    async def go() -> None:
        task = asyncio.create_task(g.save_interventions())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("the cancel must pass on")

    asyncio.run(go())
    assert g.controller.unsaved == pending
