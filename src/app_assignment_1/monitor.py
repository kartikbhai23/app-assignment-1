"""In-memory failure monitor and controller for the backends.

Windows and disabled backends live in the gateway process. Each window sample
is saved through `Store` (`unsaved_samples`), and `restore` rebuilds the windows
and running cooldowns at startup, so with Postgres they survive a restart.
The gateway saves each intervention to its store (see store.py), which may be
Postgres; `unsaved` holds the ones it has not saved yet.
"""

import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

from app_assignment_1.config import Settings
from app_assignment_1.schemas import (
    BackendMonitor,
    BackendName,
    ControllerRules,
    Intervention,
    InterventionAction,
    MonitorSample,
)

Outcome = Literal["ok", "format_failure", "call_error"]
# Interventions that start or end a cooldown. Both kinds of enable clear the
# window; an operator enable also ends a running cooldown.
WINDOW_EVENTS = ("disabled", "re-enabled", "operator_enabled")
OUTCOMES: tuple[Outcome, ...] = ("ok", "format_failure", "call_error")


class Controller:
    """Disables a backend whose recent failure rate crosses a threshold.

    A failure is a reply that fails the format check or a call that raises.

    Rules: a sliding window per backend, a minimum sample count before acting,
    a cooldown after which the backend is re-enabled with a fresh window, and an
    action limit that never disables the last usable backend.
    """

    def __init__(
        self,
        names: Iterable[BackendName],
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.windows: dict[BackendName, deque[Outcome]] = {
            name: deque(maxlen=settings.monitor_window) for name in names
        }
        self.window = settings.monitor_window
        self.min_samples = settings.monitor_min_samples
        self.threshold = settings.monitor_failure_threshold
        self.cooldown = settings.controller_cooldown_seconds
        self.clock = clock
        self.disabled_until: dict[BackendName, float] = {}
        # Disabled by an operator: no cooldown, only an operator enables it.
        self.held: set[BackendName] = set()
        self.warned: set[BackendName] = set()
        self.log: deque[Intervention] = deque(maxlen=50)
        # Every intervention since start; the log above keeps only the latest.
        self.action_counts: Counter[tuple[str, BackendName]] = Counter()
        self.unsaved: list[Intervention] = []
        # New window samples, waiting for the gateway to save them, so the
        # windows can be rebuilt after a restart (see `restore`).
        self.unsaved_samples: list[MonitorSample] = []

    def _note(
        self,
        name: BackendName,
        action: InterventionAction,
        reason: str,
        ago: float = 0,
    ) -> None:
        at = datetime.now(UTC) - timedelta(seconds=ago)
        item = Intervention(at=at, backend=name, action=action, reason=reason)
        self.log.append(item)
        self.unsaved.append(item)
        self.action_counts[(action, name)] += 1

    def operator_disable(self, name: BackendName, reason: str) -> None:
        """Hold a backend disabled until operator_enable. The caller checks that
        another backend stays usable."""
        if name in self.held:
            return
        self.held.add(name)
        self._note(name, "operator_disabled", reason)

    def operator_enable(self, name: BackendName, reason: str) -> None:
        """Lift an operator hold and any controller cooldown, with a fresh window."""
        if name not in self.held and name not in self.disabled_until:
            return
        self.held.discard(name)
        self.disabled_until.pop(name, None)
        self.windows[name].clear()
        self.warned.discard(name)
        self._note(name, "operator_enabled", reason)

    def is_disabled(self, name: BackendName) -> bool:
        # Check the cooldown first, so one that ended under a hold still logs.
        cooling = self._cooling_down(name)
        return cooling or name in self.held

    def _cooling_down(self, name: BackendName) -> bool:
        until = self.disabled_until.get(name)
        if until is None:
            return False
        now = self.clock()
        if now < until:
            return True
        # Re-enabling happens on the next check; log when the cooldown ended.
        del self.disabled_until[name]
        self.windows[name].clear()
        self._note(name, "re-enabled", "cooldown elapsed; window reset", now - until)
        return False

    def record(self, name: BackendName, failed: bool, can_disable: bool) -> str | None:
        """Record one reply's format check; return the action taken, if any."""
        outcome: Outcome = "format_failure" if failed else "ok"
        return self._record(name, outcome, can_disable)

    def record_error(self, name: BackendName, can_disable: bool) -> str | None:
        """Record one call that raised; return the action taken, if any."""
        return self._record(name, "call_error", can_disable)

    def _record(
        self, name: BackendName, outcome: Outcome, can_disable: bool
    ) -> str | None:
        window = self.windows[name]
        window.append(outcome)
        self.unsaved_samples.append(
            MonitorSample(
                at=datetime.now(UTC), model=name, monitor="failure", outcome=outcome
            )
        )
        failures = sum(item != "ok" for item in window)
        rate = failures / len(window)
        if len(window) < self.min_samples or rate < self.threshold:
            self.warned.discard(name)
            return None
        if name in self.disabled_until:
            return None
        errors = window.count("call_error")
        kind = (
            "format failure"
            if not errors
            else "call error"
            if errors == failures
            else "failure"
        )
        reason = f"{kind} rate {rate:.0%} over last {len(window)} calls"
        if not can_disable:
            # Log once per run above the threshold, not on every failing request.
            if name not in self.warned:
                self.warned.add(name)
                self._note(name, "kept_last_backend", reason)
            return None
        self.warned.discard(name)
        self.disabled_until[name] = self.clock() + self.cooldown
        self._note(name, "disabled", reason)
        return f"disabled {name}: {reason}"

    def restore(
        self,
        interventions: Sequence[Intervention],
        samples: Sequence[MonitorSample],
        now: datetime,
    ) -> None:
        """Rebuild the windows, cooldowns and operator holds after a restart.

        A cooldown still running when the gateway stopped resumes for the time
        left, unless an operator ended it. A re-enable, by the controller or an
        operator, clears a model's window, so only samples after the last one
        go back in, or after a disable whose cooldown has since ended. A model
        whose last operator action was a disable is held again.
        `interventions` and `samples` are oldest first.
        """
        for name, window in self.windows.items():
            events = [
                i
                for i in interventions
                if i.backend == name and i.action in WINDOW_EVENTS
            ]
            cutoff: datetime | None = None
            if events:
                last = events[-1]
                left = self.cooldown - (now - last.at).total_seconds()
                if last.action == "disabled" and left > 0:
                    self.disabled_until[name] = self.clock() + left
                    enabled = [e for e in events if e.action != "disabled"]
                    cutoff = enabled[-1].at if enabled else None
                else:
                    cutoff = last.at
            holds = [
                i.action
                for i in interventions
                if i.backend == name
                and i.action in ("operator_disabled", "operator_enabled")
            ]
            if holds and holds[-1] == "operator_disabled":
                self.held.add(name)
            outcomes = [
                s.outcome
                for s in samples
                if s.model == name
                and s.monitor == "failure"
                and (cutoff is None or s.at > cutoff)
            ]
            window.clear()
            window.extend(o for o in outcomes if o in OUTCOMES)

    def rules(self) -> ControllerRules:
        return ControllerRules(
            window=self.window,
            min_samples=self.min_samples,
            failure_threshold=self.threshold,
            cooldown_seconds=self.cooldown,
        )

    def stats(self) -> list[BackendMonitor]:
        result = []
        for name, window in self.windows.items():
            disabled = self._cooling_down(name)
            format_failures = window.count("format_failure")
            call_errors = window.count("call_error")
            failures = format_failures + call_errors
            result.append(
                BackendMonitor(
                    name=name,
                    samples=len(window),
                    format_failures=format_failures,
                    call_errors=call_errors,
                    failure_rate=round(failures / len(window), 2) if window else None,
                    disabled_by_controller=disabled,
                    disabled_by_operator=name in self.held,
                    disabled_for_seconds=(
                        round(self.disabled_until[name] - self.clock(), 1)
                        if disabled
                        else None
                    ),
                )
            )
        return result
