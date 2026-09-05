"""Background outcome-sweeper thread on the experiment runner."""

from __future__ import annotations

import threading
import time


def test_sweeper_fires_at_interval(monkeypatch):
    """The background thread calls attach_outcomes_to_index at each tick
    after the warmup delay."""
    from agentic_investor.experiments import runner as runner_mod

    # Shrink the warmup + interval so the test doesn't sit for a minute.
    calls: list[tuple] = []

    def _fake_sweep():
        calls.append(("swept", time.monotonic()))
        return (2, 1)

    # Intercept the deferred import inside _sweep.
    import agentic_investor.memory.outcomes as outcomes_mod
    monkeypatch.setattr(
        outcomes_mod, "attach_outcomes_to_index", _fake_sweep,
    )

    stop = threading.Event()
    # Replace the initial 60s warmup with something short via patch.
    orig_wait = stop.wait

    def _short_wait(timeout=None):
        # First call: warmup (60). Second onwards: interval * 60. Cap both.
        return orig_wait(0.1 if timeout is None else min(timeout, 0.1))

    stop.wait = _short_wait  # type: ignore[method-assign]

    t = runner_mod._start_outcome_sweeper(interval_min=1, stop_event=stop)
    # Give the loop enough ticks to fire at least twice.
    time.sleep(0.6)
    stop.set()
    t.join(timeout=2.0)

    assert len(calls) >= 2


def test_sweeper_error_does_not_kill_thread(monkeypatch):
    """A failing sweep must not stop future sweeps."""
    from agentic_investor.experiments import runner as runner_mod

    call_count = {"n": 0}

    def _flaky_sweep():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated chroma outage")
        return (5, 3)

    import agentic_investor.memory.outcomes as outcomes_mod
    monkeypatch.setattr(
        outcomes_mod, "attach_outcomes_to_index", _flaky_sweep,
    )

    stop = threading.Event()
    orig_wait = stop.wait
    stop.wait = lambda timeout=None: orig_wait(
        0.1 if timeout is None else min(timeout, 0.1),
    )  # type: ignore[method-assign]

    t = runner_mod._start_outcome_sweeper(interval_min=1, stop_event=stop)
    time.sleep(0.6)
    stop.set()
    t.join(timeout=2.0)

    # First call raised; subsequent calls still fired.
    assert call_count["n"] >= 2
