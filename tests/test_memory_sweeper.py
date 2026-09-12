"""OutcomeSweeper subprocess loop.

The sweeper now runs as its own subprocess so a chromadb Rust crash
isolates from the paper-experiment supervisor. Tests exercise the
underlying loop class in-process; the subprocess entry point
(run_outcome_sweeper) is just a signal-handling wrapper around it.
"""

from __future__ import annotations

import threading
import time


def test_sweeper_fires_at_interval(monkeypatch):
    """The loop calls attach_outcomes_to_index at each tick after warmup."""
    from agentic_investor.experiments.outcome_sweeper import OutcomeSweeper

    calls: list[tuple] = []

    def _fake_sweep():
        calls.append(("swept", time.monotonic()))
        return (2, 1)

    import agentic_investor.memory.outcomes as outcomes_mod
    monkeypatch.setattr(
        outcomes_mod, "attach_outcomes_to_index", _fake_sweep,
    )

    sweeper = OutcomeSweeper(interval_sec=1, warmup_sec=0)
    t = threading.Thread(target=sweeper.run, daemon=True)
    t.start()
    # Enough time for warmup + at least two firings at 1s interval.
    time.sleep(2.5)
    sweeper.stop()
    t.join(timeout=2.0)

    assert len(calls) >= 2


def test_sweeper_error_does_not_kill_loop(monkeypatch):
    """A failing sweep must not stop future sweeps."""
    from agentic_investor.experiments.outcome_sweeper import OutcomeSweeper

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

    sweeper = OutcomeSweeper(interval_sec=1, warmup_sec=0)
    t = threading.Thread(target=sweeper.run, daemon=True)
    t.start()
    time.sleep(2.5)
    sweeper.stop()
    t.join(timeout=2.0)

    # First call raised; subsequent calls still fired.
    assert call_count["n"] >= 2


def test_sweeper_disabled_when_interval_nonpositive():
    """interval_sec <= 0 short-circuits the loop and returns immediately."""
    from agentic_investor.experiments.outcome_sweeper import OutcomeSweeper

    sweeper = OutcomeSweeper(interval_sec=0, warmup_sec=0)
    rc = sweeper.run()
    assert rc == 0
