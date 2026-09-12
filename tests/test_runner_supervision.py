"""Supervision + tee-log + healthcheck plumbing on the paper-experiment runner.

Covers priorities 2 (restart-supervisor loop), 3 (TeeStream stdout capture),
and 4 (outcome-sweeper startup healthcheck) from the silent-cascade fix.
"""

from __future__ import annotations

import io
import subprocess
import threading
import time

import pytest


# -----------------------------
# Priority 3: TeeStream
# -----------------------------


def test_tee_stream_writes_to_all_underlying_streams():
    from agentic_investor.experiments.runner import _TeeStream

    a = io.StringIO()
    b = io.StringIO()
    tee = _TeeStream(a, b)
    tee.write("hello\n")
    tee.flush()
    assert a.getvalue() == "hello\n"
    assert b.getvalue() == "hello\n"


def test_tee_stream_survives_a_broken_pipe_on_one_stream():
    """If one stream's write raises, the other still gets the message."""
    from agentic_investor.experiments.runner import _TeeStream

    class Broken:
        def write(self, _t):
            raise BrokenPipeError("shell closed the pipe")

        def flush(self):
            raise BrokenPipeError("shell closed the pipe")

    good = io.StringIO()
    tee = _TeeStream(Broken(), good)
    tee.write("survived\n")
    tee.flush()
    assert good.getvalue() == "survived\n"


# -----------------------------
# Priority 2: _supervise
# -----------------------------


class _FakeProc:
    """subprocess.Popen stub for supervision tests.

    - poll_returns is a list of values yielded on successive .poll() calls.
      None means "still running". A number means "exited with that rc".
      Test drives lifecycle by populating this list.
    """

    def __init__(self, poll_returns, stdout_bytes=b""):
        self._poll_returns = list(poll_returns)
        self._last_rc = None
        self.stdout = io.BytesIO(stdout_bytes)
        self.signal_sent = None
        self.terminated = False

    def poll(self):
        if self._poll_returns:
            self._last_rc = self._poll_returns.pop(0)
        return self._last_rc

    def wait(self, timeout=None):
        # Return whatever the last poll would have; used by drain phase.
        while self.poll() is None:
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self._last_rc

    def send_signal(self, sig):
        self.signal_sent = sig

    def terminate(self):
        self.terminated = True


def test_supervise_respawns_a_crashed_child(monkeypatch):
    from agentic_investor.experiments import runner as rmod

    # Initial proc: reports rc=42 immediately (already crashed).
    initial = _FakeProc(poll_returns=[42])
    # Respawn proc: stays "running" forever until shutdown.
    respawn = _FakeProc(poll_returns=[None, None, None, None, None])

    spawn_calls = []

    def fake_popen(cmd, env, stdout=None, stderr=None, bufsize=None):
        spawn_calls.append(cmd)
        return respawn

    monkeypatch.setattr(rmod.subprocess, "Popen", fake_popen)

    procs = {
        "victim": rmod._ProcSpec(
            name="victim", cmd=["dummy"], env={}, proc=initial,
        ),
    }
    shutdown = threading.Event()

    def _flag_shutdown_after(sec):
        time.sleep(sec)
        shutdown.set()

    threading.Thread(target=_flag_shutdown_after, args=(0.3,), daemon=True).start()

    rc = rmod._supervise(
        procs, threads=[], max_restarts=5,
        shutdown_flag=shutdown, poll_sec=0,
    )

    # A respawn was attempted exactly once (initial was crashed).
    assert len(spawn_calls) == 1
    # The spec's restarts counter incremented.
    # (procs["victim"] may have been drained; use captured spec object).
    assert rc == 42  # aggregate rc records the crash


def test_supervise_gives_up_after_max_restarts(monkeypatch):
    from agentic_investor.experiments import runner as rmod

    # Initial proc crashed.
    initial = _FakeProc(poll_returns=[9])

    spawn_calls = []

    def fake_popen(cmd, env, stdout=None, stderr=None, bufsize=None):
        spawn_calls.append(cmd)
        # Every respawn immediately crashes too.
        return _FakeProc(poll_returns=[9])

    monkeypatch.setattr(rmod.subprocess, "Popen", fake_popen)

    procs = {
        "flappy": rmod._ProcSpec(
            name="flappy", cmd=["dummy"], env={}, proc=initial,
        ),
    }
    shutdown = threading.Event()

    rc = rmod._supervise(
        procs, threads=[], max_restarts=3,
        shutdown_flag=shutdown, poll_sec=0,
    )

    # Initial crash + 3 respawns that also crash = 3 Popen calls.
    assert len(spawn_calls) == 3
    # After budget exhausted, spec is removed from procs.
    assert "flappy" not in procs
    assert rc == 9


def test_supervise_clean_exit_does_not_respawn(monkeypatch):
    from agentic_investor.experiments import runner as rmod

    proc = _FakeProc(poll_returns=[0])
    spawn_calls = []

    def fake_popen(cmd, env, stdout=None, stderr=None, bufsize=None):
        spawn_calls.append(cmd)
        return _FakeProc(poll_returns=[0])

    monkeypatch.setattr(rmod.subprocess, "Popen", fake_popen)

    procs = {
        "clean": rmod._ProcSpec(
            name="clean", cmd=["dummy"], env={}, proc=proc,
        ),
    }
    rc = rmod._supervise(
        procs, threads=[], max_restarts=5,
        shutdown_flag=threading.Event(), poll_sec=0,
    )

    # rc=0 must never trigger a respawn.
    assert spawn_calls == []
    assert rc == 0


# -----------------------------
# Priority 4: healthcheck
# -----------------------------


def test_healthcheck_returns_true_when_sync_sweep_succeeds(monkeypatch):
    from agentic_investor.experiments import runner as rmod

    def fake_run(cmd, capture_output=None, text=None, timeout=None):
        class R:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return R()

    monkeypatch.setattr(rmod.subprocess, "run", fake_run)
    assert rmod._healthcheck_outcome_sweeper() is True


def test_healthcheck_returns_false_when_sync_sweep_crashes(monkeypatch):
    from agentic_investor.experiments import runner as rmod

    def fake_run(cmd, capture_output=None, text=None, timeout=None):
        class R:
            returncode = 1
            stdout = ""
            stderr = "chroma corruption"
        return R()

    monkeypatch.setattr(rmod.subprocess, "run", fake_run)
    assert rmod._healthcheck_outcome_sweeper() is False


def test_healthcheck_returns_false_on_timeout(monkeypatch):
    from agentic_investor.experiments import runner as rmod

    def fake_run(cmd, capture_output=None, text=None, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(rmod.subprocess, "run", fake_run)
    assert rmod._healthcheck_outcome_sweeper(timeout_sec=5) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
