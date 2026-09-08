"""Pre-flight healthcheck for paper-experiment launches."""

from __future__ import annotations

import socket

from agentic_investor.ops.preflight import (
    check_arm_db_writable,
    check_arm_state_resumable,
    check_experiment_dir,
    check_port_available,
)


def test_port_available_when_free():
    # High ports unlikely to be bound.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    # After close, port frees; brief re-check.
    name, ok, detail = check_port_available(free_port)
    assert name == f"port:{free_port}"
    assert ok is True
    assert "available" in detail


def test_port_available_flags_bound_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        bound = s.getsockname()[1]
        name, ok, detail = check_port_available(bound)
    assert ok is False
    assert "in use" in detail


def test_experiment_dir_missing_is_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name, ok, detail = check_experiment_dir("nonexistent-exp")
    assert ok is True
    assert "clean" in detail


def test_experiment_dir_with_bus_leftovers_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp_dir = tmp_path / "out" / "experiments" / "leftover"
    exp_dir.mkdir(parents=True)
    (exp_dir / "news_bus.db").touch()
    (exp_dir / "price_bus.db").touch()
    (exp_dir / "A.log").touch()
    name, ok, detail = check_experiment_dir("leftover")
    assert ok is True


def test_experiment_dir_with_arm_dbs_is_ok_resume_semantics(tmp_path, monkeypatch):
    # Post-M13 restart policy: existing arm DBs are RESUME state, not stale.
    # check_experiment_dir passes; check_arm_state_resumable surfaces the
    # loud "will resume" banner separately.
    monkeypatch.chdir(tmp_path)
    exp_dir = tmp_path / "out" / "experiments" / "resume"
    exp_dir.mkdir(parents=True)
    (exp_dir / "A.db").touch()
    (exp_dir / "B.db").touch()
    name, ok, detail = check_experiment_dir("resume")
    assert ok is True
    assert "resume" in detail.lower() or "exists" in detail.lower()


def test_arm_state_resumable_no_db_reports_fresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = check_arm_state_resumable("brand-new", ["A", "B"])
    assert len(results) == 2
    for _, ok, detail in results:
        assert ok is True
        assert "fresh" in detail


def test_arm_state_resumable_populated_db_reports_will_resume(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.chdir(tmp_path)
    exp_dir = tmp_path / "out" / "experiments" / "populated"
    exp_dir.mkdir(parents=True)
    conn = sqlite3.connect(exp_dir / "A.db")
    conn.execute(
        "CREATE TABLE loop_state (account_key TEXT PRIMARY KEY, payload_json TEXT)"
    )
    conn.execute(
        "INSERT INTO loop_state VALUES ('default', '{}')"
    )
    conn.commit()
    conn.close()
    results = check_arm_state_resumable("populated", ["A"])
    assert len(results) == 1
    _, ok, detail = results[0]
    # ok stays True - this is a warning banner, not a failure.
    assert ok is True
    assert "WILL RESUME" in detail


def test_arm_db_writable_reports_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = check_arm_db_writable("smoke", ["A", "B"])
    assert len(results) == 2
    for _, ok, _ in results:
        assert ok is True
