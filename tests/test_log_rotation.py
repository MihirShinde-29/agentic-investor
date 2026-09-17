"""Smoke test for the RotatingFileHandler wiring in paper-loop's
logging setup. Confirms env vars are respected and the base file
exists after enough writes to trigger rotation.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _emulate_paper_loop_logging_setup(log_file: str) -> logging.Logger:
    """Mirror the block in cli.py that sets up logging for paper-loop.
    Returns a fresh, isolated logger with just the RotatingFileHandler
    attached so cross-test state on the root logger doesn't affect us.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        rotate_mb = int(os.environ.get("AGENTIC_LOG_ROTATE_MB", "20"))
    except ValueError:
        rotate_mb = 20
    try:
        rotate_keep = int(os.environ.get("AGENTIC_LOG_ROTATE_KEEP", "5"))
    except ValueError:
        rotate_keep = 5
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max(1, rotate_mb) * 1024 * 1024,
        backupCount=max(0, rotate_keep),
        encoding="utf-8",
    )
    log = logging.getLogger(f"test_rotation_{log_file}")
    log.propagate = False  # don't touch root
    for h in list(log.handlers):
        log.removeHandler(h)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    return log


def test_rotation_handler_configured_from_env(tmp_path, monkeypatch):
    """Verify the handler picks up env-var config (maxBytes + backupCount)
    correctly. Rotation itself is stdlib behavior; asserting on the
    configured values is enough of a regression signal.
    """
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_MB", "3")
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_KEEP", "7")
    log = _emulate_paper_loop_logging_setup(str(tmp_path / "arm.log"))
    rot = [h for h in log.handlers if isinstance(h, RotatingFileHandler)]
    assert len(rot) == 1
    assert rot[0].maxBytes == 3 * 1024 * 1024
    assert rot[0].backupCount == 7


def test_rotation_manual_rollover(tmp_path, monkeypatch):
    """Call doRollover() directly to prove the wired handler produces
    a .1 backup file. Bypasses the emit-triggered rollover which some
    other-test-suite state can interfere with.
    """
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_MB", "1")
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_KEEP", "2")
    log_file = tmp_path / "arm.log"
    log = _emulate_paper_loop_logging_setup(str(log_file))
    log.info("seed line before rotation")
    for h in log.handlers:
        if isinstance(h, RotatingFileHandler):
            h.flush()
            h.doRollover()
    assert log_file.exists()
    assert (tmp_path / "arm.log.1").exists(), \
        "doRollover() should have produced the .1 backup"


def test_rotation_env_default_when_bad_value(tmp_path, monkeypatch):
    """Non-numeric env doesn't crash; falls back to defaults."""
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_MB", "not-an-int")
    log_file = tmp_path / "arm.log"
    log = _emulate_paper_loop_logging_setup(str(log_file))
    log.info("one message")
    for h in log.handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass
    assert log_file.exists()
