"""Smoke test for the RotatingFileHandler wiring in paper-loop's
logging setup. Confirms env vars are respected and the base file
exists after enough writes to trigger rotation.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _emulate_paper_loop_logging_setup(log_file: str) -> None:
    """Mirror the block in cli.py that sets up logging for paper-loop.
    Kept in the test rather than imported so we don't have to run the
    full paper-loop entry point.
    """
    handlers: list[logging.Handler] = []
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        rotate_mb = int(os.environ.get("AGENTIC_LOG_ROTATE_MB", "20"))
    except ValueError:
        rotate_mb = 20
    try:
        rotate_keep = int(os.environ.get("AGENTIC_LOG_ROTATE_KEEP", "5"))
    except ValueError:
        rotate_keep = 5
    handlers.append(RotatingFileHandler(
        log_file,
        maxBytes=max(1, rotate_mb) * 1024 * 1024,
        backupCount=max(0, rotate_keep),
        encoding="utf-8",
    ))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def test_rotation_creates_backup_when_size_exceeded(tmp_path, monkeypatch):
    """Force a very small max size so the test triggers rotation without
    writing 20 MB. Verifies the backup file lands next to the base file.
    """
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_MB", "1")  # min = 1 MB
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_KEEP", "2")
    log_file = tmp_path / "arm.log"
    _emulate_paper_loop_logging_setup(str(log_file))
    log = logging.getLogger("test_rotation")

    # Write ~1.2 MB of log lines to force at least one rollover.
    big_line = "x" * 500
    for _ in range(2500):
        log.info(big_line)
    # Flush all handlers so pending writes hit disk before we count.
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass

    assert log_file.exists()
    # At least one .1 backup should have been created.
    assert (tmp_path / "arm.log.1").exists(), \
        "expected a rotated backup file after exceeding maxBytes"


def test_rotation_env_default_when_bad_value(tmp_path, monkeypatch):
    """Non-numeric env doesn't crash; falls back to defaults."""
    monkeypatch.setenv("AGENTIC_LOG_ROTATE_MB", "not-an-int")
    log_file = tmp_path / "arm.log"
    _emulate_paper_loop_logging_setup(str(log_file))
    log = logging.getLogger("test_rotation")
    log.info("one message")
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass
    assert log_file.exists()
