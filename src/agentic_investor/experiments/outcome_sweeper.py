"""Standalone outcome-sweeper subprocess for paper-experiment.

Refreshes multi-horizon P/L on every rec in the sqlite-vec rec store
(post-#146 - was chromadb pre-migration). Isolated as a subprocess so
a store crash kills only the sweeper rather than the supervisor + all
its orphaned arms. Runner spawns this the same way it spawns news-bus
/ price-bus and monitors its exit alongside the others.
"""

from __future__ import annotations

import logging
import signal
import threading

logger = logging.getLogger(__name__)

_DEFAULT_WARMUP_SEC = 60


class OutcomeSweeper:
    """Loop that calls attach_outcomes_to_index every interval_sec.

    Own class (not just a function) so tests can drive stop() directly
    without racing on signal delivery.
    """

    def __init__(self, interval_sec: int, warmup_sec: int = _DEFAULT_WARMUP_SEC):
        self.interval_sec = interval_sec
        self.warmup_sec = warmup_sec
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> int:
        if self.interval_sec <= 0:
            logger.info(
                "outcome sweeper disabled (interval_sec=%d)", self.interval_sec,
            )
            return 0
        logger.info(
            "outcome sweeper starting; warmup=%ds interval=%ds",
            self.warmup_sec, self.interval_sec,
        )
        # Warmup so arms have written at least one rec before first sweep.
        if self._stop.wait(self.warmup_sec):
            return 0
        while not self._stop.is_set():
            try:
                from agentic_investor.memory.outcomes import (
                    attach_outcomes_to_index,
                )
                n_updated, n_with = attach_outcomes_to_index()
                logger.info(
                    "refreshed %d recs, %d have at least one outcome",
                    n_updated, n_with,
                )
            except Exception as e:  # noqa: BLE001 - transient chroma errors ok
                logger.warning("sweep error: %s", e)
            if self._stop.wait(self.interval_sec):
                return 0
        return 0


def run_outcome_sweeper(interval_min: int) -> int:
    """CLI entry point: convert minutes -> seconds, wire signals, run.

    Blocks until SIGINT / SIGTERM. Returns the sweeper's exit code (0
    on graceful shutdown).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    sweeper = OutcomeSweeper(interval_sec=int(interval_min) * 60)

    def _shutdown(_sig=None, _frame=None):
        sweeper.stop()

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, AttributeError):
        # SIGTERM not settable on Windows in some contexts; SIGINT covers
        # the normal Ctrl+C path.
        pass
    return sweeper.run()
