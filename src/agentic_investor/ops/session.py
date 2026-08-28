"""Session recorder for live paper-trading runs.

Every event during a run - news arrivals, decision moments, LLM calls, trades,
snapshots - lands in two places:

- Human-readable console line (via logger) so you can watch the run live
- Structured JSONL row in out/sessions/<start>/session.jsonl for post-market
  analysis (grep, jq, pandas)

A markdown summary is generated on shutdown with counts, trade log, and P&L
curve pointers. All timestamps are UTC ISO 8601.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("session")


@dataclass
class SessionRecorder:
    """One recorder per paper-loop run. Threadsafe via a single lock."""

    out_dir: Path
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def start(cls, base_dir: str = "out/sessions") -> SessionRecorder:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        out = Path(base_dir) / stamp
        out.mkdir(parents=True, exist_ok=True)
        rec = cls(out_dir=out)
        rec.log("session_start", {"out_dir": str(out)})
        return rec

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / "session.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "SUMMARY.md"

    def log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """Append a single event to jsonl + emit a pretty console line +
        publish to the dashboard event bus for live WebSocket clients."""
        payload = payload or {}
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        with self._lock:
            self._counts[event] = self._counts.get(event, 0) + 1
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        logger.info("[%s] %s", event, _pretty(payload))
        # Dashboard fanout (no-op when the dashboard server isn't running).
        try:
            from agentic_investor.dashboard.events import get_bus
            get_bus().publish(row)
        except Exception:  # noqa: BLE001 - never let telemetry break the loop
            pass

    def summary_lines(self) -> list[str]:
        lines = [f"# Session {self.started_at}", ""]
        lines.append(f"Output: `{self.out_dir}`  ")
        lines.append(f"JSONL: `{self.jsonl_path.name}`  ")
        lines.append("")
        lines.append("## Event counts")
        for k, v in sorted(self._counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{k}**: {v}")
        return lines

    def finalize(self) -> Path:
        """Write the markdown summary. Called on graceful shutdown."""
        self.summary_path.write_text("\n".join(self.summary_lines()), encoding="utf-8")
        self.log("session_end", {"summary": str(self.summary_path)})
        return self.summary_path


def _pretty(payload: dict[str, Any]) -> str:
    """One-line render of a payload for the console."""
    if not payload:
        return ""
    parts = []
    for k, v in payload.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        elif isinstance(v, list | tuple):
            parts.append(f"{k}=[{len(v)}]")
        elif isinstance(v, dict):
            parts.append(f"{k}={{{len(v)}}}")
        else:
            s = str(v)
            parts.append(f"{k}={s[:80]}" + ("..." if len(s) > 80 else ""))
    return " ".join(parts)
