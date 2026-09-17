"""Session recorder for live paper-trading runs.

Every event during a run - news arrivals, decision moments, LLM calls, trades,
snapshots - lands in two places:

- Human-readable console line (via logger) so you can watch the run live
- Structured JSONL row in out/sessions/<start>/session.jsonl for post-market
  analysis (grep, jq, pandas)

Each JSONL row has:
  ts        ISO-8601 UTC timestamp
  event     short name (e.g. "order_submitted", "decision_moment")
  arm_id    optional; set when AGENTIC_ARM_ID env is populated
  ...       payload fields flattened at top level

Payloads pass through `_safe_json_payload` so Pydantic models get
`.model_dump()`d (not `str()`d, which was the earlier behavior via
json.dumps(default=str)), Path becomes its string form, datetime
becomes ISO. Anything unhandled falls back to str() so telemetry
never blocks the loop.

Analysis: use `iter_events(session_dir, ...)` from Python, or the
provided jq examples in docs/INTERVIEW_NOTES.md B14 from a shell.

A markdown summary is generated on shutdown with counts, trade log, and P&L
curve pointers. All timestamps are UTC ISO 8601.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger("session")

# Tagging events with the arm_id lets the dashboard-subprocess (which
# doesn't share the in-process event bus with the arm subprocess) locate
# the right session.jsonl per arm and stream events to the frontend.
#
# NOT going through flags.ARM_ID here: that resolves to "solo" when
# AGENTIC_ARM_ID is unset, but for session tagging we want "unset" to
# stay untagged so dashboard listeners distinguish experiment-arm
# events from legacy single-arm ones.
_ARM_ID = os.environ.get("AGENTIC_ARM_ID")


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
        # Append arm_id so two arm subprocesses starting the same clock
        # second get separate session dirs. Silent shared-jsonl bug otherwise.
        if _ARM_ID:
            stamp = f"{stamp}_{_ARM_ID}"
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
        publish to the dashboard event bus for live WebSocket clients.

        Payload runs through `_safe_json_payload` so Pydantic models,
        Path, datetime, Decimal, set, tuple all serialize cleanly.
        Unhandled types fall back to `str()` (via json.dumps' default)
        so the loop never dies on telemetry.
        """
        payload = _safe_json_payload(payload or {})
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        if _ARM_ID:
            row["arm_id"] = _ARM_ID
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


def _safe_json_payload(value: Any) -> Any:
    """Recursively coerce `value` into a form `json.dumps` can serialize
    without falling back to `str()`.

    - Pydantic BaseModel  -> model_dump() (via `.model_dump()` when
      present, so nested submodels also unwrap cleanly)
    - Path                -> str(path)
    - datetime / date     -> ISO 8601 string
    - Decimal             -> float
    - set / tuple / frozenset -> list of sanitized items
    - dict                -> new dict with sanitized values (keys must be
                             str/int; anything else becomes str())
    - list                -> list of sanitized items
    - everything else     -> pass through; json.dumps(default=str) still
                             catches truly-weird types without raising

    Never raises: telemetry must not take down the loop.
    """
    try:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            # Pydantic v2 model. `mode="json"` returns JSON-safe primitives
            # (ISO datetime, etc.) so we don't re-coerce below.
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                key = k if isinstance(k, (str, int)) else str(k)
                out[key] = _safe_json_payload(v)
            return out
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_safe_json_payload(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        return value
    except Exception:  # noqa: BLE001 - telemetry never blocks the loop
        return str(value)


def iter_events(
    session_dir: Path | str,
    *,
    event_types: list[str] | None = None,
    since_ts: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield each event row from `session_dir/session.jsonl`.

    - `event_types`: if given, only rows whose `event` is in the set
      are yielded. Fast pre-filter to avoid decoding every payload.
    - `since_ts`: ISO 8601 lower bound; rows with `ts < since_ts` are
      skipped. String comparison is fine because ISO 8601 sorts
      lexicographically the way you'd want.

    Malformed JSON lines are logged at DEBUG and skipped; a partial
    write at the tail of a still-open jsonl (last line missing a
    newline) will just get skipped, matching how `jq` handles it.
    """
    path = Path(session_dir)
    if path.is_dir():
        path = path / "session.jsonl"
    if not path.exists():
        return
    type_set = set(event_types) if event_types else None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.debug("skipping malformed jsonl line: %s", e)
                continue
            if since_ts is not None:
                ts = row.get("ts")
                if ts is not None and ts < since_ts:
                    continue
            if type_set is not None and row.get("event") not in type_set:
                continue
            yield row
