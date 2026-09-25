"""Opening-intent block: pre-market plans held for the 9:30 fresh regen.

Pre-market regens produce trade plans but hold them (pre_market_hold
in loop.py:2282) because the market isn't open yet. Without this
block those plans die at 9:30 - 30 min of LLM work wasted every
morning, and any news signal absorbed pre-market that the fresh
9:30 regen doesn't independently rediscover is lost.

This block reads today's pre_market_hold events from the arm's own
session.jsonl and renders them as an "opening intent" prompt
section. The LLM sees "here's what you wanted to do before open,
now that you have real open prices, do you still want to?" and can
honor, adjust, or reject. No forcing - the LLM stays in the
driver's seat.

Design decisions:
- Only renders when today has pre_market_hold events. Post-open
  sessions with no pre-market work get "".
- Reads across all of today's session dirs for the arm (arms can
  restart mid-day; every restart makes a new session dir).
- Deduplicates by (ts, tickers-tuple) so a repeated hold event
  from the same regen doesn't get double-counted.
- Truncates to the 3 most recent hold events - older intents get
  stale as prices move.
- Guarded with try/except so a prompt-block bug never breaks the
  regen path.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def _today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _load_today_holds(arm_id: str, today: str) -> list[dict]:
    """Every pre_market_hold event today for this arm, sorted oldest
    first. Empty list if none - caller renders "" in that case.
    """
    holds: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for d in sorted(glob.glob(f"out/sessions/{today}T*_{arm_id}/")):
        path = os.path.join(d, "session.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") != "pre_market_hold":
                    continue
                plans = r.get("plans") or []
                if not plans:
                    continue
                # Dedup key: (ts, sorted tickers) to survive relaunches
                tkey = ",".join(sorted(
                    str(p.get("ticker", "")).upper() for p in plans
                ))
                key = (r.get("ts", ""), tkey)
                if key in seen:
                    continue
                seen.add(key)
                holds.append(r)
    holds.sort(key=lambda r: r.get("ts", ""))
    return holds


def build_opening_intent_block() -> str:
    """Prompt section listing today's pre-market held plans.

    Returns "" if no pre_market_hold events landed today for this
    arm - so post-open regens on a normally-run session get nothing
    extra, and if the arm launched post-open there's simply nothing
    to render.

    Arm identity comes from AGENTIC_ARM_ID (set by the supervisor
    when spawning arms). Solo runs default to "solo" which will
    match session dirs tagged the same way.
    """
    try:
        arm_id = os.environ.get("AGENTIC_ARM_ID") or "solo"
        holds = _load_today_holds(arm_id, _today_utc())
        if not holds:
            return ""
        # Keep the 3 most recent holds: older ones go stale as
        # pre-market prices move, and cramming 10 stale intents
        # into the prompt costs tokens for no useful signal.
        recent = holds[-3:]
        lines = [
            "## 13. Opening intent (plans held during pre-market)",
            "You proposed these trades during the pre-market lead "
            "window when the market was closed. Now that you have "
            "real open-print prices, decide whether each still "
            "makes sense - honor, adjust, or reject. Fresh news "
            "in the batch above supersedes stale intent.",
        ]
        for h in recent:
            ts = str(h.get("ts", ""))[:19]
            rec_id = h.get("rec_id")
            plans = h.get("plans") or []
            lines.append(f"- rec {rec_id} @ {ts}:")
            for p in plans[:8]:  # cap plans per hold too
                tk = str(p.get("ticker", "?")).upper()
                side = str(p.get("side", "?")).upper()
                qty = p.get("qty", "?")
                lines.append(f"  - {side} {qty} {tk}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        logger.debug("opening-intent block skipped: %s", e)
        return ""
