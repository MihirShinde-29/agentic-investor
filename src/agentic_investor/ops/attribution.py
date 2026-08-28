"""Rolling attribution analysis across paper-loop sessions.

Walks every session directory in `out/sessions/`, extracts the events we care
about (regens, skips, trades, tick_costs), and produces per-day metrics that
let us judge whether our discipline filters are actually helping.

Not a backtest - a filter-behavior report. Backtest attribution needs
would-be-allocation logging which is a follow-up (see CHANGE_LOG for plan).

Output: `out/attribution/rolling_report.md` with per-day rows for:

- decision moments (news / price / force / interval)
- filter skips (barely-moved vs aggregate-too-large)
- trades submitted + estimated turnover
- LLM cost
- session P&L (from snapshots first vs last)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SessionMetrics:
    session_id: str
    date: str
    started_at: str | None
    ended_at: str | None
    equity_open: float = 0.0
    equity_close: float = 0.0
    regens_by_trigger: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    skips_by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    trades_submitted: int = 0
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    news_events: int = 0
    total_turnover_pp: float = 0.0  # sum of aggregate turnover across all fires
    skipped_turnover_pp: float = 0.0  # sum of turnover we AVOIDED via skips

    @property
    def return_pct(self) -> float:
        if self.equity_open <= 0:
            return 0.0
        return (self.equity_close / self.equity_open - 1) * 100

    @property
    def total_decision_moments(self) -> int:
        return sum(self.regens_by_trigger.values()) + sum(self.skips_by_reason.values())


def _load_events(session_dir: Path) -> list[dict]:
    jsonl = session_dir / "session.jsonl"
    if not jsonl.exists():
        return []
    events = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _summarize_session(session_dir: Path) -> SessionMetrics:
    events = _load_events(session_dir)
    if not events:
        return SessionMetrics(session_id=session_dir.name, date="", started_at=None, ended_at=None)

    started = next(
        (e["ts"] for e in events if e["event"] == "session_start"),
        events[0]["ts"],
    )
    ended = next(
        (e["ts"] for e in reversed(events) if e["event"] == "session_end"),
        events[-1]["ts"],
    )
    m = SessionMetrics(
        session_id=session_dir.name, date=started[:10],
        started_at=started, ended_at=ended,
    )

    # Pull equity from paper_snapshots via the same store that recorded them.
    try:
        from agentic_investor.tools.paper_store import list_snapshots

        started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        ended_dt = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        snaps = list_snapshots(limit=1000)
        window = []
        for s in snaps:
            ts = datetime.fromisoformat(s["captured_at"].replace("Z", "+00:00"))
            if started_dt <= ts <= ended_dt:
                window.append(s)
        window.sort(key=lambda s: s["captured_at"])
        if window:
            m.equity_open = float(window[0]["account"]["equity"])
            m.equity_close = float(window[-1]["account"]["equity"])
    except Exception:  # noqa: BLE001
        pass

    for e in events:
        et = e["event"]
        if et == "regen_done":
            trigger = e.get("trigger", "unknown")
            m.regens_by_trigger[trigger] += 1
        elif et == "opinion_drift_skip":
            skip = e.get("skip_reason", "unknown")
            m.skips_by_reason[skip] += 1
            m.skipped_turnover_pp += float(e.get("aggregate_turnover_pp", 0) or 0)
        elif et == "trade_plan":
            trades = e.get("trades", [])
            m.trades_submitted += len(trades) if isinstance(trades, list) else 0
        elif et == "order_submitted":
            # trade_plan already counts; skip to avoid double-counting.
            pass
        elif et == "tick_cost":
            m.llm_calls += int(e.get("llm_calls", 0) or 0)
            cost = e.get("cost_usd", 0)
            if isinstance(cost, str):
                cost = float(cost.lstrip("$"))
            m.llm_cost_usd += float(cost or 0)
        elif et == "news_received":
            m.news_events += 1
    return m


def _render_would_be_section() -> list[str]:
    """0f attribution: sample of recorded filter skips with would-be
    allocations for later counterfactual simulation.
    """
    from agentic_investor.tools.paper_store import list_filter_skips

    lines: list[str] = ["", "## Filter-skip attribution log", ""]
    skips = list_filter_skips(limit=20)
    if not skips:
        lines.append("_no filter skips recorded yet_")
        return lines
    lines.append(
        f"**{len(skips)} recent skips logged with would-be allocations.**  "
    )
    lines.append(
        "Each row is a rebalance we SKIPPED. would_be_allocation_json holds "
        "the LLM's proposed target weights; actual_positions_json holds what "
        "we kept. Compare P&L over the next N minutes to measure filter "
        "false-positive rate."
    )
    lines.append("")
    lines.append(
        "| Time | Reason | Trigger | avg drift | max delta | ticker |"
    )
    lines.append("|---|---|---|---:|---:|---|")
    for s in skips[:10]:
        avg = s.get("avg_drift_pp") or 0
        mx = s.get("max_delta_pp") or 0
        lines.append(
            f"| {s['skipped_at'][11:19]} | {s['skip_reason']} | "
            f"{s.get('trigger_reason','?')} | {avg:.2f}pp | {mx:.2f}pp | "
            f"{s.get('max_delta_ticker','-')} |"
        )
    return lines


def _render_report(session_metrics: list[SessionMetrics]) -> str:
    lines: list[str] = ["# Rolling Attribution Report", ""]
    if not session_metrics:
        lines.append("_no sessions found_")
        return "\n".join(lines)

    # Sort newest-first for the header table.
    ordered = sorted(session_metrics, key=lambda m: m.started_at or "", reverse=True)

    lines.append(f"Sessions analyzed: **{len(ordered)}**  ")
    lines.append(f"Date range: `{ordered[-1].date}` → `{ordered[0].date}`  ")
    lines.append("")
    lines.append("## Per-session metrics")
    lines.append("")
    lines.append(
        "| Session | Return% | Regens | Skips | Trades | Turnover skipped | LLM cost | News evts |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in ordered:
        regens = sum(m.regens_by_trigger.values())
        skips = sum(m.skips_by_reason.values())
        lines.append(
            f"| `{m.session_id}` | {m.return_pct:+.2f}% | {regens} | "
            f"{skips} | {m.trades_submitted} | {m.skipped_turnover_pp:.0f}pp | "
            f"${m.llm_cost_usd:.4f} | {m.news_events} |"
        )
    lines.append("")

    lines.append("## Aggregate over all sessions")
    lines.append("")
    total_regens_by_trigger: dict[str, int] = defaultdict(int)
    total_skips_by_reason: dict[str, int] = defaultdict(int)
    for m in ordered:
        for k, v in m.regens_by_trigger.items():
            total_regens_by_trigger[k] += v
        for k, v in m.skips_by_reason.items():
            total_skips_by_reason[k] += v
    total_trades = sum(m.trades_submitted for m in ordered)
    total_cost = sum(m.llm_cost_usd for m in ordered)
    total_skipped_pp = sum(m.skipped_turnover_pp for m in ordered)

    lines.append("### Regens by trigger")
    for k, v in sorted(total_regens_by_trigger.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("### Skips by reason (filter fires)")
    if total_skips_by_reason:
        for k, v in sorted(total_skips_by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("_no skips recorded_")
    lines.append("")
    lines.append("### Totals")
    lines.append(f"- Trades submitted: **{total_trades}**")
    lines.append(f"- LLM cost: **${total_cost:.4f}**")
    lines.append(f"- Turnover avoided by skips: **{total_skipped_pp:.0f}pp**")
    lines.append(
        f"- Est. friction savings @ 1bp spread: **${total_skipped_pp / 100 * 100:.2f}** "
        "(assumes account = $100 per pp of turnover)"
    )
    lines.append("")
    lines.append("## Interpretation notes (honesty)")
    lines.append("")
    lines.append(
        "- **Skip counts are one-sided**: we know we skipped a rebalance, but not "
        "whether skipping was RIGHT. Next feature: log would-be allocations on "
        "skips + simulate their outcome over next N minutes."
    )
    lines.append(
        "- **Cost savings are lower-bound**: assumes 1bp spread. Real friction is "
        "spread + slippage + market impact, which can be 2-5bps on less liquid "
        "or larger orders."
    )
    lines.append(
        "- **P&L is NOT alpha**: session return includes market movement, not "
        "just our decisions. To attribute alpha, compare to SPY/benchmark over "
        "same window."
    )
    lines.append(
        "- **Parameters are hand-tuned**: 3pp opinion drift, 25pp aggregate cap, "
        "2% price move, 30min force regen were picked by intuition from today's "
        "observed failures. Backtest optimization is a M13-adjacent milestone."
    )
    lines.extend(_render_would_be_section())
    return "\n".join(lines)


def build_rolling_report(
    sessions_dir: str = "out/sessions",
    out_path: str = "out/attribution/rolling_report.md",
) -> Path:
    """Aggregate all session artifacts into a rolling markdown report."""
    root = Path(sessions_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    session_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "session.jsonl").exists()]
    metrics = [_summarize_session(p) for p in session_dirs]
    metrics = [m for m in metrics if m.total_decision_moments > 0 or m.trades_submitted > 0]
    body = _render_report(metrics)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out
