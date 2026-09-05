"""Cross-arm comparison of an experiment's per-arm SQLite databases.

Reads out/experiments/{name}/{arm}.db and reports:
  - final account equity (from Alpaca via the same-account broker)
  - order count + notional traded
  - realized P/L (from filled buy/sell pairs)
  - LLM cost (rolled up from tick_cost events in the arm's session logs)
  - filter firing rates per arm

Text output. Doesn't attempt statistical tests - N=1 session isn't
enough for a paired test. The point is quick eyeballing during an
experiment and a durable per-arm summary for the notes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _load_arm_orders(db_path: Path) -> list[dict]:
    """Return all paper_orders rows for this arm's DB."""
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT ticker, side, qty, filled_avg_price, filled_at, "
                "submitted_at, status FROM paper_orders "
                "WHERE status IN ('filled', 'partially_filled') "
                "ORDER BY submitted_at"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]


def _load_arm_state(db_path: Path) -> dict:
    """Return the persisted LoopState dict for the primary account_key."""
    if not db_path.exists():
        return {}
    with sqlite3.connect(str(db_path)) as conn:
        try:
            row = conn.execute(
                "SELECT state_json FROM loop_state LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return {}
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


def _summarize_orders(orders: list[dict]) -> dict:
    """Notional traded + rough FIFO realized P/L."""
    from collections import defaultdict

    buys_notional = 0.0
    sells_notional = 0.0
    fifo: dict[str, list[tuple[float, float]]] = defaultdict(list)  # ticker -> [(qty, price)]
    realized = 0.0
    for o in orders:
        qty = float(o["qty"] or 0)
        px = float(o["filled_avg_price"] or 0)
        if not qty or not px:
            continue
        notional = qty * px
        tk = str(o["ticker"]).upper()
        if o["side"] == "buy":
            buys_notional += notional
            fifo[tk].append((qty, px))
        else:
            sells_notional += notional
            remaining = qty
            while remaining > 0 and fifo[tk]:
                bqty, bpx = fifo[tk][0]
                match = min(bqty, remaining)
                realized += match * (px - bpx)
                remaining -= match
                if match >= bqty:
                    fifo[tk].pop(0)
                else:
                    fifo[tk][0] = (bqty - match, bpx)
    return {
        "n_orders": len(orders),
        "buys_notional": round(buys_notional, 2),
        "sells_notional": round(sells_notional, 2),
        "realized_pl": round(realized, 2),
    }


def _scan_session_logs(exp_dir: Path, arm_id: str) -> dict:
    """Aggregate LLM cost + knob firings from this arm's session.jsonl files.

    The arm's paper-loop writes to out/sessions/{ts}/ - we can't reliably
    tag those with arm_id without further plumbing, so we approximate by
    reading the arm's log file for tick_cost lines and knob_fired lines.
    Cheap and works today.
    """
    log_path = exp_dir / f"{arm_id}.log"
    if not log_path.exists():
        return {"cost_usd": 0.0, "n_regens": 0, "knob_fires": {}}
    total = 0.0
    n_regens = 0
    knobs: dict[str, int] = {}
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[tick_cost]" in line or "tick_cost:" in line:
                # crude extraction; format is: cost_usd=$0.0059 anywhere in line
                idx = line.find("cost_usd=$")
                if idx >= 0:
                    val = line[idx + len("cost_usd=$"):].split()[0]
                    try:
                        total += float(val)
                    except ValueError:
                        pass
            if "regen_done" in line:
                n_regens += 1
            if "knob_fired" in line:
                # extract name=X
                idx = line.find("name=")
                if idx >= 0:
                    name = line[idx + 5:].split()[0].rstrip(",")
                    knobs[name] = knobs.get(name, 0) + 1
    return {"cost_usd": round(total, 4), "n_regens": n_regens, "knob_fires": knobs}


def build_ab_report(experiment_name: str) -> str:
    exp_dir = Path("out") / "experiments" / experiment_name
    if not exp_dir.exists():
        return f"no experiment dir: {exp_dir}"
    # news_bus.db + price_bus.db are shared upstream feeds, not arms.
    _SHARED_BUS_FILES = {"news_bus.db", "price_bus.db"}
    arm_dbs = sorted(
        p for p in exp_dir.glob("*.db") if p.name not in _SHARED_BUS_FILES
    )
    if not arm_dbs:
        return f"no arm DBs under {exp_dir}"
    lines = [f"\nExperiment: {experiment_name}", f"Dir: {exp_dir}\n"]
    header = (
        f"{'':<12} {'orders':>7} {'buys $':>10} {'sells $':>10} "
        f"{'realized P/L':>13} {'regens':>7} {'cost $':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    aggregate: dict[str, dict] = {}
    for db_path in arm_dbs:
        arm_id = db_path.stem
        orders = _load_arm_orders(db_path)
        summary = _summarize_orders(orders)
        logs = _scan_session_logs(exp_dir, arm_id)
        row = (
            f"{arm_id:<12} "
            f"{summary['n_orders']:>7} "
            f"{summary['buys_notional']:>10.2f} "
            f"{summary['sells_notional']:>10.2f} "
            f"{summary['realized_pl']:>+13.2f} "
            f"{logs['n_regens']:>7} "
            f"{logs['cost_usd']:>9.4f}"
        )
        lines.append(row)
        aggregate[arm_id] = {"summary": summary, "logs": logs}
    lines.append("")
    # Knob firing breakdown per arm
    all_knobs = sorted({
        k for arm in aggregate.values() for k in arm["logs"]["knob_fires"]
    })
    if all_knobs:
        lines.append("Knob firings per arm:")
        knob_header = f"  {'knob':<28s}" + "".join(
            f" {arm_id:>10s}" for arm_id in aggregate
        )
        lines.append(knob_header)
        for kn in all_knobs:
            line = f"  {kn:<28s}"
            for data in aggregate.values():
                line += f" {data['logs']['knob_fires'].get(kn, 0):>10}"
            lines.append(line)
    return "\n".join(lines)
