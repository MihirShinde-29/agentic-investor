"""End-of-day shutdown + consolidated report.

Runs after market close. One command collapses the manual sequence
I was doing nightly:
  1. Pull Alpaca equity + positions snapshot
  2. Run close_postmortem.py
  3. Run gate_divergences.py --with-forward-pnl
  4. Run ab_significance.py --since <N>
  5. Scan today's experiment.out for issues (ERRORs, memory_recycles,
     websocket errors, Jev tail spikes, force_regens)
  6. Write everything to out/eod/<date>_eod.md
  7. Optionally kill the supervisor tree and rotate the log

Usage:
    python scripts/eod_summary.py                   # dry report, no kill
    python scripts/eod_summary.py --shutdown        # also kill + rotate
    python scripts/eod_summary.py --since 2026-09-22   # significance window
    python scripts/eod_summary.py --date 2026-10-01    # back-run a day

The report is append-safe (overwrites per date) and markdown-formatted
for pasting into INTERVIEW_NOTES.md later.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "out" / "logs" / "experiment.out"
OUT_DIR = REPO / "out" / "eod"
POSTMORTEM_DIR = REPO / "out" / "postmortems"
ANALYTICS_DIR = REPO / "out" / "analytics"

# Thresholds for the issues scanner. Baseline Jev latency is ~250 ms
# per-headline; anything 2x+ is a tail. memory_recycle events are
# unconditionally flagged - they mean the supervisor killed an arm.
JEV_TAIL_MIN_MS = 500
LAYA_TAIL_MIN_MS = 500
MEMORY_FLAG_MB_FROM_THRESHOLD = 200  # within 200 MB of threshold


def _run(cmd: list[str], tag: str) -> tuple[int, str]:
    """Run a subprocess, return (rc, combined stdout+stderr)."""
    try:
        result = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        return result.returncode, combined
    except Exception as exc:  # noqa: BLE001
        return -1, f"[{tag}] subprocess failed: {exc}"


def _snapshot_arms() -> list[dict]:
    """Alpaca equity + positions snapshot for all 3 arms."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
    except ImportError:
        return []

    arms = {
        "A": ("ALPACA_API_KEY", "ALPACA_API_SECRET"),
        "B": ("ALPACA_API_KEY_B", "ALPACA_API_SECRET_B"),
        "C": ("ALPACA_API_KEY_C", "ALPACA_API_SECRET_C"),
    }
    today_iso = date.today().isoformat()
    out: list[dict] = []
    for arm, (kid, sid) in arms.items():
        key = os.environ.get(kid)
        secret = os.environ.get(sid)
        if not (key and secret):
            continue
        try:
            tc = TradingClient(key, secret, paper=True)
            acct = tc.get_account()
            positions = tc.get_all_positions()
            req = GetOrdersRequest(
                status=QueryOrderStatus.ALL, limit=500,
                after=datetime.fromisoformat(f"{today_iso}T00:00:00+00:00"),
            )
            orders = tc.get_orders(filter=req)
            filled = sum(1 for o in orders
                         if str(o.status) == "OrderStatus.FILLED")
            eq = float(acct.equity)
            last_eq = float(acct.last_equity)
            out.append({
                "arm": arm,
                "equity": eq,
                "cash": float(acct.cash),
                "day_pnl": eq - last_eq,
                "day_pct": (100 * (eq - last_eq) / last_eq) if last_eq else 0.0,
                "n_positions": len(positions),
                "n_orders_filled": filled,
                "positions": sorted([
                    {
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "mv": float(p.market_value),
                        "pnl": float(p.unrealized_intraday_pl),
                    }
                    for p in positions
                ], key=lambda x: x["pnl"], reverse=True),
            })
        except Exception:  # noqa: BLE001
            continue
    return out


def _scan_issues(log_path: Path, day: str) -> dict:
    """Walk today's log, bucket the stuff worth escalating."""
    issues = {
        "errors": [],        # ERROR / Traceback / CRITICAL lines (non-VIX)
        "vix_flakes": 0,     # yfinance VIX noise, auto-counted not listed
        "memory_recycles": [],
        "memory_near_threshold": [],
        "force_regens": defaultdict(int),  # arm -> count
        "jev_tail_spikes": [],
        "laya_tail_spikes": [],
        "websocket_errors": [],
    }
    if not log_path.exists():
        return issues
    gate_lat_re = re.compile(
        r"\[([BC])\] " + day + r" (\d{2}:\d{2}:\d{2}),\d+ .*"
        r"\[(laya|jev)_materiality_gate\] .* "
        r"\w+_ms_total=([0-9.]+) \w+_ms_max=([0-9.]+)"
    )
    memory_hb_re = re.compile(
        r"\[([ABC])\] " + day + r" (\d{2}:\d{2}:\d{2}),\d+ .*"
        r"memory_watchdog heartbeat: private=(\d+) MB / threshold=(\d+) MB"
    )
    force_regen_re = re.compile(
        r"\[([ABC])\] " + day + r" .* \[force_regen_trigger\]"
    )
    recycle_re = re.compile(
        r"\[([ABC])\] " + day + r" .* memory_recycle\[watchdog\]: "
        r"private=(\d+) MB"
    )
    with log_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if day not in line:
                continue
            if "VIX" in line and "yfinance" in line:
                issues["vix_flakes"] += 1
                continue
            if "connection limit exceeded" in line:
                issues["websocket_errors"].append(line.strip()[:160])
                continue
            if any(tag in line for tag in ("CRITICAL", "Traceback")):
                issues["errors"].append(line.strip()[:200])
                continue
            if " ERROR " in line and "yfinance" not in line:
                issues["errors"].append(line.strip()[:200])
                continue
            m = recycle_re.search(line)
            if m:
                issues["memory_recycles"].append(
                    {"arm": m.group(1), "mb": int(m.group(2))}
                )
                continue
            m = force_regen_re.search(line)
            if m:
                issues["force_regens"][m.group(1)] += 1
                continue
            m = gate_lat_re.search(line)
            if m:
                arm, hms, gate_kind, _total, max_ms = m.groups()
                max_f = float(max_ms)
                bucket = ("jev_tail_spikes" if gate_kind == "jev"
                          else "laya_tail_spikes")
                threshold = (JEV_TAIL_MIN_MS if gate_kind == "jev"
                             else LAYA_TAIL_MIN_MS)
                if max_f >= threshold:
                    issues[bucket].append({"ts": hms, "arm": arm,
                                           "max_ms": max_f})
                continue
            m = memory_hb_re.search(line)
            if m:
                arm, hms, mb, thr = m.groups()
                if int(thr) - int(mb) <= MEMORY_FLAG_MB_FROM_THRESHOLD:
                    issues["memory_near_threshold"].append(
                        {"ts": hms, "arm": arm, "mb": int(mb),
                         "threshold": int(thr)}
                    )
    return issues


def _suggest_fixes(issues: dict, snaps: list[dict]) -> list[str]:
    """Deterministic rules on what the day's data earned a flag for."""
    suggestions: list[str] = []
    if issues["memory_recycles"]:
        by_arm = defaultdict(int)
        for r in issues["memory_recycles"]:
            by_arm[r["arm"]] += 1
        for arm, n in by_arm.items():
            suggestions.append(
                f"[{arm}] {n} memory_recycle fire(s) today -> consider "
                "raising AGENTIC_MEM_RECYCLE_MB (current default 6144 in "
                "runner.py, we've been running 7500 since 10-01)"
            )
    if issues["memory_near_threshold"]:
        tight_arms = {h["arm"] for h in issues["memory_near_threshold"]}
        suggestions.append(
            f"memory headroom <200MB at least once on arm(s) "
            f"{sorted(tight_arms)} -> watch tomorrow, raise threshold "
            "if it recurs"
        )
    if len(issues["jev_tail_spikes"]) >= 5:
        max_spike = max(s["max_ms"] for s in issues["jev_tail_spikes"])
        suggestions.append(
            f"{len(issues['jev_tail_spikes'])} Jev tail spike(s) >="
            f"{JEV_TAIL_MIN_MS}ms today (biggest {max_spike:.0f}ms) -> "
            "parallel fan-out is still holding batches under force-regen, "
            "but if daily count keeps climbing, raise "
            "AGENTIC_JEV_MATERIALITY_MAX_WORKERS or add a Jev timeout"
        )
    if issues["websocket_errors"]:
        suggestions.append(
            f"{len(issues['websocket_errors'])} Alpaca websocket "
            "'connection limit exceeded' error(s) -> an arm or standalone "
            "process is competing with the shared bus for the same "
            "account's stream slot; standalone revive paths need a "
            "cool-down OR a different Alpaca account"
        )
    if issues["errors"]:
        suggestions.append(
            f"{len(issues['errors'])} non-VIX ERROR/Traceback line(s) in "
            "today's log; see Issues section for the first ones"
        )
    # Position-level flag: identical qty on same ticker across 2+ arms
    cross_arm_qty: dict[tuple[str, float], list[str]] = defaultdict(list)
    for s in snaps:
        for p in s["positions"]:
            # Round qty to 2dp so near-identical sizing is caught too.
            cross_arm_qty[(p["symbol"], round(p["qty"], 2))].append(s["arm"])
    for (sym, qty), arms_holding in cross_arm_qty.items():
        if len(arms_holding) >= 2:
            suggestions.append(
                f"cross-arm concentration: {sym} x {qty} held by arms "
                f"{sorted(arms_holding)} - independent decision stacks "
                "sized identically, concentration risk if that ticker rolls"
            )
    return suggestions


def _render_report(day: str, snaps: list[dict], issues: dict,
                   suggestions: list[str],
                   postmortem_tail: str, divergences_tail: str,
                   significance_tail: str) -> str:
    """Markdown report body."""
    lines: list[str] = []
    lines.append(f"# EOD report - {day}")
    lines.append("")
    lines.append(f"generated at {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## 1. Final equity + positions")
    lines.append("")
    if snaps:
        lines.append("```")
        hdr = (f"{'arm':<4} {'equity':>12} {'cash':>12} "
               f"{'pos':>4} {'day P&L':>11} {'day %':>7} {'orders':>7}")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        snaps_sorted = sorted(snaps, key=lambda s: s["day_pnl"], reverse=True)
        for s in snaps_sorted:
            tag = " ***" if s is snaps_sorted[0] else ""
            lines.append(
                f"{s['arm']:<4} ${s['equity']:>10,.2f} ${s['cash']:>10,.2f} "
                f"{s['n_positions']:>4} ${s['day_pnl']:>+9,.2f} "
                f"{s['day_pct']:>+6.2f}% {s['n_orders_filled']:>7}{tag}"
            )
        lines.append("```")
    else:
        lines.append("(no Alpaca snapshot - credentials missing?)")
    lines.append("")

    lines.append("## 2. Per-arm position tape")
    lines.append("")
    for s in snaps:
        lines.append(f"**arm {s['arm']}** - {s['n_positions']} positions:")
        for p in s["positions"][:8]:
            lines.append(
                f"- `{p['symbol']}` qty={p['qty']:.2f} "
                f"mv=${p['mv']:,.0f} pnl=${p['pnl']:+,.2f}"
            )
        if len(s["positions"]) > 8:
            lines.append(f"- ... and {len(s['positions']) - 8} more")
        lines.append("")

    lines.append("## 3. Divergence + significance")
    lines.append("")
    lines.append("```")
    lines.append(divergences_tail.strip() or "(no divergence output)")
    lines.append("```")
    lines.append("")
    lines.append("```")
    lines.append(significance_tail.strip() or "(no significance output)")
    lines.append("```")
    lines.append("")

    lines.append("## 4. Issues observed")
    lines.append("")
    recycles = issues["memory_recycles"]
    lines.append(f"- VIX flakes (upstream yfinance): {issues['vix_flakes']}")
    if recycles:
        rec_summary = ", ".join(
            f"[{r['arm']}] {r['mb']}MB" for r in recycles[:5]
        )
        lines.append(f"- memory_recycles: {len(recycles)} ({rec_summary})")
    else:
        lines.append("- memory_recycles: 0")
    lines.append(f"- memory near threshold hits: "
                 f"{len(issues['memory_near_threshold'])}")
    forces = issues["force_regens"]
    lines.append("- force_regens: "
                 + (", ".join(f"[{a}] x{n}" for a, n in sorted(forces.items()))
                    if forces else "none"))
    lines.append(f"- Jev tail spikes (>={JEV_TAIL_MIN_MS}ms per-headline): "
                 f"{len(issues['jev_tail_spikes'])}")
    lines.append(f"- Laya tail spikes (>={LAYA_TAIL_MIN_MS}ms per-headline): "
                 f"{len(issues['laya_tail_spikes'])}")
    lines.append(f"- Alpaca websocket errors: "
                 f"{len(issues['websocket_errors'])}")
    lines.append(f"- non-VIX ERROR/Traceback lines: {len(issues['errors'])}")
    if issues["errors"]:
        lines.append("")
        lines.append("First few:")
        for e in issues["errors"][:5]:
            lines.append(f"  - `{e}`")
    lines.append("")

    lines.append("## 5. Suggested fixes")
    lines.append("")
    if suggestions:
        for s in suggestions:
            lines.append(f"- {s}")
    else:
        lines.append("- (nothing flagged — day was quiet)")
    lines.append("")

    lines.append("## 6. close_postmortem tail")
    lines.append("")
    lines.append("```")
    lines.append(postmortem_tail.strip()[-4000:] or "(no postmortem output)")
    lines.append("```")
    return "\n".join(lines)


def _kill_supervisor() -> str:
    """taskkill the agentic-investor supervisor and its tree.
    Returns a short status message for the caller to log."""
    import platform
    if platform.system() != "Windows":
        return "shutdown skipped: not on Windows"
    try:
        # taskkill by image name is a blunt instrument but we only ever
        # run one supervisor at a time; /T descends the tree, /F
        # bypasses graceful shutdown (acceptable - arms re-read Alpaca
        # positions on next launch per restart policy).
        result = subprocess.run(
            ["taskkill", "/IM", "agentic-investor.exe", "/T", "/F"],
            capture_output=True, text=True, timeout=30,
        )
        return f"taskkill rc={result.returncode}: {result.stdout.strip()}"
    except Exception as exc:  # noqa: BLE001
        return f"taskkill failed: {exc}"


def _rotate_log(day: str) -> str:
    """Rename today's experiment.out so tomorrow's launch gets a
    fresh file."""
    if not LOG.exists():
        return "log rotate skipped: experiment.out missing"
    weekday = datetime.strptime(day, "%Y-%m-%d").strftime("%a").lower()
    target = LOG.parent / f"experiment.{weekday}-{day}.out"
    try:
        LOG.rename(target)
        return f"log rotated to {target.name}"
    except OSError as exc:
        return f"log rotate failed: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD, default today")
    ap.add_argument("--since", default="2026-09-22",
                    help="significance window start (YYYY-MM-DD)")
    ap.add_argument("--shutdown", action="store_true",
                    help="also kill supervisor + rotate log after the "
                         "report writes")
    args = ap.parse_args()

    day = args.date or date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[eod] running for {day}")
    snaps = _snapshot_arms()
    print(f"[eod] snapshot: {len(snaps)} arm(s) reachable")

    print("[eod] close_postmortem...")
    pm_rc, pm_out = _run(
        [sys.executable, "scripts/close_postmortem.py"],
        "close_postmortem",
    )
    print(f"[eod] close_postmortem rc={pm_rc}")

    print("[eod] gate_divergences --with-forward-pnl...")
    gd_rc, gd_out = _run(
        [sys.executable, "scripts/gate_divergences.py",
         "--date", day, "--with-forward-pnl"],
        "gate_divergences",
    )
    print(f"[eod] gate_divergences rc={gd_rc}")

    print(f"[eod] ab_significance --since {args.since}...")
    sig_rc, sig_out = _run(
        [sys.executable, "scripts/ab_significance.py",
         "--since", args.since],
        "ab_significance",
    )
    print(f"[eod] ab_significance rc={sig_rc}")

    print("[eod] scanning log for issues...")
    issues = _scan_issues(LOG, day)
    print(f"[eod] issues: {len(issues['memory_recycles'])} recycles, "
          f"{len(issues['jev_tail_spikes'])} Jev tails, "
          f"{len(issues['websocket_errors'])} ws errors, "
          f"{len(issues['errors'])} other errors")

    suggestions = _suggest_fixes(issues, snaps)
    report = _render_report(day, snaps, issues, suggestions,
                            pm_out, gd_out, sig_out)
    out_path = OUT_DIR / f"{day}_eod.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"[eod] wrote report -> {out_path}")

    if args.shutdown:
        print("[eod] shutting down supervisor...")
        print(f"[eod] {_kill_supervisor()}")
        import time
        time.sleep(3)  # let OS release the log file handle
        print(f"[eod] {_rotate_log(day)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
