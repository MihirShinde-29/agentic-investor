"""Pre-flight checks before a paper-experiment launch.

Every failure is something you'd rather discover 5 minutes before market
open than 30 seconds after news arrives and the LLM tries to trade.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt_row(name: str, ok: bool, detail: str = "") -> str:
    tag = "OK  " if ok else "FAIL"
    prefix = "[+]" if ok else "[!]"
    return f"  {prefix} {tag} {name:<32s} {detail}"


def check_alpaca_accounts(
    expected: dict[str, tuple[str, float, int]] | None = None,
) -> list[tuple[str, bool, str]]:
    """Verify each configured Alpaca account is reachable + matches expected shape.

    expected maps arm label -> (min-cash, min-equity, max-positions) so a
    fresh experiment expects "primary/secondary/tertiary at $100K / 0
    positions" but a resume from yesterday's session might expect
    different starting equity per account.
    """
    from agentic_investor.tools.paper_broker import AlpacaPaperBroker

    if expected is None:
        expected = {label: (0.0, 0.0, 999) for label in ("primary", "secondary", "tertiary")}
    results: list[tuple[str, bool, str]] = []
    for label, (min_cash, min_equity, max_positions) in expected.items():
        try:
            b = AlpacaPaperBroker(account=label)
            acct = b.get_account()
            positions = b.get_positions()
            ok = (
                float(acct.cash) >= min_cash
                and float(acct.equity) >= min_equity
                and len(positions) <= max_positions
            )
            detail = (
                f"{acct.account_number}  cash=${float(acct.cash):>10,.2f}  "
                f"equity=${float(acct.equity):>10,.2f}  positions={len(positions)}"
            )
            results.append((f"alpaca:{label}", ok, detail))
        except Exception as e:  # noqa: BLE001
            results.append((f"alpaca:{label}", False, f"unreachable: {e}"))
    return results


def check_chroma_seed(min_recs: int = 200) -> tuple[str, bool, str]:
    try:
        import chromadb

        from agentic_investor.config import get_settings
        client = chromadb.PersistentClient(path=get_settings().chroma_dir)
        coll = client.get_or_create_collection(name="recommendations")
        count = coll.count()
        ok = count >= min_recs
        return ("chroma:recommendations", ok, f"{count} docs (min={min_recs})")
    except Exception as e:  # noqa: BLE001
        return ("chroma:recommendations", False, f"error: {e}")


def check_memory_outcomes_sweep() -> tuple[str, bool, str]:
    """Dry-run the sweep against the current chroma - proves the whole
    metadata-refresh pipeline works before Tuesday's arm docs start
    landing in the collection."""
    try:
        from agentic_investor.memory.outcomes import attach_outcomes_to_index
        n_updated, n_with = attach_outcomes_to_index()
        return (
            "memory:outcomes-sweep", True,
            f"updated {n_updated} recs; {n_with} have outcomes",
        )
    except Exception as e:  # noqa: BLE001
        return ("memory:outcomes-sweep", False, f"error: {e}")


def check_port_available(port: int) -> tuple[str, bool, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind(("127.0.0.1", port))
            return (f"port:{port}", True, "available")
        except OSError as e:
            return (f"port:{port}", False, f"in use ({e})")


def check_experiment_dir(experiment_name: str) -> tuple[str, bool, str]:
    """Verify the experiment dir either doesn't exist (clean start) or
    only contains leftover bus DBs / logs from a prior stopped run."""
    exp_dir = Path("out") / "experiments" / experiment_name
    if not exp_dir.exists():
        return (f"exp-dir:{experiment_name}", True, "clean (will create)")
    files = list(exp_dir.glob("*"))
    stale_arm_dbs = [
        f for f in files
        if f.suffix == ".db" and f.stem not in ("news_bus", "price_bus")
    ]
    if not stale_arm_dbs:
        return (f"exp-dir:{experiment_name}", True,
                f"exists but no stale arm DBs ({len(files)} files)")
    return (
        f"exp-dir:{experiment_name}", False,
        f"stale arm DBs from prior run: {[f.name for f in stale_arm_dbs]}",
    )


def check_arm_db_writable(
    experiment_name: str, arm_ids: list[str],
) -> list[tuple[str, bool, str]]:
    """Confirm each arm's target dir is writable. Uses os.access rather
    than actually opening SQLite so Windows file-handle races don't
    produce false negatives during preflight."""
    import os

    exp_dir = Path("out") / "experiments" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    results = []
    dir_ok = os.access(str(exp_dir), os.W_OK)
    for arm_id in arm_ids:
        results.append((
            f"arm-db:{arm_id}", dir_ok,
            "writable" if dir_ok else f"dir not writable: {exp_dir}",
        ))
    return results


def run_preflight(experiment_name: str, dashboard_port: int = 8000) -> int:
    """Print check results; return 0 if all pass, 1 otherwise."""
    from agentic_investor.experiments.manifest import load_experiment

    exp = load_experiment(experiment_name)
    arm_ids = [a.arm_id for a in exp.arms]
    accounts = [a.alpaca_account for a in exp.arms]
    fresh_expected = {acc: (0.0, 0.0, 999) for acc in accounts}

    checks: list[tuple[str, bool, str]] = []
    checks.extend(check_alpaca_accounts(fresh_expected))
    checks.append(check_chroma_seed())
    checks.append(check_memory_outcomes_sweep())
    checks.append(check_port_available(dashboard_port))
    checks.append(check_experiment_dir(experiment_name))
    checks.extend(check_arm_db_writable(experiment_name, arm_ids))

    print(f"\nPre-flight for experiment '{experiment_name}' ({len(arm_ids)} arms):\n")
    for name, ok, detail in checks:
        print(_fmt_row(name, ok, detail))
    failed = [c for c in checks if not c[1]]
    print()
    if failed:
        print(f"[!] {len(failed)} check(s) FAILED — fix before launch.")
        return 1
    print("[+] all clear — safe to launch.")
    return 0
