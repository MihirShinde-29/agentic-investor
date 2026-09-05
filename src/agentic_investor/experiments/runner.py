"""Spawn one paper-loop subprocess per experiment arm.

Each arm runs the unmodified single-arm loop with its own Alpaca
account routing + DATABASE_URL. Runner streams prefixed logs and
forwards Ctrl+C to all children.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from agentic_investor.experiments.manifest import Experiment


def _arm_env(
    arm_alpaca_account: str,
    arm_db_path: Path,
    *,
    arm_id: str,
    news_bus_url: str | None = None,
    price_bus_url: str | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    arm_db_path.parent.mkdir(parents=True, exist_ok=True)
    env["DATABASE_URL"] = f"sqlite:///{arm_db_path}"
    env["AGENTIC_ARM_ID"] = arm_id
    if news_bus_url:
        env["AGENTIC_NEWS_BUS"] = news_bus_url
    if price_bus_url:
        env["AGENTIC_PRICE_BUS"] = price_bus_url
    return env


def _config_diff_to_cli_args(diff: dict) -> list[str]:
    """Translate config_diff to paper-loop CLI flags; raise on unknown keys."""
    supported = {
        "opinion_drift_threshold_pct": "--opinion-drift-threshold-pct",
        "max_single_delta_pct": "--max-single-delta-pct",
        "max_avg_drift_pct": "--max-avg-drift-pct",
        "max_positions_override": "--max-positions",
        "band_abs_pct": "--band-abs-pct",
        "band_rel_pct": "--band-rel-pct",
    }
    args: list[str] = []
    for key, value in diff.items():
        flag = supported.get(key)
        if flag is None:
            raise ValueError(
                f"config_diff key {key!r} isn't a paper-loop CLI flag - "
                f"supported: {sorted(supported)}"
            )
        args.extend([flag, str(value)])
    return args


def _stream_prefixed(stream, prefix: str, out=sys.stdout) -> None:
    for raw in iter(stream.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
        except Exception:  # noqa: BLE001
            continue
        out.write(f"[{prefix}] {line}\n")
        out.flush()


def run_experiment(
    experiment: Experiment,
    *,
    base_paper_loop_args: list[str] | None = None,
    dry_run_launch: bool = False,
    serve_dashboard: bool = False,
    dashboard_port: int = 8000,
) -> int:
    """Spawn one paper-loop subprocess per arm and wait until all finish.

    base_paper_loop_args are shared paper-loop flags (--auto, --top-n,
    --regen-mode, --serve-dashboard, --finbert-prefilter, etc).
    Returns aggregate exit code (0 iff all arms exited cleanly).
    """
    if not experiment.arms:
        raise ValueError(f"experiment {experiment.name!r} has no arms")
    base = list(base_paper_loop_args or [])
    procs: list[tuple[str, subprocess.Popen]] = []
    threads: list[threading.Thread] = []
    exp_dir = Path("out") / "experiments" / experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    news_bus_path = exp_dir / "news_bus.db"
    news_bus_url = f"sqlite:///{news_bus_path}"
    price_bus_path = exp_dir / "price_bus.db"
    price_bus_url = f"sqlite:///{price_bus_path}"
    print(f"\nexperiment: {experiment.name}")
    print(f"arms: {[a.arm_id for a in experiment.arms]}")
    print(f"working dir: {exp_dir}")
    print(f"shared news bus:  {news_bus_path}")
    print(f"shared price bus: {price_bus_path}\n")

    # Bus writers own the single Alpaca news + market-data websockets
    # for the whole experiment; arms fan out through the bus DBs instead
    # of hitting Alpaca's 1-connection-per-key limit.
    for bus_name, subcmd, bus_url in (
        ("news-bus", "paper-news-bus", news_bus_url),
        ("price-bus", "paper-price-bus", price_bus_url),
    ):
        bus_cmd = [
            sys.executable, "-m", "agentic_investor.cli",
            subcmd, bus_url,
        ]
        print(f"  {bus_name} writer cmd: {' '.join(bus_cmd)}")
        if dry_run_launch:
            continue
        bus_proc = subprocess.Popen(
            bus_cmd, env=dict(os.environ),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
        procs.append((bus_name, bus_proc))
        t = threading.Thread(
            target=_stream_prefixed, args=(bus_proc.stdout, bus_name),
            daemon=True,
        )
        t.start()
        threads.append(t)
    if not dry_run_launch:
        # Let the writers CREATE TABLE before arms start polling.
        time.sleep(2)

    if serve_dashboard:
        dash_cmd = [
            sys.executable, "-m", "agentic_investor.cli",
            "paper-dashboard", experiment.name,
            "--port", str(dashboard_port),
        ]
        print(f"  dashboard cmd: {' '.join(dash_cmd)}")
        if not dry_run_launch:
            dash_proc = subprocess.Popen(
                dash_cmd, env=dict(os.environ),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1,
            )
            procs.append(("dashboard", dash_proc))
            t = threading.Thread(
                target=_stream_prefixed,
                args=(dash_proc.stdout, "dashboard"),
                daemon=True,
            )
            t.start()
            threads.append(t)

    for arm in experiment.arms:
        arm_db = exp_dir / f"{arm.arm_id}.db"
        env = _arm_env(
            arm.alpaca_account, arm_db,
            arm_id=arm.arm_id,
            news_bus_url=news_bus_url,
            price_bus_url=price_bus_url,
        )
        cmd = [
            sys.executable, "-m", "agentic_investor.cli",
            "paper-loop",
            "--alpaca-account", arm.alpaca_account,
            "--log-file", str(exp_dir / f"{arm.arm_id}.log"),
        ] + base + _config_diff_to_cli_args(arm.config_diff)
        print(f"  arm {arm.arm_id}: account={arm.alpaca_account} db={arm_db.name}")
        print(f"    cmd: {' '.join(cmd)}")
        if dry_run_launch:
            continue
        p = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
        procs.append((arm.arm_id, p))
        t = threading.Thread(
            target=_stream_prefixed, args=(p.stdout, arm.arm_id),
            daemon=True,
        )
        t.start()
        threads.append(t)
        # Stagger keeps arm FinBERT / cache loads from colliding.
        time.sleep(2)
    if dry_run_launch:
        return 0

    def _shutdown(_sig=None, _frame=None):
        for arm_id, p in procs:
            if p.poll() is None:
                print(f"\n[{arm_id}] sending SIGINT")
                try:
                    p.send_signal(signal.SIGINT)
                except Exception:  # noqa: BLE001
                    try:
                        p.terminate()
                    except Exception:  # noqa: BLE001
                        pass

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, AttributeError):
        pass  # SIGTERM not settable on Windows in some contexts

    rc = 0
    for arm_id, p in procs:
        r = p.wait()
        print(f"[{arm_id}] exited with code {r}")
        if r != 0:
            rc = r
    return rc
