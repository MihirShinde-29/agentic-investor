"""Spawn and monitor one subprocess per experiment arm.

Each arm is a normal `paper-loop` invocation with its Alpaca account
routing and DATABASE_URL overridden via env so books stay cleanly
separated. The runner just launches, streams logs prefixed with arm_id,
and waits for shutdown (Ctrl+C forwards to all children).

Keeps the loop code path unchanged - the existing single-arm loop we
know works well is exactly what each subprocess runs.
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
    news_bus_url: str | None = None,
) -> dict[str, str]:
    """Return env overrides for one arm's subprocess.

    Preserves existing env, only overrides:
      DATABASE_URL     -> arm's own SQLite file for LoopState + orders
      AGENTIC_NEWS_BUS -> shared news bus (if the experiment has one),
                          so the arm reads from the bus instead of opening
                          its own Alpaca websocket

    Alpaca account is passed via CLI flag, not env, so multiple arms in
    the same shell won't accidentally cross-contaminate.
    """
    env = dict(os.environ)
    arm_db_path.parent.mkdir(parents=True, exist_ok=True)
    env["DATABASE_URL"] = f"sqlite:///{arm_db_path}"
    if news_bus_url:
        env["AGENTIC_NEWS_BUS"] = news_bus_url
    return env


def _config_diff_to_cli_args(diff: dict) -> list[str]:
    """Translate a config_diff dict into paper-loop CLI arguments.

    Only the knobs that paper-loop already exposes as CLI flags are
    supported. Anything else raises so mis-typed manifests fail early
    instead of silently ignoring an override.
    """
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
    """Prefix every line from a subprocess pipe with `[arm]`."""
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
    print(f"\nexperiment: {experiment.name}")
    print(f"arms: {[a.arm_id for a in experiment.arms]}")
    print(f"working dir: {exp_dir}")
    print(f"shared news bus: {news_bus_path}\n")

    # Spawn the shared news bus writer first. It owns the single Alpaca
    # news websocket for the whole experiment; arms read from the bus DB
    # instead of trying to open their own (Alpaca's 1-connection-per-key
    # limit would reject all but one arm otherwise).
    bus_cmd = [
        sys.executable, "-m", "agentic_investor.cli",
        "paper-news-bus", news_bus_url,
    ]
    print(f"  news-bus writer cmd: {' '.join(bus_cmd)}")
    if not dry_run_launch:
        bus_proc = subprocess.Popen(
            bus_cmd, env=dict(os.environ),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
        procs.append(("news-bus", bus_proc))
        t = threading.Thread(
            target=_stream_prefixed, args=(bus_proc.stdout, "news-bus"),
            daemon=True,
        )
        t.start()
        threads.append(t)
        # Give the writer a moment to CREATE TABLE before arms start
        # polling for rows.
        time.sleep(2)

    for arm in experiment.arms:
        arm_db = exp_dir / f"{arm.arm_id}.db"
        env = _arm_env(arm.alpaca_account, arm_db, news_bus_url=news_bus_url)
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
        # Small stagger keeps their FinBERT / cache loads from hitting the
        # same 100ms window and stalling each other.
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
