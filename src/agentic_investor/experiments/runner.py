"""Spawn one paper-loop subprocess per experiment arm.

Each arm runs the unmodified single-arm loop with its own Alpaca
account routing + DATABASE_URL. Runner streams prefixed logs, tees
its own stdout to orchestrator.log so logs survive the launching
shell exit, forwards Ctrl+C to all children, and restart-supervises
each child up to a per-name budget so a single crash does not orphan
the rest of the run.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_investor.experiments.manifest import Experiment


# Priority 3: TeeStream so orchestrator prints land in the log file even
# after the launching shell exits. Before this, `python ... > log &` on
# Windows-via-Git-Bash left python without a working stdout once the
# shell closed the redirection, so the log froze seconds after startup
# and the memory-sweep refresh lines were never captured.
class _TeeStream:
    """Write to multiple underlying streams; tolerate any single failure.

    The typical pair is (real stdout, orchestrator.log). If either dies
    (broken pipe, unicode error, closed file) we keep writing to the
    other so we never lose observability from both channels at once.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(text) or n
            except (BrokenPipeError, UnicodeEncodeError, OSError, ValueError):
                try:
                    n = s.write(
                        text.encode("ascii", errors="replace").decode("ascii"),
                    ) or n
                except Exception:  # noqa: BLE001
                    pass
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass


# Priority 2: retain spawn recipe alongside the live handle so a crashed
# child can be respawned with its original cmd + env.
@dataclass
class _ProcSpec:
    name: str
    cmd: list[str]
    env: dict[str, str]
    proc: subprocess.Popen
    # "crash_only" -> respawn iff rc != 0; "never" -> never respawn even on
    # crash (use for children that intentionally exit, e.g. --once modes).
    restart_kind: str = "crash_only"
    restarts: int = 0


_DEFAULT_MAX_RESTARTS = 5
_SUPERVISOR_POLL_SEC = 2
_SHUTDOWN_DRAIN_SEC = 10


def _arm_env(
    arm_alpaca_account: str,
    arm_db_path: Path,
    *,
    arm_id: str,
    news_bus_url: str | None = None,
    price_bus_url: str | None = None,
    ml_service_url: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    arm_db_path.parent.mkdir(parents=True, exist_ok=True)
    env["DATABASE_URL"] = f"sqlite:///{arm_db_path}"
    env["AGENTIC_ARM_ID"] = arm_id
    if news_bus_url:
        env["AGENTIC_NEWS_BUS"] = news_bus_url
    if price_bus_url:
        env["AGENTIC_PRICE_BUS"] = price_bus_url
    if ml_service_url:
        env["AGENTIC_ML_SERVICE_URL"] = ml_service_url
    # Memory circuit breaker: recycle an arm subprocess when its private
    # commit exceeds this many MB. Guards against native-code arena leaks
    # in third-party dependencies (2026-09-17: chromadb HNSW segment
    # reservation). Overridable per-arm via arms.<id>.env in the
    # experiment YAML, or globally via the parent shell's env.
    env.setdefault("AGENTIC_MEM_RECYCLE_MB", "6144")
    if extra_env:
        env.update(extra_env)
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
        "news_batch_window_sec": "--news-batch-window-sec",
        "cooldown_seconds": "--cooldown-seconds",
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


def _stream_prefixed(stream, prefix: str, out=None) -> None:
    if out is None:
        out = sys.stdout  # picked up dynamically so the TeeStream wraps it
    for raw in iter(stream.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
        except Exception:  # noqa: BLE001
            continue
        text = f"[{prefix}] {line}\n"
        try:
            out.write(text)
        except UnicodeEncodeError:
            # Windows cp1252 console can't encode some Unicode punctuation
            # (Alpaca headlines sometimes carry U+FFFD replacement chars,
            # smart quotes, em-dashes). ASCII-fallback keeps the relay
            # thread alive; a dead thread stalls the arm on pipe
            # backpressure once its stdout buffer fills.
            out.write(text.encode("ascii", errors="replace").decode("ascii"))
        out.flush()


def _spawn_supervised(
    name: str, cmd: list[str], env: dict[str, str],
    procs: dict[str, _ProcSpec], threads: list[threading.Thread],
    *, restart_kind: str = "crash_only",
) -> _ProcSpec:
    """Popen + start a stdout relay thread + register a supervised spec.

    The relay thread reads the child's stdout and prefixes each line with
    `[name]` so grepping is one-step. The spec lives in `procs` so the
    supervisor loop can respawn on crash.
    """
    p = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1,
    )
    spec = _ProcSpec(
        name=name, cmd=cmd, env=env, proc=p, restart_kind=restart_kind,
    )
    procs[name] = spec
    t = threading.Thread(
        target=_stream_prefixed, args=(p.stdout, name), daemon=True,
    )
    t.start()
    threads.append(t)
    return spec


# Priority 4: run one sync sweep before starting the daemon so chromadb
# corruption is loud at launch instead of manifesting 30 min later.
def _healthcheck_outcome_sweeper(timeout_sec: int = 60) -> bool:
    """Return True iff one sync memory-outcomes call completes cleanly."""
    print("[preflight] running outcome-sweeper healthcheck (sync)...")
    try:
        check = subprocess.run(
            [sys.executable, "-m", "agentic_investor.cli", "memory-outcomes"],
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[preflight] outcome-sweeper healthcheck TIMEOUT after {timeout_sec}s"
        )
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] outcome-sweeper healthcheck spawn error: {e}")
        return False
    if check.returncode == 0:
        print("[preflight] outcome-sweeper healthcheck OK")
        return True
    tail = ((check.stderr or "") + (check.stdout or "")).strip()[-500:]
    print(
        f"[preflight] outcome-sweeper healthcheck FAILED "
        f"(rc={check.returncode}); tail: {tail}"
    )
    return False


def _supervise(
    procs: dict[str, _ProcSpec], threads: list[threading.Thread],
    *, max_restarts: int, shutdown_flag: threading.Event,
    poll_sec: int = _SUPERVISOR_POLL_SEC,
) -> int:
    """Poll every child; respawn any that crash up to max_restarts each.

    Priority 2: without this, `for _, p in procs: p.wait()` would block on
    procs[0] (news-bus, designed to run forever) and never notice arm B or
    arm C dying. Also, a crashed arm would just log-and-forget with no way
    to recover during an unattended multi-day A/B.

    Returns 0 iff every child either exited cleanly OR was respawned back
    to a live state at loop exit; otherwise the last non-zero rc seen.
    """
    aggregate_rc = 0
    while procs and not shutdown_flag.is_set():
        time.sleep(poll_sec)
        for name in list(procs.keys()):
            spec = procs[name]
            rc = spec.proc.poll()
            if rc is None:
                continue  # still running
            if rc == 0:
                print(f"[{name}] exited cleanly (rc=0)")
                del procs[name]
                continue
            aggregate_rc = rc
            print(f"[{name}] crashed with rc={rc}")
            if spec.restart_kind == "never":
                del procs[name]
                continue
            if spec.restarts >= max_restarts:
                print(
                    f"[{name}] restart budget exhausted "
                    f"({spec.restarts}/{max_restarts}); removing from supervision"
                )
                del procs[name]
                continue
            spec.restarts += 1
            print(
                f"[{name}] respawning (attempt {spec.restarts}/{max_restarts})"
            )
            try:
                new_proc = subprocess.Popen(
                    spec.cmd, env=spec.env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=1,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[{name}] respawn failed: {e}; removing")
                del procs[name]
                continue
            spec.proc = new_proc
            t = threading.Thread(
                target=_stream_prefixed, args=(new_proc.stdout, name),
                daemon=True,
            )
            t.start()
            threads.append(t)
    # Drain: wait a bounded amount for children to notice shutdown.
    for name, spec in list(procs.items()):
        try:
            spec.proc.wait(timeout=_SHUTDOWN_DRAIN_SEC)
        except subprocess.TimeoutExpired:
            print(f"[{name}] didn't exit in {_SHUTDOWN_DRAIN_SEC}s; terminating")
            try:
                spec.proc.terminate()
            except Exception:  # noqa: BLE001
                pass
    return aggregate_rc


def run_experiment(
    experiment: Experiment,
    *,
    base_paper_loop_args: list[str] | None = None,
    dry_run_launch: bool = False,
    serve_dashboard: bool = False,
    dashboard_port: int = 8000,
    memory_sweep_interval_min: int = 30,
    fresh: bool = False,
    max_restarts: int = _DEFAULT_MAX_RESTARTS,
) -> int:
    """Spawn one paper-loop subprocess per arm and supervise until done.

    base_paper_loop_args are shared paper-loop flags (--auto, --top-n,
    --regen-mode, --serve-dashboard, --finbert-prefilter, etc).
    max_restarts caps per-child respawn attempts for the whole run.
    Returns aggregate exit code (0 iff all children ended cleanly).
    """
    if not experiment.arms:
        raise ValueError(f"experiment {experiment.name!r} has no arms")
    base = list(base_paper_loop_args or [])
    procs: dict[str, _ProcSpec] = {}
    threads: list[threading.Thread] = []
    exp_dir = Path("out") / "experiments" / experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Priority 3: tee stdout into orchestrator.log so logs survive the
    # launching shell exit. Owned by this process, so writes don't depend
    # on the shell redirection staying open.
    log_path = exp_dir / "orchestrator.log"
    log_file = None
    original_stdout = sys.stdout
    if not dry_run_launch:
        log_file = open(  # noqa: SIM115 - closed in finally below
            log_path, "a", buffering=1, encoding="utf-8", errors="replace",
        )
        sys.stdout = _TeeStream(sys.__stdout__, log_file)
        print(f"[log] orchestrator log: {log_path}")

    news_bus_path = exp_dir / "news_bus.db"
    news_bus_url = f"sqlite:///{news_bus_path}"
    price_bus_path = exp_dir / "price_bus.db"
    price_bus_url = f"sqlite:///{price_bus_path}"
    print(f"\nexperiment: {experiment.name}")
    print(f"arms: {[a.arm_id for a in experiment.arms]}")
    print(f"working dir: {exp_dir}")
    print(f"shared news bus:  {news_bus_path}")
    print(f"shared price bus: {price_bus_path}")

    # --fresh wipes each arm's DB + log so the run starts from an empty
    # LoopState. Bus DBs are preserved so the writers don't have to
    # rebuild schemas on every restart.
    if fresh:
        targets = [
            exp_dir / f"{arm.arm_id}{sfx}"
            for arm in experiment.arms for sfx in (".db", ".log")
            if (exp_dir / f"{arm.arm_id}{sfx}").exists()
        ]
        if dry_run_launch:
            print(f"--fresh (dry-run): WOULD wipe {[p.name for p in targets]}")
        else:
            for p in targets:
                p.unlink()
            print(f"--fresh: wiped {len(targets)} file(s): "
                  f"{[p.name for p in targets]}")
    else:
        existing = [
            arm.arm_id for arm in experiment.arms
            if (exp_dir / f"{arm.arm_id}.db").exists()
        ]
        if existing:
            print(f"resume: {len(existing)}/{len(experiment.arms)} arm(s) "
                  f"have existing state and will resume: {existing}")
            print("(pass --fresh to wipe arm DBs and start clean)")
    print()

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
        _spawn_supervised(bus_name, bus_cmd, dict(os.environ), procs, threads)
    if not dry_run_launch:
        # Let the writers CREATE TABLE before arms start polling.
        time.sleep(2)

    # ML service: one shared subprocess owns finBERT + sentence-transformers.
    # Arms consume via AGENTIC_ML_SERVICE_URL (set on each arm's env below)
    # and fall back to loading models locally if the service is
    # unreachable. Motivated by 2026-09-17: 3 arms x ~530 MB duplicated
    # was enough to push a laptop over the memory-pressure threshold.
    ml_service_port = 8765
    ml_service_url = f"http://127.0.0.1:{ml_service_port}"
    ml_service_cmd = [
        sys.executable, "-m", "agentic_investor.cli",
        "paper-ml-service", "--port", str(ml_service_port),
    ]
    print(f"  ml-service cmd: {' '.join(ml_service_cmd)}")
    if not dry_run_launch:
        _spawn_supervised(
            "ml-service", ml_service_cmd, dict(os.environ), procs, threads,
        )
        # Wait for /health to report both models loaded before spawning
        # arms so their first tick doesn't fall through to local loading.
        # ~30 s is typical (finBERT ~15 s + embed ~10 s + margin).
        from agentic_investor.tools.ml_client import wait_for_healthy
        if wait_for_healthy(ml_service_url, timeout_sec=60.0):
            print("  ml-service ready")
        else:
            print("  ml-service warm-up timed out; arms will fall back to local")

    # M17 outcome sweeper: subprocess (priority 1) with startup
    # healthcheck (priority 4) so store corruption is loud NOW instead
    # of surfacing 30 min into the run.
    from agentic_investor.flags import flags
    if memory_sweep_interval_min > 0 and flags.MEMORY_RAG:
        sweep_cmd = [
            sys.executable, "-m", "agentic_investor.cli",
            "paper-outcome-sweeper",
            "--interval-min", str(memory_sweep_interval_min),
        ]
        print(f"  outcome-sweeper cmd: {' '.join(sweep_cmd)}")
        if not dry_run_launch:
            _healthcheck_outcome_sweeper()
            _spawn_supervised(
                "outcome-sweeper", sweep_cmd, dict(os.environ),
                procs, threads,
            )

    if serve_dashboard:
        dash_cmd = [
            sys.executable, "-m", "agentic_investor.cli",
            "paper-dashboard", experiment.name,
            "--port", str(dashboard_port),
        ]
        print(f"  dashboard cmd: {' '.join(dash_cmd)}")
        if not dry_run_launch:
            _spawn_supervised(
                "dashboard", dash_cmd, dict(os.environ), procs, threads,
            )

    for arm in experiment.arms:
        arm_db = exp_dir / f"{arm.arm_id}.db"
        env = _arm_env(
            arm.alpaca_account, arm_db,
            arm_id=arm.arm_id,
            news_bus_url=news_bus_url,
            price_bus_url=price_bus_url,
            ml_service_url=(ml_service_url if not dry_run_launch else None),
            extra_env=arm.env,
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
        _spawn_supervised(arm.arm_id, cmd, env, procs, threads)
        # Stagger keeps arm FinBERT / cache loads from colliding.
        time.sleep(2)
    if dry_run_launch:
        return 0

    shutdown_flag = threading.Event()

    def _shutdown(_sig=None, _frame=None):
        shutdown_flag.set()
        for name, spec in list(procs.items()):
            if spec.proc.poll() is None:
                print(f"\n[{name}] sending SIGINT")
                try:
                    spec.proc.send_signal(signal.SIGINT)
                except Exception:  # noqa: BLE001
                    try:
                        spec.proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, AttributeError):
        pass  # SIGTERM not settable on Windows in some contexts

    try:
        rc = _supervise(
            procs, threads,
            max_restarts=max_restarts,
            shutdown_flag=shutdown_flag,
        )
    finally:
        # Restore stdout and flush the log file before exit.
        if log_file is not None:
            try:
                sys.stdout = original_stdout
            except Exception:  # noqa: BLE001
                pass
            try:
                log_file.flush()
                log_file.close()
            except Exception:  # noqa: BLE001
                pass
    return rc
