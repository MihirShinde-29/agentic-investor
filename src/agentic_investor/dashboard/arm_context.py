"""Per-request arm context for the multi-arm experiment dashboard.

When the dashboard is launched in experiment mode (`paper-dashboard {name}`),
the server knows about all arms in `out/experiments/{name}/`. Each HTTP
request selects an arm via `?arm=X`; every DB / broker call in that
request routes to that arm's paper_store DB URL and Alpaca account.

Single-arm (legacy) mode: no experiment loaded, arm context is empty,
endpoints default to the process's ambient DATABASE_URL + primary Alpaca
account (i.e., exactly what `paper-loop --serve-dashboard` did before
M13 A/B shipped).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArmSpec:
    """One arm's routing info for the dashboard: which SQLite + which broker."""

    arm_id: str
    alpaca_account: str  # "primary" | "secondary" | "tertiary"
    db_url: str          # sqlite:///out/experiments/{name}/{arm_id}.db


@dataclass(frozen=True)
class ExperimentContext:
    """What the server needs to know when it's serving an experiment."""

    name: str
    exp_dir: Path
    arms: tuple[ArmSpec, ...]

    def default_arm(self) -> ArmSpec:
        return self.arms[0]

    def arm(self, arm_id: str) -> ArmSpec | None:
        for a in self.arms:
            if a.arm_id == arm_id:
                return a
        return None


def load_experiment_context(experiment_name: str) -> ExperimentContext:
    """Read the manifest + inspect the on-disk arm DBs.

    Uses the same manifest loader the runner uses, so the arm ordering
    and alpaca_account routing are guaranteed to match what the arms
    actually run under. arm DBs that don't yet exist on disk still get
    added to the context (the loop will create them on first tick).
    """
    from agentic_investor.experiments.manifest import load_experiment

    exp = load_experiment(experiment_name)
    exp_dir = Path("out") / "experiments" / exp.name
    arms = tuple(
        ArmSpec(
            arm_id=a.arm_id,
            alpaca_account=a.alpaca_account,
            db_url=f"sqlite:///{exp_dir / f'{a.arm_id}.db'}",
        )
        for a in exp.arms
    )
    return ExperimentContext(name=exp.name, exp_dir=exp_dir, arms=arms)
