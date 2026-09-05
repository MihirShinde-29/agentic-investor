"""Experiment context loaded at dashboard startup from an arm manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    alpaca_account: str  # "primary" | "secondary" | "tertiary"
    db_url: str          # sqlite:///out/experiments/{name}/{arm_id}.db


@dataclass(frozen=True)
class ExperimentContext:
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
