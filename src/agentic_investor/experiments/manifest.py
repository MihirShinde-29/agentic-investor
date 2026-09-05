"""Experiment manifest loader for parallel-arm A/B testing.

Each experiment defines N arms; each arm has an Alpaca account routing
(primary or secondary) and an optional config_diff that overrides
LoopConfig fields for that arm. The multi-arm loop wires everything up.

Manifest shape (YAML):

    name: cash-floor-tightening
    arms:
      A:
        alpaca_account: primary
        config_diff:
          cash_floor_pct: 10
      B:
        alpaca_account: secondary
        config_diff:
          cash_floor_pct: 15
          opinion_drift_threshold_pct: 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_VALID_ACCOUNTS = ("primary", "secondary", "tertiary")


@dataclass
class ArmSpec:
    """One arm of an experiment: which broker + which config overrides."""

    arm_id: str
    alpaca_account: str = "primary"  # "primary" | "secondary" | "tertiary"
    config_diff: dict[str, Any] = field(default_factory=dict)
    # Extra env vars to set on this arm's subprocess. For env-controlled
    # feature flags that don't have a corresponding LoopConfig knob
    # (AGENTIC_MEMORY_RAG, AGENTIC_MEMORY_RAG_K, etc).
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Experiment:
    name: str
    arms: list[ArmSpec]

    def arm(self, arm_id: str) -> ArmSpec:
        for a in self.arms:
            if a.arm_id == arm_id:
                return a
        raise KeyError(f"no such arm {arm_id!r} in experiment {self.name!r}")


def load_experiment(path: str | Path) -> Experiment:
    """Read a YAML manifest and validate it before we start burning LLM tokens."""
    import yaml

    p = Path(path)
    if not p.exists():
        # Also accept a bare experiment name; look under ./experiments/{name}.yaml
        alt = Path("experiments") / f"{p.name}.yaml"
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"experiment manifest not found: {path}")
    raw = yaml.safe_load(p.read_text())
    name = raw.get("name") or p.stem
    arms_raw = raw.get("arms") or {}
    if not arms_raw:
        raise ValueError(f"experiment {name!r} has no arms")
    arms: list[ArmSpec] = []
    seen_accounts: set[str] = set()
    for arm_id, spec in arms_raw.items():
        account = (spec or {}).get("alpaca_account", "primary")
        if account not in _VALID_ACCOUNTS:
            raise ValueError(
                f"arm {arm_id!r}: alpaca_account must be one of "
                f"{list(_VALID_ACCOUNTS)} (got {account!r})"
            )
        if account in seen_accounts:
            raise ValueError(
                f"arm {arm_id!r}: alpaca_account={account!r} already used "
                "by another arm (each arm needs its own account to keep books "
                "cleanly separated)"
            )
        seen_accounts.add(account)
        env_raw = (spec or {}).get("env") or {}
        arms.append(ArmSpec(
            arm_id=str(arm_id),
            alpaca_account=account,
            config_diff=dict((spec or {}).get("config_diff") or {}),
            env={str(k): str(v) for k, v in env_raw.items()},
        ))
    return Experiment(name=name, arms=arms)
