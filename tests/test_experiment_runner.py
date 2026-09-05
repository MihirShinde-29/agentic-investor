"""Runner-side smoke tests: config-diff -> CLI translation + dry-run launch."""

from __future__ import annotations

import pytest


def test_config_diff_to_cli_translates_supported_knobs():
    from agentic_investor.experiments.runner import _config_diff_to_cli_args

    args = _config_diff_to_cli_args({
        "opinion_drift_threshold_pct": 3.0,
        "band_abs_pct": 4.0,
    })
    # Order isn't guaranteed but both flags must appear with their values.
    assert "--opinion-drift-threshold-pct" in args
    assert "3.0" in args
    assert "--band-abs-pct" in args
    assert "4.0" in args


def test_config_diff_rejects_unknown_key():
    from agentic_investor.experiments.runner import _config_diff_to_cli_args

    with pytest.raises(ValueError, match="isn't a paper-loop CLI flag"):
        _config_diff_to_cli_args({"nonexistent_knob": 5})


def test_dry_run_launch_prints_but_does_not_spawn(tmp_path, monkeypatch, capsys):
    from agentic_investor.experiments.manifest import (
        ArmSpec,
        Experiment,
    )
    from agentic_investor.experiments.runner import run_experiment

    monkeypatch.chdir(tmp_path)
    exp = Experiment(
        name="dryrun-smoke",
        arms=[
            ArmSpec(arm_id="A", alpaca_account="primary",
                    config_diff={"opinion_drift_threshold_pct": 5}),
            ArmSpec(arm_id="B", alpaca_account="secondary",
                    config_diff={"opinion_drift_threshold_pct": 3}),
        ],
    )
    rc = run_experiment(
        exp, base_paper_loop_args=["--auto"],
        dry_run_launch=True,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "experiment: dryrun-smoke" in out
    assert "arm A: account=primary" in out
    assert "arm B: account=secondary" in out
    # per-arm DB path should get planted under out/experiments/{name}/
    assert (tmp_path / "out" / "experiments" / "dryrun-smoke").exists()
