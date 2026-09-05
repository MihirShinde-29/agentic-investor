"""Experiment manifest loader validates arms + accounts + config diffs."""

from __future__ import annotations

import pytest


def test_loads_two_arm_manifest(tmp_path):
    from agentic_investor.experiments.manifest import load_experiment

    yml = tmp_path / "exp.yaml"
    yml.write_text("""
name: sample
arms:
  A:
    alpaca_account: primary
    config_diff:
      opinion_drift_threshold_pct: 5
  B:
    alpaca_account: secondary
    config_diff:
      opinion_drift_threshold_pct: 3
""".strip())
    exp = load_experiment(str(yml))
    assert exp.name == "sample"
    assert [a.arm_id for a in exp.arms] == ["A", "B"]
    assert exp.arm("A").alpaca_account == "primary"
    assert exp.arm("B").config_diff["opinion_drift_threshold_pct"] == 3


def test_accepts_tertiary_account(tmp_path):
    from agentic_investor.experiments.manifest import load_experiment

    yml = tmp_path / "exp.yaml"
    yml.write_text("""
name: abc
arms:
  A: { alpaca_account: primary }
  B: { alpaca_account: secondary }
  C: { alpaca_account: tertiary }
""".strip())
    exp = load_experiment(str(yml))
    assert {a.alpaca_account for a in exp.arms} == {
        "primary", "secondary", "tertiary",
    }


def test_rejects_duplicate_account(tmp_path):
    from agentic_investor.experiments.manifest import load_experiment

    yml = tmp_path / "dup.yaml"
    yml.write_text("""
name: dup
arms:
  A: { alpaca_account: primary }
  B: { alpaca_account: primary }
""".strip())
    with pytest.raises(ValueError, match="already used"):
        load_experiment(str(yml))


def test_rejects_unknown_account(tmp_path):
    from agentic_investor.experiments.manifest import load_experiment

    yml = tmp_path / "bad.yaml"
    yml.write_text("""
name: bad
arms:
  A: { alpaca_account: quaternary }
""".strip())
    with pytest.raises(ValueError, match="must be one of"):
        load_experiment(str(yml))


def test_missing_arms_raises(tmp_path):
    from agentic_investor.experiments.manifest import load_experiment

    yml = tmp_path / "empty.yaml"
    yml.write_text("name: empty\narms: {}\n")
    with pytest.raises(ValueError, match="no arms"):
        load_experiment(str(yml))


def test_missing_file_raises(tmp_path):
    from agentic_investor.experiments.manifest import load_experiment

    with pytest.raises(FileNotFoundError):
        load_experiment(str(tmp_path / "nope.yaml"))


def test_bare_name_resolves_under_experiments_dir(tmp_path, monkeypatch):
    from agentic_investor.experiments.manifest import load_experiment

    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "sample.yaml").write_text(
        "name: sample\narms:\n  A: { alpaca_account: primary }\n"
    )
    monkeypatch.chdir(tmp_path)
    exp = load_experiment("sample")
    assert exp.name == "sample"
