"""Tests for StrategyProfile + presets + overrides."""

import pytest
from pydantic import ValidationError

from agentic_investor.orchestrator.strategy import (
    AGGRESSIVE,
    CONSERVATIVE,
    MODERATE,
    PRESETS,
    StrategyProfile,
    apply_overrides,
    get_preset,
    load_profile,
    regime_adjusted_profile,
)


def test_presets_registry_has_three_tiers():
    assert set(PRESETS) == {"conservative", "moderate", "aggressive"}


def test_conservative_uses_inverse_vol_with_bonds_and_gold():
    assert CONSERVATIVE.allocator == "inverse_vol"
    assert CONSERVATIVE.rebalance == "quarterly"
    assert "TLT" in CONSERVATIVE.universe_extras
    assert "GLD" in CONSERVATIVE.universe_extras
    assert CONSERVATIVE.max_single_pct == 20.0
    assert CONSERVATIVE.cash_floor_pct == 20.0


def test_moderate_uses_llm_with_bands_and_asymmetric_buy_multiplier():
    assert MODERATE.allocator == "llm"
    assert MODERATE.rebalance == "bands"
    assert MODERATE.band_buy_multiplier == 2.0
    assert MODERATE.dd_buy_pause_pct == 15.0


def test_aggressive_uses_higher_caps_and_zero_cash_floor():
    assert AGGRESSIVE.max_single_pct == 50.0
    assert AGGRESSIVE.cash_floor_pct == 0.0
    assert AGGRESSIVE.rebalance == "monthly"


def test_get_preset_returns_a_copy_not_the_shared_instance():
    p1 = get_preset("moderate")
    p1.max_single_pct = 99.0
    p2 = get_preset("moderate")
    assert p2.max_single_pct == 35.0  # mutation of p1 didn't leak into preset


def test_get_preset_rejects_unknown_risk():
    with pytest.raises(ValueError):
        get_preset("yolo")  # type: ignore[arg-type]


def test_apply_overrides_only_touches_provided_fields():
    base = get_preset("moderate")
    updated = apply_overrides(base, allocator="equal_weight", cash_yield_annual=None)
    assert updated.allocator == "equal_weight"
    # cash_yield_annual is None so must NOT change
    assert updated.cash_yield_annual == base.cash_yield_annual
    # Other fields untouched
    assert updated.rebalance == base.rebalance


def test_strategy_profile_validates_caps():
    with pytest.raises(ValidationError):
        StrategyProfile(max_single_pct=150.0)


# load_profile


def test_load_profile_returns_preset_by_name():
    p = load_profile("moderate")
    assert p.name == "moderate"
    assert p.allocator == "llm"


def test_load_profile_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent-preset-name")


def test_regime_adjusted_profile_high_vol_tightens_everything():
    p = get_preset("moderate")
    adj, notes = regime_adjusted_profile(p, "high_vol")
    assert adj.cash_floor_pct == p.cash_floor_pct + 10.0
    assert adj.max_positions == p.max_positions - 3
    assert adj.max_single_pct == p.max_single_pct - 10.0
    assert adj.band_abs_pct == p.band_abs_pct + 2.0
    assert notes and "high_vol" in notes[0]


def test_regime_adjusted_profile_bear_lifts_cash_and_trims_caps():
    p = get_preset("moderate")
    adj, notes = regime_adjusted_profile(p, "bear")
    assert adj.cash_floor_pct == p.cash_floor_pct + 5.0
    assert adj.max_single_pct == p.max_single_pct - 5.0
    assert adj.max_positions == p.max_positions  # unchanged
    assert notes and "bear" in notes[0]


def test_regime_adjusted_profile_bull_relaxes_cash_only():
    p = get_preset("moderate")
    adj, notes = regime_adjusted_profile(p, "bull")
    assert adj.cash_floor_pct == p.cash_floor_pct - 3.0
    assert adj.max_positions == p.max_positions
    assert adj.max_single_pct == p.max_single_pct


def test_regime_adjusted_profile_sideways_and_unknown_are_noops():
    p = get_preset("moderate")
    for label in ("sideways", "unknown", "", "wat"):
        adj, notes = regime_adjusted_profile(p, label)
        assert adj.model_dump() == p.model_dump()
        assert notes == []


def test_regime_adjusted_profile_clamps_lower_bounds():
    # Custom tight profile: verify caps don't go below the floor.
    p = StrategyProfile(
        name="tight", max_single_pct=18.0, cash_floor_pct=0.0, max_positions=6,
    )
    adj, _ = regime_adjusted_profile(p, "high_vol")
    assert adj.max_single_pct >= 15.0
    assert adj.max_positions >= 5


def test_load_profile_reads_toml_file(tmp_path):
    toml = tmp_path / "custom.toml"
    toml.write_text(
        'name = "my-conservative-plus"\n'
        'allocator = "inverse_vol"\n'
        'rebalance = "quarterly"\n'
        'cash_yield_annual = 0.05\n'
        'max_single_pct = 15.0\n'
        'cash_floor_pct = 25.0\n'
        'universe_extras = ["TLT", "GLD", "IEF"]\n'
    )
    p = load_profile(str(toml))
    assert p.name == "my-conservative-plus"
    assert p.allocator == "inverse_vol"
    assert p.max_single_pct == 15.0
    assert "IEF" in p.universe_extras
