"""Tests for confidence calibration bucketing + win scoring."""

from __future__ import annotations

import pytest

from agentic_investor.ops.calibration import (
    TradeOutcome,
    _win_for,
    bucket_outcomes,
    render_calibration_report,
)


def _o(conf: float, side: str = "buy", win: float = 1.0) -> TradeOutcome:
    return TradeOutcome(
        ticker="X", side=side, submitted_at="2026-08-28T10:00:00Z",
        filled_price=100.0, confidence=conf, rec_id=1,
        later_price=101.0 if side == "buy" else 99.0,
        horizon_minutes=60, return_pct=1.0 if side == "buy" else -1.0,
        win=win,
    )


def test_win_for_buy_up_is_win():
    ret, w = _win_for("buy", 100.0, 102.0)
    assert w == 1.0
    assert ret == pytest.approx(2.0)


def test_win_for_buy_down_is_loss():
    ret, w = _win_for("buy", 100.0, 98.0)
    assert w == 0.0


def test_win_for_sell_down_is_win():
    _, w = _win_for("sell", 100.0, 98.0)
    assert w == 1.0


def test_win_for_flat_counts_as_half():
    _, w = _win_for("buy", 100.0, 100.02)
    assert w == 0.5


def test_bucketing_puts_confidence_in_right_bucket():
    # 5 buckets: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
    outcomes = [_o(0.1), _o(0.3), _o(0.5), _o(0.7), _o(0.95)]
    buckets = bucket_outcomes(outcomes, n_buckets=5)
    assert [b.n_trades for b in buckets] == [1, 1, 1, 1, 1]


def test_bucketing_edge_case_confidence_1():
    # conf=1.0 must land in the last bucket, not overflow.
    outcomes = [_o(1.0)]
    buckets = bucket_outcomes(outcomes, n_buckets=5)
    assert buckets[-1].n_trades == 1


def test_win_rate_aggregates_correctly():
    # 3 trades in the same bucket: 2 wins, 1 loss -> 66.7% win rate.
    outcomes = [_o(0.85, win=1.0), _o(0.85, win=1.0), _o(0.85, win=0.0)]
    buckets = bucket_outcomes(outcomes, n_buckets=5)
    top = buckets[-1]
    assert top.n_trades == 3
    assert abs(top.win_rate - 2 / 3) < 1e-9


def test_report_handles_empty_outcomes():
    md = render_calibration_report([], [], horizon=60)
    assert "no trades" in md.lower()


def test_report_reports_gap():
    # LLM was 0.9 confident but only won half the time -> overconfident (-0.4).
    outcomes = [_o(0.9, win=1.0), _o(0.9, win=0.0)]
    buckets = bucket_outcomes(outcomes, n_buckets=5)
    md = render_calibration_report(outcomes, buckets, horizon=60)
    assert "50.0%" in md  # win rate
    assert "-40.0%" in md or "-0.4" in md.lower()  # gap
