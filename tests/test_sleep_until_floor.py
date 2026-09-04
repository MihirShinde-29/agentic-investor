"""_sleep_until floors sleep at 1s to prevent market-open poll spam.

Reproduces the Sept 4 open-bell log flood: at 09:30:00.09 Alpaca's clock
still reported market_closed for ~1s past the bell, so _sleep_until saw
delta<=0 and returned immediately, and the main loop's `continue` fired
the market_closed check 30+ times per second before the broker clock
caught up.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from agentic_investor.orchestrator.loop import _sleep_until


def test_sleeps_at_least_1s_when_target_already_passed(monkeypatch):
    """delta<=0 must still sleep 1s so the loop doesn't tight-spin."""
    captured: list[float] = []
    monkeypatch.setattr(time, "sleep", captured.append)
    past = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
    _sleep_until(past, now=datetime.now(UTC))
    assert captured and captured[0] >= 1.0


def test_sleeps_at_least_1s_when_target_less_than_1s_away(monkeypatch):
    captured: list[float] = []
    monkeypatch.setattr(time, "sleep", captured.append)
    soon = (datetime.now(UTC) + timedelta(milliseconds=100)).isoformat()
    _sleep_until(soon, now=datetime.now(UTC))
    assert captured and captured[0] >= 1.0


def test_caps_at_15_min_when_target_far_away(monkeypatch):
    captured: list[float] = []
    monkeypatch.setattr(time, "sleep", captured.append)
    far = (datetime.now(UTC) + timedelta(hours=8)).isoformat()
    _sleep_until(far, now=datetime.now(UTC))
    assert captured and captured[0] == 900.0


def test_sleeps_exact_delta_when_between_floor_and_cap(monkeypatch):
    captured: list[float] = []
    monkeypatch.setattr(time, "sleep", captured.append)
    ahead = (datetime.now(UTC) + timedelta(seconds=120)).isoformat()
    _sleep_until(ahead, now=datetime.now(UTC))
    assert captured
    assert 118.0 <= captured[0] <= 122.0
