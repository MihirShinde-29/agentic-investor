"""Shared test fixtures.

Tests get a fresh SQLite DB per session so state saved by one test can't
leak into another (loop_state, recommendations, paper_orders all share the
default `agentic_investor.db` in production). Without this, the order of
test execution changes results — see CI failure of
test_loop_runs_one_tick_then_exits_with_once on 2026-08-31.
"""

from __future__ import annotations

import os

import pytest

from agentic_investor.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "agentic_investor.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()
        os.environ.pop("DATABASE_URL", None)
