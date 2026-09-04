"""recent_sold_tickers must match rows regardless of ISO separator.

Bug found 2026-09-04: stored `submitted_at` uses space separator
('2026-09-04 15:03:31...') but callers using `datetime.isoformat()`
produce T separator ('2026-09-04T15:03:31...'). String comparison of
' ' (0x20) vs 'T' (0x54) silently returned no rows for hours. Fix
uses `datetime()` on both sides at the SQL layer.
"""

from __future__ import annotations

import pytest

from agentic_investor.tools.paper_store import _connect, recent_sold_tickers


@pytest.fixture
def _db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(
        "agentic_investor.tools.paper_store._resolve_url",
        lambda u=None: url,
    )
    with _connect(url) as conn:
        # Schema is created by _connect; insert a fixture row with the
        # SPACE separator that the loop actually writes.
        conn.execute(
            "INSERT INTO paper_orders (client_order_id, ticker, side, qty, "
            "order_type, status, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "c-o1", "TRV", "sell", 1.0, "market", "filled",
                "2026-09-04 15:03:31.925567+00:00",
            ),
        )
        conn.commit()
    return url


def test_query_matches_when_since_uses_T_separator(_db):
    """Caller passes ISO-T format; row stored with space separator."""
    since = "2026-09-04T14:00:00+00:00"  # 63 min before stored row
    result = recent_sold_tickers(since_iso=since)
    assert "TRV" in result


def test_query_matches_when_since_uses_space_separator(_db):
    """Legacy caller passes space format; still works."""
    since = "2026-09-04 14:00:00+00:00"
    result = recent_sold_tickers(since_iso=since)
    assert "TRV" in result


def test_query_excludes_rows_before_since(_db):
    """The window boundary still enforces — not just format-blind."""
    since = "2026-09-04T20:00:00+00:00"  # after stored row
    result = recent_sold_tickers(since_iso=since)
    assert "TRV" not in result
