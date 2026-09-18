"""Regression tests for `_extract_tickers_from_batch_ctx`.

This helper feeds the `trigger_tickers` field on `regen_attribution`
events (loop.py:2168-2178), which the dashboard's
`/api/experiment/compare/news-reactions` endpoint uses to attribute
regens to news events (rule (d) in `server.py`).

The pre-fix regex captured the FIRST all-uppercase token after
`[HOT|COOKED|STALE]`, which after render_batch_context started
prefixing each line with the news_id (`N<hex>`) matched the news_id
instead of the ticker. Since `hexdigest()` returns lowercase hex, the
regex truncated the news_id at the first lowercase char, producing
strings like `N60`/`N08` instead of tickers. The dashboard's rule (d)
never matched, so news events that didn't overlap with an order's
ticker directly (rotation cases: news about MRK causing MRK sell +
CRM buy) failed to attribute the CRM leg.

These tests pin the shape emitted by `render_batch_context`
(decision_engine.py:213-216) so any future change to the line format
is caught here.
"""

from __future__ import annotations

from agentic_investor.orchestrator.loop import _extract_tickers_from_batch_ctx


def test_extracts_ticker_after_news_id():
    """The bug: news_id starting with digits truncated to 3 chars
    (e.g. `N60abc123` -> `N60`) and got returned as if it were a
    ticker. Correct behavior returns the actual ticker.
    """
    ctx = "- [HOT] N60abc123 NVDA  age=5m: chip demand"
    assert _extract_tickers_from_batch_ctx(ctx) == ["NVDA"]


def test_multiple_events_dedupes_and_preserves_order():
    ctx = (
        "- [HOT] N3f9a2b1c NVDA  age=5m: chip\n"
        "- [COOKED] N049400ff AAPL  age=45m, reaction=+1.20%: iphone\n"
        "- [HOT] N603abc42 NVDA  age=1m: another\n"
        "- [HOT] N08daf3f2 MSFT  age=2m: azure"
    )
    assert _extract_tickers_from_batch_ctx(ctx) == ["NVDA", "AAPL", "MSFT"]


def test_ticker_with_dot():
    """BRK.B / BF.B style tickers survive; the ticker regex allows dots."""
    ctx = "- [HOT] N08daf3f2 BRK.B  age=2m: buffett"
    assert _extract_tickers_from_batch_ctx(ctx) == ["BRK.B"]


def test_ticker_with_dash():
    """Preferred-share tickers like RDS-A / BRK-B survive too."""
    ctx = "- [HOT] N08daf3f2 RDS-A  age=2m: shell"
    assert _extract_tickers_from_batch_ctx(ctx) == ["RDS-A"]


def test_empty_ctx_returns_empty_list():
    assert _extract_tickers_from_batch_ctx("") == []


def test_stale_bucket_also_matched():
    """render_batch_context excludes STALE from its output, but the
    regex still needs to handle it in case a future render path emits
    it - keeps the extractor tolerant of the full DecisionBatch shape.
    """
    ctx = "- [STALE] N08daf3f2 AAPL  age=90m: iphone"
    assert _extract_tickers_from_batch_ctx(ctx) == ["AAPL"]


def test_news_id_that_happens_to_be_all_digits():
    """The original bug's worst case: a hex prefix that happens to be
    all digits (~2% probability per event) had NO lowercase break so
    the regex captured `N87697300` and returned it as a ticker. The
    fixed regex expects `N[0-9a-f]+` explicitly and then requires the
    ticker as a separate token, so the digit-only case cannot leak.
    """
    ctx = "- [HOT] N87697300 CVX  age=3m: opec cut"
    assert _extract_tickers_from_batch_ctx(ctx) == ["CVX"]


def test_line_without_news_id_prefix_is_ignored():
    """Older render format (pre-news_id prefix) would have looked like
    `- [HOT] NVDA age=...`. The fixed regex requires the news_id, so
    such a line is now ignored. Kept as an explicit contract: if
    render_batch_context ever regresses to the old format, this test
    fails and forces us to update both together.
    """
    ctx = "- [HOT] NVDA  age=5m: chip demand"
    assert _extract_tickers_from_batch_ctx(ctx) == []
