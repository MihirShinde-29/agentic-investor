"""Tests for reconcile_orders trade-status polling."""

from types import SimpleNamespace

from agentic_investor.tools.paper_broker import PaperOrder
from agentic_investor.tools.paper_store import (
    list_orders,
    reconcile_orders,
    record_order,
)


class _MockBroker:
    def __init__(self, orders):
        self._orders = orders

    def list_orders(self, limit=50, status="all"):
        return self._orders


def _submit_side_effect(url: str, coid: str):
    """Insert a pending_new row so reconcile has something to update."""
    o = PaperOrder(
        id="", client_order_id=coid,
        ticker="AAPL", side="buy", qty=10,
        order_type="market", status="pending_new",
        submitted_at="2026-08-28T10:00:00Z",
    )
    record_order(o, source="test", url=url)


def test_record_order_persists_triggering_news_ids(tmp_path):
    """Position-level news citations survive round-trip through paper_orders
    so the compare view can join filled trades back to headlines."""
    import json as _json
    url = f"sqlite:///{tmp_path / 'news_ids.db'}"
    o = PaperOrder(
        id="", client_order_id="ai-news-1",
        ticker="AAPL", side="buy", qty=10,
        order_type="market", status="pending_new",
        submitted_at="2026-09-16T10:00:00Z",
    )
    record_order(
        o, source="loop", url=url,
        triggering_news_ids=["N3f9a2b1c", "Na1b2c3d4"],
    )
    rows = list_orders(url=url)
    assert len(rows) == 1
    stored = _json.loads(rows[0]["triggering_news_ids"])
    assert stored == ["N3f9a2b1c", "Na1b2c3d4"]


def test_record_order_null_news_ids_stays_null(tmp_path):
    url = f"sqlite:///{tmp_path / 'no_news.db'}"
    o = PaperOrder(
        id="", client_order_id="ai-nonews-1",
        ticker="MSFT", side="sell", qty=5,
        order_type="market", status="pending_new",
        submitted_at="2026-09-16T10:05:00Z",
    )
    record_order(o, source="loop", url=url)
    rows = list_orders(url=url)
    assert rows[0]["triggering_news_ids"] is None


def test_reconcile_updates_pending_to_filled(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    _submit_side_effect(url, coid="ai-abc123")

    broker = _MockBroker([
        SimpleNamespace(
            client_order_id="ai-abc123",
            id="alp-42",
            status="filled",
            filled_at="2026-08-28T10:00:15Z",
            filled_avg_price=321.55,
        )
    ])
    n = reconcile_orders(broker, url=url)
    assert n == 1

    rows = list_orders(url=url)
    assert rows[0]["status"] == "filled"
    assert rows[0]["filled_avg_price"] == 321.55
    assert rows[0]["broker_order_id"] == "alp-42"


def test_reconcile_ignores_orders_we_didnt_submit(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    _submit_side_effect(url, coid="ai-mine")

    broker = _MockBroker([
        SimpleNamespace(client_order_id="somebody-elses",
                        id="alp-99", status="filled", filled_at="x",
                        filled_avg_price=1.0),
    ])
    n = reconcile_orders(broker, url=url)
    assert n == 0


def test_reconcile_survives_broker_outage(tmp_path):
    class _Broken:
        def list_orders(self, limit=50, status="all"):
            raise ConnectionError("network down")
    n = reconcile_orders(_Broken(), url=f"sqlite:///{tmp_path / 'test.db'}")
    assert n == 0
