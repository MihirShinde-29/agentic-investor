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
