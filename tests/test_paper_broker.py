"""Tests for the paper-broker wrapper. Alpaca client is faked."""

from types import SimpleNamespace

from agentic_investor.tools.paper_broker import (
    AlpacaPaperBroker,
    PaperAccount,
    PaperOrder,
    PaperPosition,
    _alpaca_order_to_domain,
)


def _fake_order(**overrides):
    defaults = dict(
        id="alp-1", client_order_id="ai-abc123",
        symbol="AAPL", side="OrderSide.BUY", qty=10,
        order_type="OrderType.MARKET", status="OrderStatus.NEW",
        submitted_at="2026-08-27T10:00:00Z",
        filled_at=None, filled_avg_price=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeAlpacaClient:
    def __init__(self):
        self.submitted = []
        self._account = SimpleNamespace(
            account_number="PA123", cash=50000, equity=100000,
            buying_power=200000, portfolio_value=100000,
        )
        self._positions = [
            SimpleNamespace(
                symbol="AAPL", qty="10", avg_entry_price="150",
                market_value="1600", unrealized_pl="100",
                unrealized_plpc="0.0667",
            ),
        ]
        self._orders = [_fake_order()]

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return self._positions

    def get_orders(self, filter=None):  # noqa: A002 - matches alpaca-py signature
        return self._orders

    def submit_order(self, req):
        self.submitted.append(req)
        return _fake_order(client_order_id=req.client_order_id, symbol=req.symbol)

    def cancel_order_by_id(self, oid):
        self._orders = [o for o in self._orders if o.id != oid]


def test_get_account_maps_alpaca_fields():
    b = AlpacaPaperBroker(client=FakeAlpacaClient())
    acct = b.get_account()
    assert isinstance(acct, PaperAccount)
    assert acct.account_number == "PA123"
    assert acct.cash == 50000
    assert acct.equity == 100000


def test_get_positions_maps_and_percentages_unrealized_pl():
    b = AlpacaPaperBroker(client=FakeAlpacaClient())
    positions = b.get_positions()
    assert len(positions) == 1
    p = positions[0]
    assert isinstance(p, PaperPosition)
    assert p.ticker == "AAPL"
    # unrealized_plpc is 0.0667 fractional -> 6.67 pct
    assert round(p.unrealized_pl_pct, 2) == 6.67


def test_submit_market_order_uses_market_request_when_no_stops():
    from alpaca.trading.requests import MarketOrderRequest

    fake = FakeAlpacaClient()
    b = AlpacaPaperBroker(client=fake)
    order = b.submit_market_order("AAPL", "buy", 10)
    assert isinstance(order, PaperOrder)
    assert order.side == "buy"
    assert isinstance(fake.submitted[0], MarketOrderRequest)


def test_submit_market_order_generates_stable_client_order_id_when_omitted():
    fake = FakeAlpacaClient()
    b = AlpacaPaperBroker(client=fake)
    order = b.submit_market_order("AAPL", "buy", 10)
    assert order.client_order_id.startswith("ai-")
    assert len(order.client_order_id) > 4


def test_submit_bracket_order_when_stops_requested(monkeypatch):
    from alpaca.trading.requests import LimitOrderRequest

    fake = FakeAlpacaClient()
    b = AlpacaPaperBroker(client=fake)
    monkeypatch.setattr(b, "_latest_trade_price", lambda t: 100.0)
    b.submit_market_order(
        "AAPL", "buy", 10, stop_loss_pct=5.0, take_profit_pct=10.0,
    )
    req = fake.submitted[0]
    assert isinstance(req, LimitOrderRequest)
    assert req.stop_loss.stop_price == 95.0  # 5% below 100
    assert req.take_profit.limit_price == 110.0  # 10% above 100


def test_bracket_ignored_on_sell_side(monkeypatch):
    from alpaca.trading.requests import MarketOrderRequest

    fake = FakeAlpacaClient()
    b = AlpacaPaperBroker(client=fake)
    monkeypatch.setattr(b, "_latest_trade_price", lambda t: 100.0)
    # Sell should NOT wrap in a bracket - we're closing risk, not opening.
    b.submit_market_order("AAPL", "sell", 10, stop_loss_pct=5.0)
    assert isinstance(fake.submitted[0], MarketOrderRequest)


def test_side_validation_rejects_garbage():
    import pytest

    b = AlpacaPaperBroker(client=FakeAlpacaClient())
    with pytest.raises(ValueError, match="side must be"):
        b.submit_market_order("AAPL", "long", 10)


def test_get_latest_price_falls_back_to_yfinance_when_alpaca_fails(monkeypatch):
    """When Alpaca credentials are missing or the API errors, fall back to yfinance."""
    # Stub yfinance-side fetch to a known value.
    import pandas as pd

    from agentic_investor.tools import paper_broker as pb

    class _FakeDF:
        def __getitem__(self, k):
            return type("S", (), {"iloc": [123.45]})()

    monkeypatch.setattr(
        "agentic_investor.tools.market.fetch_ohlcv",
        lambda *a, **k: pd.DataFrame({"Close": [123.45]}),
    )
    # Force Alpaca path to fail by wiping settings.
    monkeypatch.setattr(pb, "get_settings", lambda: type("S", (), {
        "alpaca_api_key": None, "alpaca_api_secret": None,
    })())

    price = pb.get_latest_price("AAPL")
    assert price == 123.45


def test_alpaca_order_conversion_strips_enum_prefix():
    raw = _fake_order(status="OrderStatus.FILLED", side="OrderSide.SELL")
    o = _alpaca_order_to_domain(raw)
    assert o.status == "filled"
    assert o.side == "sell"
