"""Alpaca paper-trading client wrapper.

Wraps `alpaca-py` in a thin, testable surface: account info, positions, order
submission (market + optional bracket for stop/take-profit), and order history.
The broker is the source of truth for state; a local SQLite mirror in
paper_store.py keeps an audit trail for reports and reconciliation.

The default client points at Alpaca's paper endpoint. Live trading is off by
default and gated behind ALPACA_PAPER=false in the environment.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from agentic_investor.config import get_settings

logger = logging.getLogger(__name__)


# Domain types kept broker-agnostic so switching from Alpaca to IBKR later is
# a client swap, not a caller rewrite.


@dataclass
class PaperAccount:
    account_number: str
    cash: float
    equity: float
    buying_power: float
    portfolio_value: float


@dataclass
class PaperPosition:
    ticker: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class PaperClock:
    now: str  # ISO 8601
    is_open: bool
    next_open: str
    next_close: str


@dataclass
class PaperOrder:
    id: str  # broker order id
    client_order_id: str  # idempotency key we own
    ticker: str
    side: str  # "buy" | "sell"
    qty: float
    order_type: str  # "market" | "limit" | "bracket"
    status: str  # "new" | "accepted" | "partially_filled" | "filled" | "canceled" | "rejected"
    submitted_at: str
    filled_at: str | None = None
    filled_avg_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


# Broker Protocol - lets tests inject a fake without pulling in alpaca-py.


class PaperBroker(Protocol):
    def get_account(self) -> PaperAccount: ...
    def get_positions(self) -> list[PaperPosition]: ...
    def get_clock(self) -> PaperClock: ...
    def list_orders(self, limit: int = 50, status: str = "all") -> list[PaperOrder]: ...
    def submit_market_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        *,
        client_order_id: str | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> PaperOrder: ...
    def cancel_order(self, order_id: str) -> None: ...
    def close_position(self, ticker: str) -> PaperOrder: ...


# Real Alpaca client.


class AlpacaPaperBroker:
    """Concrete PaperBroker backed by alpaca-py's TradingClient.

    `account` selects which paper account's credentials to use:
      - "primary"   -> ALPACA_API_KEY   / ALPACA_API_SECRET
      - "secondary" -> ALPACA_API_KEY_B / ALPACA_API_SECRET_B
      - "tertiary"  -> ALPACA_API_KEY_C / ALPACA_API_SECRET_C
    Secondary and tertiary are the extra paper accounts used by the
    parallel-arm A/B/C experiment framework (M13).
    """

    _ACCOUNT_ENV = {
        "primary":   ("alpaca_api_key",   "alpaca_api_secret",   ""),
        "secondary": ("alpaca_api_key_b", "alpaca_api_secret_b", "_B"),
        "tertiary":  ("alpaca_api_key_c", "alpaca_api_secret_c", "_C"),
    }

    def __init__(self, client=None, *, account: str = "primary"):
        # Lazy import so tests can use the module without alpaca-py installed
        # (though it IS a dep, this keeps the import out of hot paths).
        if client is None:
            from alpaca.trading.client import TradingClient

            s = get_settings()
            if account not in self._ACCOUNT_ENV:
                raise ValueError(
                    f"unknown alpaca account {account!r} - "
                    f"expected one of {sorted(self._ACCOUNT_ENV)}"
                )
            key_attr, secret_attr, label = self._ACCOUNT_ENV[account]
            key, secret = getattr(s, key_attr), getattr(s, secret_attr)
            if not key or not secret:
                raise RuntimeError(
                    f"ALPACA_API_KEY{label} and ALPACA_API_SECRET{label} "
                    "must be set (get free paper keys at alpaca.markets)"
                )
            client = TradingClient(
                api_key=key, secret_key=secret, paper=s.alpaca_paper,
            )
        self._client = client
        self._account = account
        # Cache of ticker -> fractionable? so submit_market_order's asset
        # lookup runs once per symbol per process, not on every trade.
        self._fractionable_cache: dict[str, bool] = {}

    def _is_fractionable(self, ticker: str) -> bool:
        """Return True when Alpaca accepts fractional-share orders for the
        ticker. Cached in-process. Defaults to True on any lookup failure so
        we don't over-block; a real submit failure is still caught upstream.
        """
        key = ticker.upper()
        cached = self._fractionable_cache.get(key)
        if cached is not None:
            return cached
        try:
            asset = self._client.get_asset(key)
            frac = bool(getattr(asset, "fractionable", True))
        except Exception:  # noqa: BLE001
            frac = True
        self._fractionable_cache[key] = frac
        return frac

    def get_account(self) -> PaperAccount:
        a = self._client.get_account()
        return PaperAccount(
            account_number=str(a.account_number),
            cash=float(a.cash),
            equity=float(a.equity),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
        )

    def get_clock(self) -> PaperClock:
        c = self._client.get_clock()
        return PaperClock(
            now=str(c.timestamp),
            is_open=bool(c.is_open),
            next_open=str(c.next_open),
            next_close=str(c.next_close),
        )

    def get_positions(self) -> list[PaperPosition]:
        raw = self._client.get_all_positions()
        return [
            PaperPosition(
                ticker=str(p.symbol),
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_pl_pct=float(p.unrealized_plpc) * 100,
            )
            for p in raw
        ]

    def list_orders(self, limit: int = 50, status: str = "all") -> list[PaperOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        status_map = {
            "all": QueryOrderStatus.ALL,
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
        }
        req = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.ALL), limit=limit)
        raw = self._client.get_orders(filter=req)
        return [_alpaca_order_to_domain(o) for o in raw]

    def submit_market_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        *,
        client_order_id: str | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> PaperOrder:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        if side.lower() not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        # Fractionable check: some Alpaca assets (e.g. PS) only accept whole
        # shares. Round the qty down when the asset isn't fractionable so we
        # don't get 40310000 rejections. Cached on the broker instance so we
        # don't repay the get_asset() cost on every order.
        try:
            fractionable = self._is_fractionable(ticker)
            if not fractionable:
                whole = int(qty)
                if whole == 0:
                    raise ValueError(
                        f"{ticker} isn't fractionable and qty {qty} rounds to 0"
                    )
                qty = float(whole)
        except Exception as _e:  # noqa: BLE001 - fractionable check is best-effort
            logger.debug("fractionable check failed for %s: %s", ticker, _e)

        # We own the client_order_id so retries stay idempotent. Alpaca rejects
        # duplicate ids so an accidentally re-submitted order errors instead of
        # double-filling.
        coid = client_order_id or f"ai-{uuid.uuid4().hex[:16]}"

        # Bracket orders need a reference price to compute absolute levels.
        # For a MARKET buy we approximate with the latest trade price; for a
        # SELL we don't attach protection (we're closing risk, not opening).
        wants_bracket = (
            side.lower() == "buy"
            and (stop_loss_pct is not None or take_profit_pct is not None)
        )
        req: LimitOrderRequest | MarketOrderRequest
        if wants_bracket:
            ref_price = self._latest_trade_price(ticker)
            stop_loss = (
                StopLossRequest(stop_price=round(ref_price * (1 - stop_loss_pct / 100), 2))
                if stop_loss_pct
                else None
            )
            take_profit = (
                TakeProfitRequest(
                    limit_price=round(ref_price * (1 + take_profit_pct / 100), 2)
                )
                if take_profit_pct
                else None
            )
            order_class = (
                OrderClass.BRACKET
                if stop_loss and take_profit
                else OrderClass.OTO
            )
            # Bracket/OTO on Alpaca needs a limit parent, not market. Use a
            # slightly-through-market limit that behaves like a market fill
            # 99.5% of the time while satisfying Alpaca's bracket constraints.
            limit_price = round(ref_price * 1.005, 2)
            req = LimitOrderRequest(
                symbol=ticker.upper(),
                qty=qty,
                side=OrderSide(side.lower()),
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                order_class=order_class,
                stop_loss=stop_loss,
                take_profit=take_profit,
                client_order_id=coid,
            )
        else:
            req = MarketOrderRequest(
                symbol=ticker.upper(),
                qty=qty,
                side=OrderSide(side.lower()),
                time_in_force=TimeInForce.DAY,
                client_order_id=coid,
            )
        raw = self._client.submit_order(req)
        return _alpaca_order_to_domain(raw)

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)

    def close_position(self, ticker: str) -> PaperOrder:
        """Full liquidation of a position - Alpaca handles fractional shares."""
        raw = self._client.close_position(ticker.upper())
        return _alpaca_order_to_domain(raw)

    def _latest_trade_price(self, ticker: str) -> float:
        return get_latest_price(ticker)


def get_latest_price(ticker: str) -> float:
    """Latest trade price: shared price bus if available, else Alpaca REST,
    else yfinance."""
    from agentic_investor.experiments.price_bus import get_bus_client

    bus = get_bus_client()
    if bus is not None:
        try:
            bus.register({ticker})
            cached = bus.get_latest(ticker)
            if cached is not None:
                return cached
        except Exception as e:  # noqa: BLE001 - bus errors must never block trading
            logger.debug("price bus miss for %s: %s", ticker, e)
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        s = get_settings()
        if not s.alpaca_api_key or not s.alpaca_api_secret:
            raise RuntimeError("no alpaca credentials")
        data = StockHistoricalDataClient(
            api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret
        )
        resp = data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=ticker.upper())
        )
        return float(resp[ticker.upper()].price)
    except Exception as e:  # noqa: BLE001 - fall back on any failure
        logger.warning("alpaca latest-trade fallback for %s: %s", ticker, e)
        from agentic_investor.tools.market import fetch_ohlcv

        df = fetch_ohlcv(ticker.upper(), period="1y")
        return float(df["Close"].iloc[-1])


def _alpaca_order_to_domain(o) -> PaperOrder:
    """Coerce an alpaca-py Order into our domain PaperOrder."""

    def _f(v):
        return float(v) if v is not None else None

    def _s(v):
        return str(v) if v is not None else None

    submitted_at = _s(getattr(o, "submitted_at", None)) or datetime.utcnow().isoformat()
    filled_at = _s(getattr(o, "filled_at", None))
    raw_status = getattr(o, "status", None)
    status = str(raw_status).split(".")[-1].lower() if raw_status is not None else "new"
    order_type = str(getattr(o, "order_type", "market")).split(".")[-1].lower()
    return PaperOrder(
        id=str(o.id),
        client_order_id=str(getattr(o, "client_order_id", "") or ""),
        ticker=str(o.symbol),
        side=str(o.side).split(".")[-1].lower(),
        qty=float(o.qty or Decimal(0)),
        order_type=order_type,
        status=status,
        submitted_at=submitted_at,
        filled_at=filled_at,
        filled_avg_price=_f(getattr(o, "filled_avg_price", None)),
        raw={},
    )


def get_broker(*, account: str | None = None) -> PaperBroker:
    """Real Alpaca paper broker. Defaults to the arm context or primary."""
    if account is None:
        from agentic_investor.runtime_context import (
            get_active_alpaca_account,
        )
        account = get_active_alpaca_account() or "primary"
    return AlpacaPaperBroker(account=account)
