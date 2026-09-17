"""Shared Alpaca market-data feed for N-arm experiments.

Alpaca allows one concurrent StockDataStream per API key. A single writer
owns the connection; the desired subscription set is the union of arms'
`price_subscriptions` rows, reconciled every few seconds. Arms upsert
their own rows via `PriceBusClient` and read the latest tick from
`price_ticks`.

Invariants: the writer never issues subscribe_trades for a ticker
already subscribed, never unsubscribes a ticker any arm still holds.
Both fall out of set arithmetic on the union - no coordination between
arms.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_RECONCILE_INTERVAL_SEC = 3.0
_SUBSCRIPTION_TTL_SEC = 300.0
_TICK_MAX_AGE_SEC = 30.0
_REGISTER_REFRESH_INTERVAL_SEC = 30.0


def bus_path_from_url(bus_url: str) -> Path:
    if not bus_url.startswith("sqlite:///"):
        raise ValueError(
            f"only sqlite:/// price-bus URLs supported (got {bus_url!r})"
        )
    return Path(bus_url.removeprefix("sqlite:///"))


def init_price_bus_tables(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        # WAL so N readers don't block the writer or each other.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_subscriptions (
                arm_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (arm_id, ticker)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                price REAL NOT NULL,
                ts_event TEXT NOT NULL,
                ts_recv TEXT NOT NULL
            )
            """
        )
        # Hot read path for get_latest.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_ticks_ticker_id "
            "ON price_ticks(ticker, id DESC)"
        )


def _read_desired_tickers(db_path: Path) -> set[str]:
    cutoff = (
        datetime.now(UTC) - timedelta(seconds=_SUBSCRIPTION_TTL_SEC)
    ).isoformat()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM price_subscriptions "
            "WHERE updated_at > ?",
            (cutoff,),
        ).fetchall()
    return {r[0] for r in rows}


def _insert_price_tick(
    db_path: Path,
    *,
    ticker: str,
    price: float,
    ts_event: str,
    ts_recv: str,
) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO price_ticks (ticker, price, ts_event, ts_recv) "
            "VALUES (?,?,?,?)",
            (ticker, price, ts_event, ts_recv),
        )


def _backfill_ticks_since(
    db_path: Path,
    tickers: list[str],
    since_iso: str | None,
    until_iso: str,
    *,
    max_window_min: float = 15.0,
) -> int:
    """REST-query Alpaca historic trades for each ticker in the current
    subscription set over `[since_iso, until_iso]` and insert missed
    ticks. Capped at max_window_min so a long disconnect doesn't chew
    Alpaca quota (Alpaca free tier: 200 req/min).

    Returns the number of ticks inserted across all tickers.
    """
    from datetime import timedelta

    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockTradesRequest

    from agentic_investor.config import get_settings

    s = get_settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret) or not tickers:
        return 0
    client = StockHistoricalDataClient(
        api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret,
    )

    until_dt = datetime.fromisoformat(until_iso.replace("Z", "+00:00"))
    if since_iso:
        since_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    else:
        since_dt = until_dt - timedelta(minutes=max_window_min)
    if until_dt - since_dt > timedelta(minutes=max_window_min):
        since_dt = until_dt - timedelta(minutes=max_window_min)

    now_iso = datetime.now(UTC).isoformat()
    n_total = 0
    for symbol in tickers:
        try:
            req = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=since_dt,
                end=until_dt,
                limit=200,
            )
            resp = client.get_stock_trades(req)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "price backfill REST call failed for %s: %s", symbol, e,
            )
            continue
        # StockTradesResponse.data is dict[symbol, list[Trade]]
        for trades in getattr(resp, "data", {}).values():
            for t in trades:
                price = float(getattr(t, "price", 0.0))
                if price <= 0:
                    continue
                ts_event = str(getattr(t, "timestamp", None) or now_iso)
                _insert_price_tick(
                    db_path,
                    ticker=symbol.upper(),
                    price=price,
                    ts_event=ts_event,
                    ts_recv=now_iso,
                )
                n_total += 1
    return n_total


def run_price_bus_writer(bus_url: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    from alpaca.data.live.stock import StockDataStream

    from agentic_investor.config import get_settings
    from agentic_investor.experiments._bus_purge import start_purge_thread
    from agentic_investor.experiments._bus_stream import (
        BusStatus,
        run_with_reconnect,
    )
    from agentic_investor.flags import flags

    s = get_settings()
    db_path = bus_path_from_url(bus_url)
    init_price_bus_tables(db_path)
    status = BusStatus.open(db_path)
    logger.info("price bus writer starting -> %s", db_path)

    # TTL purge: drop price_ticks older than PRICE_BUS_TTL_HOURS (default
    # 2h; only current-session ticks are needed by the reaction-price
    # fetcher + drift calc). Sweep every 5 min. price_subscriptions has
    # its own 5-min-window TTL in `_read_desired_tickers`, no purge
    # needed there.
    price_ttl_h = flags.PRICE_BUS_TTL_HOURS

    def _purge_ticks(conn):
        from datetime import timedelta
        cutoff = (
            datetime.now(UTC) - timedelta(hours=price_ttl_h)
        ).isoformat()
        cur = conn.execute(
            "DELETE FROM price_ticks WHERE ts_recv < ?", (cutoff,),
        )
        return cur.rowcount

    start_purge_thread(
        db_path, _purge_ticks,
        interval_sec=300.0,
        label="price_bus",
    )

    # Stream state that reconciler thread + reconnect supervisor share.
    # `stream` is swapped on each reconnect; `subscribed` resets to empty
    # because the fresh stream has no active subscriptions.
    state: dict = {"stream": None, "subscribed": set()}
    sub_lock = threading.Lock()
    reconciler_stop = threading.Event()

    async def on_trade(trade) -> None:
        try:
            symbol = str(getattr(trade, "symbol", "")).upper()
            price = float(getattr(trade, "price", 0.0))
            ts_event = str(getattr(trade, "timestamp", datetime.now(UTC)))
            if not symbol or price <= 0:
                return
            ts_recv = datetime.now(UTC).isoformat()
            _insert_price_tick(
                db_path, ticker=symbol, price=price,
                ts_event=ts_event, ts_recv=ts_recv,
            )
            status.record_event(ts_event)
        except Exception as e:  # noqa: BLE001
            logger.warning("price bus on_trade error: %s", e)

    def _reconcile_loop() -> None:
        while not reconciler_stop.is_set():
            try:
                desired = _read_desired_tickers(db_path)
                with sub_lock:
                    stream = state["stream"]
                    if stream is None:
                        # No active stream (mid-reconnect); wait.
                        pass
                    else:
                        subscribed = state["subscribed"]
                        to_add = desired - subscribed
                        to_remove = subscribed - desired
                        if to_add:
                            stream.subscribe_trades(on_trade, *sorted(to_add))
                            subscribed.update(to_add)
                            logger.info(
                                "price bus subscribed: %s", sorted(to_add),
                            )
                        if to_remove:
                            stream.unsubscribe_trades(*sorted(to_remove))
                            subscribed.difference_update(to_remove)
                            logger.info(
                                "price bus unsubscribed: %s", sorted(to_remove),
                            )
            except Exception as e:  # noqa: BLE001 - reconcile must not die
                logger.warning("price bus reconcile error: %s", e)
            if reconciler_stop.wait(_RECONCILE_INTERVAL_SEC):
                return

    reconciler = threading.Thread(
        target=_reconcile_loop, name="price-bus-reconcile", daemon=True,
    )
    reconciler.start()

    def _factory():
        stream = StockDataStream(
            api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret,
        )
        # Publish the fresh stream + reset subscribed set. Reconciler
        # will re-add the current desired-ticker set on its next tick.
        with sub_lock:
            state["stream"] = stream
            state["subscribed"] = set()
        return stream

    def _run(stream):
        try:
            stream.run()
        finally:
            with sub_lock:
                # Stream is gone; reconciler must not try to
                # subscribe/unsubscribe on it until a new one is minted.
                state["stream"] = None

    def _backfill(before_ts_iso, now_iso):
        # Ticker set to backfill = current subscription set (what arms
        # are actually watching right now). Deliberately NOT the pre-
        # disconnect subscription set, because arms may have moved on.
        with sub_lock:
            tickers = sorted(_read_desired_tickers(db_path))
        n = _backfill_ticks_since(db_path, tickers, before_ts_iso, now_iso)
        if n > 0:
            status.record_event(now_iso, count=n)
        return n

    try:
        return run_with_reconnect(
            stream_factory=_factory,
            run_stream=_run,
            on_reconnect=_backfill,
            status=status,
            label="price-bus",
        )
    finally:
        reconciler_stop.set()


class PriceBusClient:
    """Arm-side helper: register interest in tickers, read cached prices."""

    def __init__(self, bus_url: str, arm_id: str):
        self._url = bus_url
        self._arm_id = arm_id
        self._path = bus_path_from_url(bus_url)
        self._last_registered: dict[str, float] = {}
        self._last_unregistered: dict[str, float] = {}

    def register(self, tickers: set[str]) -> None:
        """Upsert this arm's interest; debounced per-ticker."""
        if not tickers:
            return
        now_mono = time.monotonic()
        due = [
            t.upper() for t in tickers
            if now_mono - self._last_registered.get(t.upper(), 0.0)
            > _REGISTER_REFRESH_INTERVAL_SEC
        ]
        if not due:
            return
        now_iso = datetime.now(UTC).isoformat()
        with sqlite3.connect(str(self._path)) as conn:
            for t in due:
                conn.execute(
                    "INSERT INTO price_subscriptions "
                    "(arm_id, ticker, updated_at) VALUES (?,?,?) "
                    "ON CONFLICT(arm_id, ticker) DO UPDATE "
                    "SET updated_at=excluded.updated_at",
                    (self._arm_id, t, now_iso),
                )
                self._last_registered[t] = now_mono
                self._last_unregistered.pop(t, None)

    def unregister(self, tickers: set[str]) -> None:
        """Drop this arm's interest; debounced per-ticker."""
        if not tickers:
            return
        now_mono = time.monotonic()
        due = [
            t.upper() for t in tickers
            if now_mono - self._last_unregistered.get(t.upper(), 0.0)
            > _REGISTER_REFRESH_INTERVAL_SEC
        ]
        if not due:
            return
        with sqlite3.connect(str(self._path)) as conn:
            for t in due:
                conn.execute(
                    "DELETE FROM price_subscriptions "
                    "WHERE arm_id=? AND ticker=?",
                    (self._arm_id, t),
                )
                self._last_unregistered[t] = now_mono
                self._last_registered.pop(t, None)

    def get_latest(
        self,
        ticker: str,
        max_age_sec: float = _TICK_MAX_AGE_SEC,
    ) -> float | None:
        # Deterministic replay hook (task #153): find the next
        # recorded price row for THIS ticker, leaving rows for other
        # tickers in place for their own future callers. next_from_source
        # with a match filter does the ordered scan for us.
        from agentic_investor.orchestrator.recorder import (
            next_from_source,
            record_source,
        )
        replayed = next_from_source(
            "price", match={"ticker": ticker.upper()},
        )
        if replayed is not None:
            price = replayed.get("price")
            return float(price) if price is not None else None
        try:
            with sqlite3.connect(str(self._path)) as conn:
                row = conn.execute(
                    "SELECT price, ts_recv FROM price_ticks "
                    "WHERE ticker=? ORDER BY id DESC LIMIT 1",
                    (ticker.upper(),),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        result: float | None
        if not row:
            result = None
        else:
            price, ts_recv = row
            try:
                age = (
                    datetime.now(UTC) - datetime.fromisoformat(ts_recv)
                ).total_seconds()
            except Exception:  # noqa: BLE001
                result = None
            else:
                result = float(price) if age <= max_age_sec else None
        record_source("price", {
            "ticker": ticker.upper(),
            "price": result,
        })
        return result

    def list_my_tickers(self) -> set[str]:
        try:
            with sqlite3.connect(str(self._path)) as conn:
                rows = conn.execute(
                    "SELECT ticker FROM price_subscriptions WHERE arm_id=?",
                    (self._arm_id,),
                ).fetchall()
        except sqlite3.OperationalError:
            return set()
        return {r[0] for r in rows}

    def set_subscriptions(self, tickers: set[str]) -> None:
        """Replace this arm's subscription set atomically.

        Drop path bypasses the unregister debounce - an explicit "no
        longer want" should apply immediately, not wait out the TTL.
        """
        desired = {t.upper() for t in tickers}
        current = self.list_my_tickers()
        to_add = desired - current
        to_drop = current - desired
        to_keep = desired & current
        if to_add:
            self.register(to_add)
        if to_drop:
            with sqlite3.connect(str(self._path)) as conn:
                for t in to_drop:
                    conn.execute(
                        "DELETE FROM price_subscriptions "
                        "WHERE arm_id=? AND ticker=?",
                        (self._arm_id, t),
                    )
                    self._last_unregistered[t] = time.monotonic()
                    self._last_registered.pop(t, None)
        if to_keep:
            self.register(to_keep)

    def get_latest_batch(
        self,
        tickers: set[str] | None = None,
        max_age_sec: float = _TICK_MAX_AGE_SEC,
    ) -> dict[str, float | None]:
        if tickers is None:
            tickers = self.list_my_tickers()
        return {t.upper(): self.get_latest(t, max_age_sec) for t in tickers}


_client_singleton: PriceBusClient | None = None
_singleton_lock = threading.Lock()


def get_bus_client() -> PriceBusClient | None:
    """Return the per-process client if the arm is running in bus mode."""
    global _client_singleton
    with _singleton_lock:
        if _client_singleton is not None:
            return _client_singleton
        from agentic_investor.flags import flags
        bus_url = flags.PRICE_BUS
        arm_id = flags.ARM_ID
        # ARM_ID has a "solo" default so we key off the bus URL alone as
        # the "bus mode is on" signal - arms with bus configured but no
        # explicit ARM_ID are still routable ("solo" is a valid arm tag).
        if not bus_url:
            return None
        _client_singleton = PriceBusClient(bus_url, arm_id)
        return _client_singleton


def _reset_client_singleton_for_tests() -> None:
    global _client_singleton
    with _singleton_lock:
        _client_singleton = None
