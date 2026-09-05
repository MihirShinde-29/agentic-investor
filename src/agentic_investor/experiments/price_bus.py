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


def run_price_bus_writer(bus_url: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    from alpaca.data.live.stock import StockDataStream

    from agentic_investor.config import get_settings

    s = get_settings()
    db_path = bus_path_from_url(bus_url)
    init_price_bus_tables(db_path)
    logger.info("price bus writer starting -> %s", db_path)

    stream = StockDataStream(
        api_key=s.alpaca_api_key,
        secret_key=s.alpaca_api_secret,
    )
    subscribed: set[str] = set()
    lock = threading.Lock()

    async def on_trade(trade) -> None:
        try:
            symbol = str(getattr(trade, "symbol", "")).upper()
            price = float(getattr(trade, "price", 0.0))
            ts_event = str(getattr(trade, "timestamp", datetime.now(UTC)))
            if not symbol or price <= 0:
                return
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "INSERT INTO price_ticks "
                    "(ticker, price, ts_event, ts_recv) VALUES (?,?,?,?)",
                    (
                        symbol,
                        price,
                        ts_event,
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("price bus on_trade error: %s", e)

    stop = threading.Event()

    def _reconcile_loop() -> None:
        while not stop.is_set():
            try:
                desired = _read_desired_tickers(db_path)
                with lock:
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
            if stop.wait(_RECONCILE_INTERVAL_SEC):
                return

    reconciler = threading.Thread(
        target=_reconcile_loop, name="price-bus-reconcile", daemon=True,
    )
    reconciler.start()
    try:
        stream.run()
    finally:
        stop.set()
    return 0


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
        try:
            with sqlite3.connect(str(self._path)) as conn:
                row = conn.execute(
                    "SELECT price, ts_recv FROM price_ticks "
                    "WHERE ticker=? ORDER BY id DESC LIMIT 1",
                    (ticker.upper(),),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        if not row:
            return None
        price, ts_recv = row
        try:
            age = (
                datetime.now(UTC) - datetime.fromisoformat(ts_recv)
            ).total_seconds()
        except Exception:  # noqa: BLE001
            return None
        if age > max_age_sec:
            return None
        return float(price)

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
        import os
        bus_url = os.environ.get("AGENTIC_PRICE_BUS")
        arm_id = os.environ.get("AGENTIC_ARM_ID")
        if not bus_url or not arm_id:
            return None
        _client_singleton = PriceBusClient(bus_url, arm_id)
        return _client_singleton


def _reset_client_singleton_for_tests() -> None:
    global _client_singleton
    with _singleton_lock:
        _client_singleton = None
