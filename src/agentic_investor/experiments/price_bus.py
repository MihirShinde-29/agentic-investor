"""Shared market-data streaming bus for parallel experiment arms.

Alpaca's live market-data websocket has the same 1-concurrent-connection-
per-key limit that the news stream does, so N arms can't each open their
own StockDataStream. This module mirrors the news_bus fanout pattern but
adds one wrinkle: the desired subscription set is dynamic. Each arm's
held + on-deck ticker set changes as the loop trades; the bus writer has
to reconcile the union of all arms' desired sets against what it's
currently subscribed to on Alpaca.

Two-table schema on a shared SQLite file:

    price_subscriptions(arm_id, ticker, updated_at)
        - upsert-on-touch: each arm re-registers a ticker every ~30s to
          keep its row's updated_at fresh
        - rows older than _SUBSCRIPTION_TTL_SEC get pruned from the
          desired set (arm process died, ticker fell off on-deck, etc.)

    price_ticks(id, ticker, price, ts_event, ts_recv)
        - append-only trade log written by the bus writer as trades
          arrive from Alpaca
        - arms read latest-by-ticker for the cache path in
          get_latest_price

Idempotency contract (per user requirement):
  - Writer never issues subscribe_trades for a ticker it's already
    subscribed to.
  - Writer never issues unsubscribe_trades for a ticker it isn't
    subscribed to.
  - Arm-side client's register/unregister short-circuit if the local
    in-process cache says the (arm_id, ticker) state hasn't changed
    within its own TTL.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# How often the writer diffs (desired union) vs (currently subscribed on
# Alpaca) and issues the delta subscribe/unsubscribe calls. 3s keeps the
# add-latency low (arm registers -> bus subscribes -> first trade lands)
# without hammering Alpaca's control channel.
_RECONCILE_INTERVAL_SEC = 3.0

# Any subscription row older than this gets pruned from the desired set,
# even if the arm process is still up. Forces arms to periodically renew
# their interest, which is the cleanup mechanism when an arm dies or
# drops a ticker from its held/on-deck set.
_SUBSCRIPTION_TTL_SEC = 300.0

# How stale a cached tick can be before get_latest_price falls through
# to REST instead of returning the cached price. 30s is chosen so a
# ticker that stopped trading for a moment doesn't wrongly report a
# stale price - fresh REST poll gives the actual latest.
_TICK_MAX_AGE_SEC = 30.0

# How often each arm-side client re-registers its interest in a ticker
# to keep the subscription row fresh. Kept well under _SUBSCRIPTION_TTL_SEC
# so there's slack for slow ticks.
_REGISTER_REFRESH_INTERVAL_SEC = 30.0


def bus_path_from_url(bus_url: str) -> Path:
    if not bus_url.startswith("sqlite:///"):
        raise ValueError(
            f"only sqlite:/// price-bus URLs supported (got {bus_url!r})"
        )
    return Path(bus_url.removeprefix("sqlite:///"))


def init_price_bus_tables(db_path: Path) -> None:
    """Create both tables + WAL mode for concurrent reader/writers."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
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
        # Latest-per-ticker lookup is the hot read path for get_latest_price.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_ticks_ticker_id "
            "ON price_ticks(ticker, id DESC)"
        )


def _read_desired_tickers(db_path: Path) -> set[str]:
    """Return the union of arms' active subscriptions (TTL-pruned)."""
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
    """Blocking. Owns the single Alpaca StockDataStream for the experiment.

    Uses primary account keys (IEX feed on free tier). Runs a background
    reconcile thread that diffs desired vs subscribed and calls the
    delta only, then blocks on stream.run().
    """
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
        """Diff desired vs subscribed every N seconds; add/remove the delta."""
        while not stop.is_set():
            try:
                desired = _read_desired_tickers(db_path)
                with lock:
                    to_add = desired - subscribed
                    to_remove = subscribed - desired
                    # Idempotency per user requirement: only call
                    # subscribe/unsubscribe when there's actual delta.
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
    """Arm-side helper: register interest in tickers, read cached prices.

    Held as a per-process singleton by get_bus_client so we don't build
    fresh SQLite connections in the hot get_latest_price path.
    """

    def __init__(self, bus_url: str, arm_id: str):
        self._url = bus_url
        self._arm_id = arm_id
        self._path = bus_path_from_url(bus_url)
        # ticker -> monotonic ts of last upsert. Debounces re-registration
        # so we're not writing on every get_latest_price call.
        self._last_registered: dict[str, float] = {}
        # ticker -> monotonic ts of last delete. Symmetric debounce for
        # unregister so double-drops don't re-issue redundant DELETEs.
        self._last_unregistered: dict[str, float] = {}

    def register(self, tickers: set[str]) -> None:
        """Idempotent: upsert this arm's interest in each ticker.

        Skips the write for tickers already registered within the last
        _REGISTER_REFRESH_INTERVAL_SEC, so re-calling register() on every
        tick doesn't spam the subscriptions table.
        """
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
                # ON CONFLICT ensures the row exists exactly once per
                # (arm_id, ticker); the update just bumps updated_at.
                conn.execute(
                    "INSERT INTO price_subscriptions "
                    "(arm_id, ticker, updated_at) VALUES (?,?,?) "
                    "ON CONFLICT(arm_id, ticker) DO UPDATE "
                    "SET updated_at=excluded.updated_at",
                    (self._arm_id, t, now_iso),
                )
                self._last_registered[t] = now_mono
                # If we're re-registering, we're no longer unregistered.
                self._last_unregistered.pop(t, None)

    def unregister(self, tickers: set[str]) -> None:
        """Idempotent: drop this arm's interest in each ticker.

        Skips the DELETE for tickers already unregistered within the last
        _REGISTER_REFRESH_INTERVAL_SEC. If we already told the DB we don't
        want a ticker, telling it again is a no-op.
        """
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
        """Return latest cached price for ticker if fresh, else None."""
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
        """Return the set of tickers this arm currently has subscribed."""
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
        """Replace this arm's subscription set to exactly `tickers`.

        Diffs against the arm's *current* rows in price_subscriptions
        and does the minimum work:
          - tickers new to this arm  -> INSERT (writer will subscribe
            if not already subscribed for another arm)
          - tickers dropped by this arm -> DELETE (writer will
            unsubscribe only if no other arm still has the row)
          - tickers already present -> refresh updated_at so the row
            doesn't age out of the desired set (subject to the
            in-process register-refresh debounce)

        Does not touch other arms' rows. The union-based reconciliation
        in the writer is what enforces "don't unsubscribe if another arm
        still needs it".
        """
        desired = {t.upper() for t in tickers}
        current = self.list_my_tickers()
        to_add = desired - current
        to_drop = current - desired
        to_keep = desired & current
        if to_add:
            self.register(to_add)
        if to_drop:
            # Bypass the unregister-debounce for the explicit-set path:
            # if the caller declared they no longer want a ticker, we
            # apply that immediately rather than waiting for the
            # in-process TTL. Debounce is for accidental re-drops.
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
            # Refresh updated_at (subject to register-debounce inside
            # register()) so kept tickers don't age out.
            self.register(to_keep)

    def get_latest_batch(
        self,
        tickers: set[str] | None = None,
        max_age_sec: float = _TICK_MAX_AGE_SEC,
    ) -> dict[str, float | None]:
        """Return {ticker: latest_price_or_None} for a set of tickers.

        If `tickers` is None, defaults to *this arm's* current
        subscription list (list_my_tickers). That's the natural
        "give me prices for everything I care about" call.
        """
        if tickers is None:
            tickers = self.list_my_tickers()
        return {t.upper(): self.get_latest(t, max_age_sec) for t in tickers}


# Per-process singleton so get_latest_price doesn't allocate fresh
# SQLite connections + register-debounce dicts on every call.
_client_singleton: PriceBusClient | None = None
_singleton_lock = threading.Lock()


def get_bus_client() -> PriceBusClient | None:
    """Return the per-process PriceBusClient if the arm is in bus mode.

    Reads AGENTIC_PRICE_BUS + AGENTIC_ARM_ID env vars. Returns None if
    either is missing (which means we're a plain single-arm paper-loop
    and should stick with the direct REST path).
    """
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
    """Test-only hook to force get_bus_client to rebuild from current env."""
    global _client_singleton
    with _singleton_lock:
        _client_singleton = None
