"""Price-bus fanout: writer reconciles arm subscriptions idempotently,
arm-side client debounces re-registration, get_latest_price prefers cache.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def _insert_tick(db_path: Path, ticker: str, price: float,
                 recv_offset_sec: float = 0.0) -> None:
    ts = (datetime.now(UTC) - timedelta(seconds=recv_offset_sec)).isoformat()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO price_ticks (ticker, price, ts_event, ts_recv) "
            "VALUES (?,?,?,?)",
            (ticker.upper(), price, ts, ts),
        )


def test_init_price_bus_creates_both_tables(tmp_path):
    from agentic_investor.experiments.price_bus import init_price_bus_tables

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    with sqlite3.connect(str(db)) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "price_subscriptions" in tables
    assert "price_ticks" in tables


def test_client_register_upserts_row(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    c.register({"AAPL", "MSFT"})
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT arm_id, ticker FROM price_subscriptions ORDER BY ticker"
        ).fetchall()
    assert rows == [("A", "AAPL"), ("A", "MSFT")]


def test_client_register_debounces_within_ttl(tmp_path):
    """Repeated register() calls for the same ticker within the TTL
    should not re-issue the UPSERT (writes are debounced client-side)."""
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    c.register({"AAPL"})
    # Grab the initial updated_at.
    with sqlite3.connect(str(db)) as conn:
        first = conn.execute(
            "SELECT updated_at FROM price_subscriptions WHERE ticker='AAPL'"
        ).fetchone()[0]
    time.sleep(0.05)
    # Second call within debounce window: should NOT update the row.
    c.register({"AAPL"})
    with sqlite3.connect(str(db)) as conn:
        second = conn.execute(
            "SELECT updated_at FROM price_subscriptions WHERE ticker='AAPL'"
        ).fetchone()[0]
    assert first == second, "debounce should have skipped the re-upsert"


def test_client_register_writes_after_ttl_expires(tmp_path, monkeypatch):
    from agentic_investor.experiments import price_bus
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    monkeypatch.setattr(
        price_bus, "_REGISTER_REFRESH_INTERVAL_SEC", 0.05,
    )
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    c.register({"AAPL"})
    with sqlite3.connect(str(db)) as conn:
        first = conn.execute(
            "SELECT updated_at FROM price_subscriptions WHERE ticker='AAPL'"
        ).fetchone()[0]
    time.sleep(0.1)
    c.register({"AAPL"})
    with sqlite3.connect(str(db)) as conn:
        second = conn.execute(
            "SELECT updated_at FROM price_subscriptions WHERE ticker='AAPL'"
        ).fetchone()[0]
    assert first != second, "after TTL expiry the upsert must run"


def test_client_unregister_removes_row_idempotently(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    c.register({"AAPL"})
    c.unregister({"AAPL"})
    with sqlite3.connect(str(db)) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM price_subscriptions"
        ).fetchone()[0]
    assert n == 0
    # Second unregister within debounce: should not error, should not
    # re-issue the DELETE.
    c.unregister({"AAPL"})


def test_get_latest_returns_fresh_price(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    _insert_tick(db, "AAPL", 189.42)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    assert c.get_latest("AAPL") == pytest.approx(189.42)


def test_get_latest_returns_none_when_stale(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    _insert_tick(db, "AAPL", 189.42, recv_offset_sec=60.0)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    # max_age = 30s by default; 60s-old tick should be dropped.
    assert c.get_latest("AAPL") is None


def test_get_latest_returns_latest_when_multiple(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    _insert_tick(db, "AAPL", 100.0)
    _insert_tick(db, "AAPL", 101.0)
    _insert_tick(db, "AAPL", 102.0)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    assert c.get_latest("AAPL") == pytest.approx(102.0)


def test_get_latest_returns_none_for_unknown_ticker(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    assert c.get_latest("NOSUCH") is None


def test_read_desired_tickers_unions_arms_and_ttl_prunes(tmp_path,
                                                          monkeypatch):
    """Desired set should be the union across arms with stale rows pruned."""
    from agentic_investor.experiments import price_bus
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        _read_desired_tickers,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    # Shrink TTL so we can prove pruning without waiting 5 minutes.
    monkeypatch.setattr(price_bus, "_SUBSCRIPTION_TTL_SEC", 0.5)
    ca = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    cb = PriceBusClient(f"sqlite:///{db}", arm_id="B")
    ca.register({"AAPL", "MSFT"})
    cb.register({"MSFT", "NVDA"})
    desired = _read_desired_tickers(db)
    assert desired == {"AAPL", "MSFT", "NVDA"}
    # Age the rows past the TTL, verify they're pruned.
    time.sleep(0.7)
    desired = _read_desired_tickers(db)
    assert desired == set()


def test_get_bus_client_returns_none_without_env(monkeypatch):
    from agentic_investor.experiments import price_bus

    monkeypatch.delenv("AGENTIC_PRICE_BUS", raising=False)
    monkeypatch.delenv("AGENTIC_ARM_ID", raising=False)
    price_bus._reset_client_singleton_for_tests()
    assert price_bus.get_bus_client() is None


def test_get_bus_client_returns_client_with_env(tmp_path, monkeypatch):
    from agentic_investor.experiments import price_bus
    from agentic_investor.experiments.price_bus import PriceBusClient

    db = tmp_path / "pb.db"
    price_bus.init_price_bus_tables(db)
    monkeypatch.setenv("AGENTIC_PRICE_BUS", f"sqlite:///{db}")
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    price_bus._reset_client_singleton_for_tests()
    c = price_bus.get_bus_client()
    assert isinstance(c, PriceBusClient)
    # Singleton: second call returns same instance.
    assert price_bus.get_bus_client() is c


def test_bus_path_from_url_rejects_non_sqlite():
    from agentic_investor.experiments.price_bus import bus_path_from_url

    with pytest.raises(ValueError, match="only sqlite"):
        bus_path_from_url("postgres://oops")


def test_get_latest_price_hits_bus_cache(tmp_path, monkeypatch):
    """When AGENTIC_PRICE_BUS is set and cache is fresh,
    get_latest_price must return the cached value without hitting Alpaca."""
    from agentic_investor.experiments import price_bus
    from agentic_investor.experiments.price_bus import init_price_bus_tables
    from agentic_investor.tools import paper_broker

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    _insert_tick(db, "AAPL", 555.55)
    monkeypatch.setenv("AGENTIC_PRICE_BUS", f"sqlite:///{db}")
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    price_bus._reset_client_singleton_for_tests()

    # Sabotage the REST path so we KNOW the bus is what returned the price.
    def _explode(*_a, **_kw):
        raise AssertionError(
            "REST path called even though bus cache was fresh"
        )

    monkeypatch.setattr(
        "alpaca.data.historical.StockHistoricalDataClient", _explode,
    )
    price = paper_broker.get_latest_price("AAPL")
    assert price == pytest.approx(555.55)


def test_desired_union_keeps_ticker_when_only_one_arm_drops_it(tmp_path):
    """The critical multi-arm invariant: if arm A drops X but arm B
    still has X, the writer's desired set (union across arms) must
    still contain X, so the reconcile loop doesn't unsubscribe it."""
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        _read_desired_tickers,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    ca = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    cb = PriceBusClient(f"sqlite:///{db}", arm_id="B")
    ca.register({"AAPL", "MSFT"})
    cb.register({"AAPL"})
    assert _read_desired_tickers(db) == {"AAPL", "MSFT"}
    # Arm A drops AAPL via set_subscriptions. Arm B still has it.
    ca.set_subscriptions({"MSFT"})
    desired = _read_desired_tickers(db)
    # AAPL must still be desired because arm B has it.
    assert "AAPL" in desired
    assert desired == {"AAPL", "MSFT"}
    # Now arm B also drops AAPL: only then should it fall out.
    cb.set_subscriptions(set())
    desired = _read_desired_tickers(db)
    assert "AAPL" not in desired
    assert desired == {"MSFT"}


def test_set_subscriptions_diffs_add_drop_keep(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    c = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    c.set_subscriptions({"AAPL", "MSFT", "NVDA"})
    assert c.list_my_tickers() == {"AAPL", "MSFT", "NVDA"}
    # Replace: drop NVDA, keep AAPL/MSFT, add TSLA + GOOGL.
    c.set_subscriptions({"AAPL", "MSFT", "TSLA", "GOOGL"})
    assert c.list_my_tickers() == {"AAPL", "MSFT", "TSLA", "GOOGL"}


def test_set_subscriptions_empty_drops_all_for_this_arm_only(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    ca = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    cb = PriceBusClient(f"sqlite:///{db}", arm_id="B")
    ca.register({"AAPL", "MSFT"})
    cb.register({"AAPL", "GOOGL"})
    ca.set_subscriptions(set())
    assert ca.list_my_tickers() == set()
    # Arm B's rows must be untouched.
    assert cb.list_my_tickers() == {"AAPL", "GOOGL"}


def test_list_my_tickers_is_per_arm(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    ca = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    cb = PriceBusClient(f"sqlite:///{db}", arm_id="B")
    ca.register({"AAPL"})
    cb.register({"MSFT", "GOOGL"})
    assert ca.list_my_tickers() == {"AAPL"}
    assert cb.list_my_tickers() == {"MSFT", "GOOGL"}


def test_get_latest_batch_defaults_to_my_tickers(tmp_path):
    from agentic_investor.experiments.price_bus import (
        PriceBusClient,
        init_price_bus_tables,
    )

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    ca = PriceBusClient(f"sqlite:///{db}", arm_id="A")
    cb = PriceBusClient(f"sqlite:///{db}", arm_id="B")
    ca.register({"AAPL", "MSFT"})
    cb.register({"GOOGL"})
    _insert_tick(db, "AAPL", 100.0)
    _insert_tick(db, "MSFT", 200.0)
    _insert_tick(db, "GOOGL", 300.0)  # arm A shouldn't see this in default
    batch = ca.get_latest_batch()
    assert set(batch.keys()) == {"AAPL", "MSFT"}
    assert batch["AAPL"] == pytest.approx(100.0)
    assert batch["MSFT"] == pytest.approx(200.0)


def test_get_latest_price_registers_ticker_on_first_call(tmp_path,
                                                         monkeypatch):
    """First get_latest_price for a ticker should upsert into
    price_subscriptions so the bus writer will subscribe to it."""
    from agentic_investor.experiments import price_bus
    from agentic_investor.experiments.price_bus import init_price_bus_tables
    from agentic_investor.tools import paper_broker

    db = tmp_path / "pb.db"
    init_price_bus_tables(db)
    monkeypatch.setenv("AGENTIC_PRICE_BUS", f"sqlite:///{db}")
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    price_bus._reset_client_singleton_for_tests()

    # Sabotage REST + yfinance so both fail loudly; we only care that
    # the register-side effect happened, not what the price came back as.
    def _fail(*_a, **_kw):
        raise RuntimeError("simulated network fail")

    monkeypatch.setattr(
        "alpaca.data.historical.StockHistoricalDataClient", _fail,
    )
    monkeypatch.setattr(
        "agentic_investor.tools.market.fetch_ohlcv", _fail,
    )
    with pytest.raises(RuntimeError):
        paper_broker.get_latest_price("TSLA")

    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT arm_id, ticker FROM price_subscriptions"
        ).fetchall()
    assert rows == [("A", "TSLA")]
