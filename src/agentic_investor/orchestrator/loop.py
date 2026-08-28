"""Continuous paper-trading loop with two-tier decision cadence.

Real quant systems separate "model output" (slow, expensive: fresh news +
technicals + allocator) from "execution" (fast, cheap: check drift, place
orders). We do the same:

- Once per trading day at first tick: regenerate a Recommendation (LLM cost).
- Every tick after: compute drift vs target; if any position drifts beyond
  band_abs_pct, submit the diff. Snapshot account + positions each tick.

Market-hours awareness comes from Alpaca's own clock (holiday-safe). Outside
market hours the loop sleeps until next_open rather than ticking uselessly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from agentic_investor.orchestrator.rebalancer import (
    compute_trade_plan,
    execute_trade_plan,
)
from agentic_investor.orchestrator.state import OrchestratorRequest, Recommendation
from agentic_investor.tools.paper_broker import PaperBroker, PaperOrder
from agentic_investor.tools.paper_store import record_order, record_snapshot

# Event-driven imports live at call time to avoid pulling websocket deps
# for anyone using the simpler --regen-mode daily path.

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """User-supplied config for one paper-loop invocation."""

    profile_name: str = "moderate"
    amount: float = 10_000.0

    # Ticker selection: either explicit list OR auto-pick from a universe.
    tickers: list[str] = field(default_factory=list)
    auto: bool = False
    universe: str = "mega_tech"
    top_n: int = 8

    # Cadence + risk
    interval_seconds: int = 30 * 60  # 30 min default
    band_abs_pct: float = 5.0  # only rebalance when a position drifts this many pp
    min_trade_dollars: float = 50.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    # Ops toggles
    dry_run: bool = False
    once: bool = False  # run one tick then exit (for testing)


@dataclass
class TickResult:
    tick_at: str
    rec_id: int | None
    regenerated_rec: bool
    plan_count: int
    submitted: list[PaperOrder]
    equity: float
    error: str | None = None


@dataclass
class LoopState:
    last_rec_id: int | None = None
    last_rec_date: str | None = None  # "YYYY-MM-DD" of last regeneration
    ticks_run: int = 0
    orders_submitted: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _drift_exceeds_band(
    rec: Recommendation,
    positions_dollars: dict[str, float],
    total_equity: float,
    band_abs_pct: float,
) -> bool:
    """Any position (target OR current) drifted more than band_abs_pct from target?"""
    if total_equity <= 0:
        return True
    target = {p.ticker.upper(): p.weight_pct for p in rec.allocation.positions}
    current_pct = {
        t: (v / total_equity) * 100.0 for t, v in positions_dollars.items()
    }
    tickers = set(target) | set(current_pct)
    for t in tickers:
        drift = abs(target.get(t, 0.0) - current_pct.get(t, 0.0))
        if drift >= band_abs_pct:
            return True
    return False


def _generate_recommendation(
    cfg: LoopConfig,
    *,
    as_of: str | None = None,
) -> Recommendation:
    """Run the orchestrator once for today's decision. Uses the M6 profile."""
    from agentic_investor.orchestrator.graph import run_orchestrator
    from agentic_investor.orchestrator.strategy import load_profile

    profile = load_profile(cfg.profile_name)
    tickers = list(cfg.tickers)
    if cfg.auto:
        from agentic_investor.orchestrator.picker import pick_top_n
        from agentic_investor.universes import get_universe

        pool = get_universe(cfg.universe)
        picks = pick_top_n(pool, top_n=cfg.top_n, as_of=as_of)
        tickers = [p.ticker for p in picks]
    # Always include profile universe_extras (e.g. TLT + GLD for conservative).
    for extra in profile.universe_extras:
        if extra not in tickers:
            tickers.append(extra)

    risk = (
        profile.name
        if profile.name in {"conservative", "moderate", "aggressive"}
        else "moderate"
    )
    req = OrchestratorRequest(
        tickers=[t.upper() for t in tickers], amount=cfg.amount, risk=risk,
    )
    return run_orchestrator(req, profile=profile)


def run_tick(
    cfg: LoopConfig,
    state: LoopState,
    broker: PaperBroker,
    *,
    save_rec: Callable[[Recommendation], int] | None = None,
    price_fetcher: Callable[[str], float] | None = None,
    now: datetime | None = None,
    session=None,  # SessionRecorder | None - optional live logging
) -> TickResult:
    """One iteration of the loop. Callable independently for cron-style ops."""
    now = now or datetime.now(UTC)
    today = now.strftime("%Y-%m-%d")
    tick_at = now.isoformat()

    if save_rec is None:
        from agentic_investor.orchestrator.store import save_recommendation as _save

        save_rec = _save

    if price_fetcher is None:
        from agentic_investor.tools.market import fetch_ohlcv

        def price_fetcher(t: str) -> float:  # type: ignore[misc]
            return float(fetch_ohlcv(t, period="1y")["Close"].iloc[-1])

    # Two-tier: regenerate the recommendation once per day; reuse otherwise.
    regenerated = False
    if state.last_rec_date != today or state.last_rec_id is None:
        logger.info("tick %s: regenerating recommendation for %s", tick_at, today)
        if session:
            session.log("regen_start", {"reason": "new-day-or-first-tick"})
        rec = _generate_recommendation(cfg)
        rec_id = save_rec(rec)
        state.last_rec_id = rec_id
        state.last_rec_date = today
        regenerated = True
        if session:
            session.log("regen_done", {
                "rec_id": rec_id,
                "targets": {p.ticker: p.weight_pct for p in rec.allocation.positions},
                "cash_pct": rec.allocation.cash_pct,
            })
    else:
        from agentic_investor.orchestrator.store import load_recommendation

        rec = load_recommendation(state.last_rec_id)
        if rec is None:
            raise RuntimeError(f"recommendation #{state.last_rec_id} vanished from store")

    acct = broker.get_account()
    positions = broker.get_positions()
    positions_dollars = {p.ticker.upper(): p.market_value for p in positions}
    record_snapshot(acct, positions)

    if not regenerated and not _drift_exceeds_band(
        rec, positions_dollars, acct.equity, cfg.band_abs_pct
    ):
        logger.info(
            "tick %s: no drift beyond %.1fpp - nothing to trade", tick_at, cfg.band_abs_pct
        )
        return TickResult(
            tick_at=tick_at, rec_id=state.last_rec_id,
            regenerated_rec=regenerated, plan_count=0,
            submitted=[], equity=acct.equity,
        )

    tickers = {p.ticker.upper() for p in rec.allocation.positions} | set(positions_dollars)
    prices: dict[str, float] = {}
    for t in tickers:
        try:
            prices[t] = price_fetcher(t)
        except Exception as e:  # noqa: BLE001
            logger.warning("price fetch failed for %s: %s", t, e)

    plans = compute_trade_plan(
        rec, positions_dollars, acct.equity,
        prices=prices, min_trade_dollars=cfg.min_trade_dollars,
    )

    if cfg.dry_run or not plans:
        return TickResult(
            tick_at=tick_at, rec_id=state.last_rec_id,
            regenerated_rec=regenerated, plan_count=len(plans),
            submitted=[], equity=acct.equity,
        )

    if session:
        session.log("trade_plan", {
            "rec_id": state.last_rec_id,
            "trades": [
                {"ticker": p.ticker, "side": p.side, "qty": p.qty, "dollars": p.dollars}
                for p in plans
            ],
        })
    submitted = execute_trade_plan(
        plans, broker, rec_id=state.last_rec_id,
        stop_loss_pct=cfg.stop_loss_pct, take_profit_pct=cfg.take_profit_pct,
        day=today,
    )
    for o in submitted:
        record_order(o, source="loop", rec_id=state.last_rec_id)
        if session:
            session.log("order_submitted", {
                "ticker": o.ticker, "side": o.side, "qty": o.qty,
                "broker_order_id": o.id, "status": o.status,
                "client_order_id": o.client_order_id,
            })
    state.orders_submitted += len(submitted)
    return TickResult(
        tick_at=tick_at, rec_id=state.last_rec_id,
        regenerated_rec=regenerated, plan_count=len(plans),
        submitted=submitted, equity=acct.equity,
    )


def _sleep_until(when_iso: str, *, now: datetime | None = None) -> None:
    """Sleep until the given ISO timestamp (UTC)."""
    now = now or datetime.now(UTC)
    target = datetime.fromisoformat(when_iso.replace("Z", "+00:00"))
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    delta = (target - now).total_seconds()
    if delta > 0:
        # Cap sleep at 15 min per call so a SIGINT lands quickly.
        time.sleep(min(delta, 900))


def run_event_loop(
    cfg: LoopConfig,
    broker: PaperBroker,
    *,
    session=None,
    poll_seconds: float = 5.0,
) -> LoopState:
    """Event-driven variant: subscribe to news, fire LLM on decision moments.

    Runs the interval-based fallback tick every cfg.interval_seconds so we
    still snapshot + rebalance to bands even during quiet news periods.
    """
    import queue as _q
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from agentic_investor.orchestrator.decision_engine import (
        DecisionState,
        build_batch,
        drain_queue,
        ingest,
        should_fire,
    )
    from agentic_investor.tools.news_stream import NewsStreamer

    state = LoopState()
    decision_state = DecisionState()
    event_q: _q.Queue = _q.Queue()

    initial_tickers = list(cfg.tickers) or ["SPY"]  # streamer needs a subscription
    streamer = NewsStreamer(initial_tickers, event_queue=event_q)
    streamer.start()
    if session:
        session.log("streamer_start", {"tickers": initial_tickers})

    last_interval_tick = _dt.now(_UTC)
    try:
        while True:
            clock = broker.get_clock()
            if not clock.is_open:
                if session:
                    session.log("market_closed", {"next_open": clock.next_open})
                if cfg.once:
                    break
                _sleep_until(clock.next_open, now=_dt.now(_UTC))
                continue

            now = _dt.now(_UTC)
            new_events = drain_queue(event_q)
            if new_events and session:
                for e in new_events:
                    session.log("news_received", {
                        "ticker": e.ticker,
                        "headline": e.headline[:120],
                    })
            ingest(decision_state, new_events, now)

            fire, reason = should_fire(decision_state, now)
            interval_due = (
                (now - last_interval_tick).total_seconds() >= cfg.interval_seconds
            )
            if fire or interval_due:
                if fire:
                    batch = build_batch(decision_state, now)
                    if session:
                        session.log("decision_moment", {
                            "reason": reason,
                            **batch.summary(),
                        })
                try:
                    result = run_tick(cfg, state, broker, now=now, session=session)
                    state.ticks_run += 1
                    last_interval_tick = now
                    _log_tick(result)
                except Exception as e:  # noqa: BLE001
                    logger.exception("tick failed: %s", e)
                    if session:
                        session.log("tick_error", {"error": str(e)})
                if cfg.once:
                    break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("event loop interrupted; shutting down")
    finally:
        streamer.stop()
        if session:
            session.log("streamer_stop", {})
    return state


def run_loop(
    cfg: LoopConfig,
    broker: PaperBroker,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    session=None,
) -> LoopState:
    """The continuous outer loop. Handles market hours + ticks + shutdown."""
    state = LoopState()
    logger.info("paper-loop starting: %s", cfg)
    try:
        while True:
            clock = broker.get_clock()
            if not clock.is_open:
                if cfg.once:
                    logger.info("market closed; --once specified - exiting")
                    return state
                logger.info(
                    "market closed; sleeping until next_open=%s", clock.next_open
                )
                _sleep_until(clock.next_open, now=now_fn())
                continue

            try:
                result = run_tick(cfg, state, broker, now=now_fn(), session=session)
                state.ticks_run += 1
                _log_tick(result)
            except Exception as e:  # noqa: BLE001 - one tick failing must not kill the loop
                logger.exception("tick failed: %s", e)
                if session:
                    session.log("tick_error", {"error": str(e)})

            if cfg.once:
                return state
            sleep_fn(cfg.interval_seconds)
    except KeyboardInterrupt:
        logger.info("paper-loop interrupted; exiting cleanly")
        return state


def _log_tick(r: TickResult) -> None:
    tag = " (fresh rec)" if r.regenerated_rec else ""
    logger.info(
        "tick %s%s: equity=$%.2f plans=%d submitted=%d",
        r.tick_at, tag, r.equity, r.plan_count, len(r.submitted),
    )


def format_session_summary(state: LoopState) -> str:
    started = datetime.fromisoformat(state.started_at)
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    dur = datetime.now(UTC) - started
    return (
        f"\nSession summary: {state.ticks_run} ticks, "
        f"{state.orders_submitted} orders submitted, "
        f"ran for {timedelta(seconds=int(dur.total_seconds()))} "
        f"(started {state.started_at})"
    )
