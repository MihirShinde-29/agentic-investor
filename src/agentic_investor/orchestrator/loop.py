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

    # Ticker selection
    tickers: list[str] = field(default_factory=list)
    auto: bool = False
    universe: str = "dow30"
    top_n: int = 8

    # Tick cadence
    interval_seconds: int = 30 * 60
    band_abs_pct: float = 5.0
    # Size-aware bands: small positions get proportionally tighter thresholds.
    # effective_base = min(band_abs_pct, target_pct * band_rel_pct/100).
    # e.g. band_rel_pct=20 -> a 5%-target position gets a 1pp band; a
    # 25%-target position stays at the full 5pp abs band. 0 disables.
    band_rel_pct: float = 20.0
    min_trade_dollars: float = 50.0

    # Triggers
    price_move_threshold_pct: float = 2.0
    force_regen_seconds: int = 30 * 60
    # Disabled: LLM-based technical stance is too noisy to trigger regens.
    # Re-enable once stance derives from deterministic indicators.
    enable_technical_change_trigger: bool = False

    # Opinion-drift filter (LLM variance skip logic)
    opinion_drift_threshold_pct: float = 3.0
    max_avg_drift_pct: float = 5.0
    max_single_delta_pct: float = 15.0

    # 0w: cap on beneficiary tickers promoted from news bodies per regen.
    # Bounds prompt-size growth from wildcard news + body-ticker extraction.
    max_promotions_per_regen: int = 3

    # P1 #3: finBERT sentiment pre-filter. When enabled, scores each news
    # batch locally with finBERT and skips the full LLM regen if aggregate
    # sentiment barely moved from the last fired batch (delta below threshold).
    # Cheap heuristic that shaves LLM calls on quiet news. Opt-in because the
    # model download is ~440MB.
    finbert_prefilter_enabled: bool = False
    finbert_min_delta: float = 0.15

    # Portfolio-level trade cooldown: after any rebalance with >= N trades,
    # require >= Y seconds before another N-trade rebalance can execute.
    # Blocks the whole-portfolio flip-flop pattern (rec #105->#106->#107 and
    # rec #132's diversified->concentrated reversal on 2026-08-31 live
    # session). 30 min matches the observed 15-20 min flip-flop cadence
    # with margin to catch the immediate counter-reaction too.
    big_rebalance_min_trades: int = 5
    big_rebalance_cooldown_seconds: int = 1800
    # Even a "concentration reshape" (n_new <= n_dropped) gets blocked
    # during cooldown if the total notional of the plan exceeds this
    # share of NAV. Catches the CRWD-style $2.5k re-open on a book that
    # already had CRWD - low position-count delta, high dollar delta.
    big_rebalance_max_bypass_notional_pct: float = 15.0

    # Discipline layer (vetoes at the rebalancer boundary)
    cooldown_seconds: int = 900              # 0q temporal cooldown
    adverse_move_threshold_pct: float = 1.0  # 0r don't buy falling knives
    halt_buys_drawdown_pct: float = 5.0      # 0r don't average down on losers
    small_drawdown_hold_pct: float = 3.0     # 0g don't sell into bounces
    force_loss_cut_pct: float = 8.0          # 0g auto-exit deep losers

    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    # Ops toggles
    dry_run: bool = False
    once: bool = False
    force_open: bool = False


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
    """Mutable per-loop runtime state. Persisted to SQLite on every regen."""

    last_rec_id: int | None = None
    last_rec_date: str | None = None  # YYYY-MM-DD; enables cross-day regen
    ticks_run: int = 0
    orders_submitted: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Consumed by run_tick to force a fresh rec that sees the batch context.
    pending_news_context: str | None = None
    # Sticky picker output - prevents portfolio churn from re-scoring the
    # universe on every regen. Reset across days.
    frozen_picker_tickers: list[str] | None = None
    # Reference prices for the price-move trigger + adverse-move veto.
    baseline_prices: dict[str, float] = field(default_factory=dict)
    # Force-regen timing: fires when (now - last_regen_at) > force_regen_seconds.
    last_regen_at: datetime | None = None
    # Last-rec stances, used by the (currently disabled) technical-change trigger.
    last_stances: dict[str, str] = field(default_factory=dict)
    # P1 #3: last batch's finBERT aggregate sentiment (in [-1, 1]). Used by
    # the pre-filter to skip regens when sentiment barely moves.
    last_finbert_score: float | None = None
    # Portfolio-level cooldown: when the last big rebalance (>=N trades)
    # happened, so a subsequent big rebalance is blocked until Y seconds pass.
    last_big_rebalance_at: datetime | None = None
    # {ticker: (side, iso_ts)} - most-recent trade per ticker, used by the
    # temporal-cooldown veto in compute_trade_plan.
    recent_trades: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Fingerprint of the news-batch context that drove the last saved rec.
    # If the next tick's batch fingerprint matches, we reuse the prior rec
    # instead of firing the LLM again - same input, same output, save the
    # ~$0.008 per skip.
    last_batch_fingerprint: str | None = None

    def to_dict(self) -> dict:
        """Serialize for SQLite persistence."""
        return {
            "last_rec_id": self.last_rec_id,
            "last_rec_date": self.last_rec_date,
            "ticks_run": self.ticks_run,
            "orders_submitted": self.orders_submitted,
            "started_at": self.started_at,
            "pending_news_context": self.pending_news_context,
            "frozen_picker_tickers": self.frozen_picker_tickers,
            "baseline_prices": dict(self.baseline_prices),
            "last_regen_at": (
                self.last_regen_at.isoformat() if self.last_regen_at else None
            ),
            "last_stances": dict(self.last_stances),
            "recent_trades": dict(self.recent_trades),
            "last_finbert_score": self.last_finbert_score,
            "last_big_rebalance_at": (
                self.last_big_rebalance_at.isoformat()
                if self.last_big_rebalance_at else None
            ),
            "last_batch_fingerprint": self.last_batch_fingerprint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LoopState:
        """Rebuild from persisted state. Missing keys get defaults."""
        last_regen_at = d.get("last_regen_at")
        if isinstance(last_regen_at, str):
            last_regen_at = datetime.fromisoformat(last_regen_at)
            if last_regen_at.tzinfo is None:
                last_regen_at = last_regen_at.replace(tzinfo=UTC)
        return cls(
            last_rec_id=d.get("last_rec_id"),
            last_rec_date=d.get("last_rec_date"),
            ticks_run=int(d.get("ticks_run") or 0),
            orders_submitted=int(d.get("orders_submitted") or 0),
            started_at=d.get("started_at") or datetime.now(UTC).isoformat(),
            pending_news_context=d.get("pending_news_context"),
            frozen_picker_tickers=d.get("frozen_picker_tickers"),
            baseline_prices=dict(d.get("baseline_prices") or {}),
            last_regen_at=last_regen_at,
            last_stances=dict(d.get("last_stances") or {}),
            recent_trades={
                k: (v[0], v[1]) if isinstance(v, list | tuple) else v
                for k, v in (d.get("recent_trades") or {}).items()
            },
            last_finbert_score=d.get("last_finbert_score"),
            last_big_rebalance_at=(
                datetime.fromisoformat(d["last_big_rebalance_at"])
                if isinstance(d.get("last_big_rebalance_at"), str) else None
            ),
            last_batch_fingerprint=d.get("last_batch_fingerprint"),
        )


def _price_move_trigger(
    baseline_prices: dict[str, float],
    current_prices: dict[str, float],
    threshold_pct: float,
) -> tuple[bool, dict[str, float]]:
    """Any held ticker moved more than threshold_pct from its baseline price?

    Returns (should_fire, per_ticker_move_pct). Uses only tickers present in
    BOTH baseline and current so a freshly-added position doesn't trigger.
    """
    if not baseline_prices:
        return False, {}
    moves: dict[str, float] = {}
    should_fire = False
    for t, base in baseline_prices.items():
        cur = current_prices.get(t)
        if cur is None or base <= 0:
            continue
        pct = (cur / base - 1) * 100
        moves[t] = round(pct, 2)
        if abs(pct) >= threshold_pct:
            should_fire = True
    return should_fire, moves


def _technical_stance_changed(
    prev_stances: dict[str, str],
    current_stances: dict[str, str],
) -> tuple[bool, dict[str, str]]:
    """Any ticker's technical stance flipped since last rec?

    Returns (should_fire, per_ticker_transition) where transitions are
    "prev -> current" strings only for tickers that actually changed.
    """
    changes: dict[str, str] = {}
    for t, cur in current_stances.items():
        prev = prev_stances.get(t)
        if prev and prev != cur:
            changes[t] = f"{prev} -> {cur}"
    return bool(changes), changes


def _opinion_barely_moved(
    new_rec: Recommendation,
    prev_rec: Recommendation | None,
    threshold_pct: float,
) -> tuple[bool, dict[str, float]]:
    """True when EVERY position's target weight changed less than threshold.

    Also returns the per-ticker weight delta INCLUDING cash so callers get
    a complete picture (0k fix - cash was excluded from deltas before, so
    a 5pp cash rotation slipped through the aggregate check).
    Compares by ticker so a new/dropped position always counts as "moved"
    (weight change from 0% to X% is meaningful).
    """
    if prev_rec is None:
        return False, {}
    prev = {p.ticker.upper(): p.weight_pct for p in prev_rec.allocation.positions}
    new = {p.ticker.upper(): p.weight_pct for p in new_rec.allocation.positions}
    tickers = set(prev) | set(new)
    deltas = {t: abs(new.get(t, 0.0) - prev.get(t, 0.0)) for t in tickers}
    # 0k: include cash rotation in the deltas dict so aggregate turnover
    # calculations are accurate. Uses "__cash__" as a sentinel key that
    # doesn't collide with real tickers.
    cash_delta = abs(new_rec.allocation.cash_pct - prev_rec.allocation.cash_pct)
    if cash_delta > 0:
        deltas["__cash__"] = cash_delta
    barely = all(d < threshold_pct for d in deltas.values())
    return barely, deltas


def _filter_should_skip(
    new_rec: Recommendation,
    prev_rec: Recommendation | None,
    *,
    opinion_drift_threshold_pct: float,
    max_avg_drift_pct: float,
    max_single_delta_pct: float,
    news_batch_tickers: set[str] | None = None,
    confidence_by_ticker: dict[str, float] | None = None,
) -> tuple[bool, str, dict[str, float], dict[str, float]]:
    """Filter v2: scale-invariant + context-aware skip decision.

    Skip if:
      (a) all deltas < opinion_drift_threshold (LLM noise floor)
      (b) avg_drift > max_avg_drift_pct (portfolio-wide churn)
      (c) max_single_delta > max_single_delta_pct AND the moving ticker is
          NOT in news_batch_tickers AND its confidence < 0.7
          (dramatic single move without justification)

    Returns (should_skip, skip_reason, deltas, stats). stats has
    "avg_drift", "max_delta", "max_delta_ticker", "n_tickers".
    """
    barely, deltas = _opinion_barely_moved(
        new_rec, prev_rec, opinion_drift_threshold_pct
    )
    if barely:
        return True, "barely-moved", deltas, {}
    # Exclude cash from per-position stats (avg/max) since cash isn't a
    # ticker with signal - it's the residual.
    position_deltas = {k: v for k, v in deltas.items() if k != "__cash__"}
    n = len(position_deltas)
    if n == 0:
        return False, "", deltas, {}
    max_single = max(position_deltas.values())
    max_ticker = max(position_deltas, key=lambda k: position_deltas[k])
    avg_drift = sum(position_deltas.values()) / n
    stats = {
        "avg_drift": round(avg_drift, 2),
        "max_delta": round(max_single, 2),
        "max_delta_ticker": max_ticker,
        "n_tickers": n,
    }
    # Rule (b): scale-invariant avg-drift check
    if avg_drift > max_avg_drift_pct:
        return True, "avg-drift-too-high", deltas, stats
    # Rule (c): context-aware max-single check
    if max_single > max_single_delta_pct:
        news_tickers = news_batch_tickers or set()
        conf = (confidence_by_ticker or {}).get(max_ticker, 0.5)
        justified = max_ticker in news_tickers or conf >= 0.7
        if not justified:
            return True, "max-delta-unjustified", deltas, stats
    return False, "", deltas, stats


def _extract_tickers_from_batch_ctx(batch_ctx: str) -> list[str]:
    """Pull tickers out of a rendered batch context.

    render_batch_context() emits lines like "- [HOT] NVDA  age=..." so a
    simple regex over the second column recovers the ticker set.
    """
    import re

    tickers = re.findall(r"\[(?:HOT|COOKED|STALE)\]\s+([A-Z][A-Z0-9.\-]+)", batch_ctx)
    return list(dict.fromkeys(tickers))


def _effective_band(
    band_abs_pct: float,
    confidence: float | None,
    *,
    target_pct: float = 0.0,
    band_rel_pct: float = 0.0,
) -> float:
    """Effective drift band for one position.

    Two composable layers:
    1. Size-aware base = min(band_abs_pct, target_pct * band_rel_pct/100)
       when band_rel_pct > 0. Small positions get proportionally tighter
       thresholds. A 5% target with band_rel_pct=20 -> 1pp base band.
       Zeroed-out positions (target=0) get the abs band unchanged.
    2. Confidence scaling: factor = 1.5 - confidence  (in [0.5, 1.5])
       - High conviction 1.0 -> 0.5x base (act on smaller drift)
       - Neutral        0.5 -> 1.0x base (unchanged)
       - Low conviction 0.0 -> 1.5x base (anti-churn while uncertain)
       Missing confidence -> factor 1.0 (backward compat).
    """
    base = band_abs_pct
    if band_rel_pct > 0 and target_pct > 0:
        rel_band = target_pct * band_rel_pct / 100.0
        base = min(band_abs_pct, rel_band)
    if confidence is None:
        return base
    c = max(0.0, min(1.0, confidence))
    return base * (1.5 - c)


def _drift_exceeds_band(
    rec: Recommendation,
    positions_dollars: dict[str, float],
    total_equity: float,
    band_abs_pct: float,
    band_rel_pct: float = 0.0,
) -> bool:
    """Any position drifted more than its (size- and confidence-adjusted) band?"""
    if total_equity <= 0:
        return True
    target = {p.ticker.upper(): p.weight_pct for p in rec.allocation.positions}
    confidence = {p.ticker.upper(): p.confidence for p in rec.allocation.positions}
    current_pct = {
        t: (v / total_equity) * 100.0 for t, v in positions_dollars.items()
    }
    tickers = set(target) | set(current_pct)
    for t in tickers:
        target_pct = target.get(t, 0.0)
        drift = abs(target_pct - current_pct.get(t, 0.0))
        band = _effective_band(
            band_abs_pct,
            confidence.get(t),
            target_pct=target_pct,
            band_rel_pct=band_rel_pct,
        )
        if drift >= band:
            return True
    return False


def _generate_recommendation(
    cfg: LoopConfig,
    *,
    as_of: str | None = None,
    news_batch_context: str | None = None,
    pre_picked_tickers: list[str] | None = None,
    previous_rec: Recommendation | None = None,
    extra_tickers: list[str] | None = None,
) -> tuple[Recommendation, list[str]]:
    """Run the orchestrator once for today's decision. Uses the M6 profile.

    news_batch_context: optional rendered HOT/COOKED news events from the
    event-driven loop, forwarded to the allocator prompt.

    pre_picked_tickers: if provided, skip the picker and use these tickers
    directly. Enables sticky picker output across regens - the loop caches
    the first regen's picks so news-triggered regens don't churn the
    portfolio by re-running the picker (mega_tech scores shift by the minute).

    extra_tickers: promoted beneficiary candidates from news bodies (0w).
    Appended after the picker/frozen set so they surface to the allocator
    with signals; the LLM decides whether to weight them.

    Returns (rec, tickers_used) so callers can cache the ticker set.
    """
    from agentic_investor.orchestrator.graph import run_orchestrator
    from agentic_investor.orchestrator.strategy import load_profile

    profile = load_profile(cfg.profile_name)
    tickers = list(cfg.tickers)
    if pre_picked_tickers is not None:
        tickers = list(pre_picked_tickers)
    elif cfg.auto:
        from agentic_investor.orchestrator.picker import pick_top_n
        from agentic_investor.universes import get_universe

        pool = get_universe(cfg.universe)
        picks = pick_top_n(pool, top_n=cfg.top_n, as_of=as_of)
        tickers = [p.ticker for p in picks]
    # Always include profile universe_extras (e.g. TLT + GLD for conservative).
    for extra in profile.universe_extras:
        if extra not in tickers:
            tickers.append(extra)
    # 0w: news-body beneficiary promotions. Append after core universe so the
    # allocator sees them as extras, not primary picks.
    if extra_tickers:
        for t in extra_tickers:
            up = t.upper()
            if up not in [x.upper() for x in tickers]:
                tickers.append(up)

    risk = (
        profile.name
        if profile.name in {"conservative", "moderate", "aggressive"}
        else "moderate"
    )
    final_tickers = [t.upper() for t in tickers]
    req = OrchestratorRequest(
        tickers=final_tickers, amount=cfg.amount, risk=risk,
    )
    prev_alloc = previous_rec.allocation if previous_rec is not None else None
    rec = run_orchestrator(
        req, profile=profile,
        news_batch_context=news_batch_context,
        previous_allocation=prev_alloc,
    )
    return rec, final_tickers


def run_tick(
    cfg: LoopConfig,
    state: LoopState,
    broker: PaperBroker,
    *,
    save_rec: Callable[[Recommendation], int] | None = None,
    price_fetcher: Callable[[str], float] | None = None,
    now: datetime | None = None,
    session=None,  # SessionRecorder | None - optional live logging
    trigger: str | None = None,
) -> TickResult:
    """One iteration of the loop. Callable independently for cron-style ops."""
    from agentic_investor.llm.client import get_call_stats

    now = now or datetime.now(UTC)
    today = now.strftime("%Y-%m-%d")
    tick_at = now.isoformat()
    stats_before = get_call_stats()

    if save_rec is None:
        from agentic_investor.orchestrator.store import save_recommendation as _save

        save_rec = _save

    if price_fetcher is None:
        from agentic_investor.tools.paper_broker import get_latest_price

        def price_fetcher(t: str) -> float:  # type: ignore[misc]
            return get_latest_price(t)

    def _log_tick_cost() -> None:
        if not session:
            return
        after = get_call_stats()
        cost = after.estimated_cost_usd - stats_before.estimated_cost_usd
        prompt_delta = after.prompt_tokens - stats_before.prompt_tokens
        cached_delta = after.cached_tokens - stats_before.cached_tokens
        session.log("tick_cost", {
            "tick_at": tick_at,
            "llm_calls": after.n_calls - stats_before.n_calls,
            "prompt_tokens": prompt_delta,
            "cached_tokens": cached_delta,
            "cache_hit_pct": (
                round(cached_delta / prompt_delta * 100, 1)
                if prompt_delta else 0
            ),
            "completion_tokens": after.completion_tokens - stats_before.completion_tokens,
            "cost_usd": f"${cost:.4f}",
        })

    # Two-tier: regenerate the recommendation once per day OR when the event
    # loop pushed news batch context onto state; reuse otherwise.
    regenerated = False
    batch_ctx = state.pending_news_context
    if (
        state.last_rec_date != today
        or state.last_rec_id is None
        or batch_ctx
    ):
        # Prefer the explicit trigger from the event loop; fall back to
        # inference from batch_ctx / date only for direct-call test paths.
        reason = trigger or (
            "news-batch" if batch_ctx
            else ("new-day" if state.last_rec_date != today else "first-tick")
        )
        logger.info("tick %s: regenerating recommendation (%s)", tick_at, reason)
        if session:
            session.log("regen_start", {"reason": reason, "has_news_batch": bool(batch_ctx)})
        # Frozen picker: cached after first regen of the day so news-triggered
        # regens don't reshuffle the ticker set.
        is_new_day = state.last_rec_date != today
        pre_picked = None if is_new_day else state.frozen_picker_tickers

        # 0w: news-body beneficiary promotion. Extract every ticker named in
        # the batch (both Alpaca-tagged and body-extracted, per 0p fan-out),
        # subtract anything already in the pick set, cap at N to bound prompt
        # size. Filtered to the sp500 large-cap whitelist so micro-cap punts
        # from single news headlines (BRVE, SLI, GPRO, WBUY, ATTO, RCBC-style)
        # don't reach the allocator.
        promoted: list[str] = []
        if batch_ctx:
            from agentic_investor.universes import is_actionable_ticker

            already = set()
            if pre_picked:
                already.update(x.upper() for x in pre_picked)
            if cfg.tickers:
                already.update(x.upper() for x in cfg.tickers)
            for t in _extract_tickers_from_batch_ctx(batch_ctx):
                up = t.upper()
                if up in already or up in {p.upper() for p in promoted}:
                    continue
                if not is_actionable_ticker(up):
                    continue  # micro-cap filter (0w hardening)
                promoted.append(up)
                if len(promoted) >= cfg.max_promotions_per_regen:
                    break

        # Deterministic pre-check: if the news batch fingerprint matches the
        # one that drove the last saved rec, the LLM would see identical
        # input and produce the same output. Reuse the prior rec, skip the
        # LLM call, and fall through to drift-vs-book below.
        import hashlib as _hashlib

        batch_fp = (
            _hashlib.sha1(batch_ctx.encode("utf-8")).hexdigest()[:16]
            if batch_ctx else None
        )
        pre_check_skipped = (
            batch_fp is not None
            and batch_fp == state.last_batch_fingerprint
            and not is_new_day
            and state.last_rec_id is not None
        )
        if pre_check_skipped:
            logger.info(
                "tick %s: pre-check skip (batch unchanged since rec #%d)",
                tick_at, state.last_rec_id,
            )
            if session:
                session.log("pre_check_skip", {
                    "rec_id": state.last_rec_id,
                    "batch_fingerprint": batch_fp,
                })
            from agentic_investor.orchestrator.store import (
                load_recommendation as _load_prev,
            )
            rec = _load_prev(state.last_rec_id)
            regenerated = False
            tickers_used = state.frozen_picker_tickers or []
        else:
            # Delta-form prompt anchor (non-first-day only).
            prev_rec_for_prompt = None
            if not is_new_day and state.last_rec_id is not None:
                from agentic_investor.orchestrator.store import (
                    load_recommendation as _load_for_anchor,
                )
                prev_rec_for_prompt = _load_for_anchor(state.last_rec_id)
            rec, tickers_used = _generate_recommendation(
                cfg, news_batch_context=batch_ctx, pre_picked_tickers=pre_picked,
                previous_rec=prev_rec_for_prompt,
                extra_tickers=promoted or None,
            )
            state.last_batch_fingerprint = batch_fp

        # Opinion-drift filter: compare new rec to previous; skip on LLM
        # noise (barely-moved) or over-swing (avg-drift / max-delta).
        if not is_new_day and state.last_rec_id is not None:
            from agentic_investor.orchestrator.store import (
                load_recommendation as _load,
            )
            prev_rec = _load(state.last_rec_id)
            batch_tickers_for_filter = set(
                _extract_tickers_from_batch_ctx(batch_ctx)
            ) if batch_ctx else set()
            confidence_lookup = {
                p.ticker.upper(): (p.confidence or 0.5)
                for p in rec.allocation.positions
            }
            should_skip, skip_reason, deltas, stats = _filter_should_skip(
                rec, prev_rec,
                opinion_drift_threshold_pct=cfg.opinion_drift_threshold_pct,
                max_avg_drift_pct=cfg.max_avg_drift_pct,
                max_single_delta_pct=cfg.max_single_delta_pct,
                news_batch_tickers=batch_tickers_for_filter,
                confidence_by_ticker=confidence_lookup,
            )
            if should_skip:
                logger.info(
                    "tick %s: filter skip (%s) - avg %.2fpp, max %.2fpp on %s",
                    tick_at, skip_reason,
                    stats.get("avg_drift", 0),
                    stats.get("max_delta", 0),
                    stats.get("max_delta_ticker", "?"),
                )
                state.pending_news_context = None
                # last_regen_at MUST update on skip too - otherwise force-regen
                # returns True on every poll and re-fires the LLM in a loop.
                state.last_regen_at = now
                # Reset baseline prices on skip too. Without this, the
                # price-move trigger re-fires on every poll because baseline
                # never updates (stuck at pre-restart values), producing an
                # endless regen -> skip cascade at ~5 LLM calls per 14s.
                try:
                    for p in rec.allocation.positions:
                        state.baseline_prices[p.ticker.upper()] = float(
                            price_fetcher(p.ticker.upper())
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning("baseline reset on skip failed: %s", e)
                try:
                    from agentic_investor.tools.paper_store import save_loop_state
                    save_loop_state(state.to_dict())
                except Exception as e:  # noqa: BLE001
                    logger.warning("save_loop_state failed on skip: %s", e)
                # Attribution: record the would-be allocation for later
                # counterfactual analysis (false-positive rate measurement).
                try:
                    from agentic_investor.tools.paper_store import record_filter_skip
                    skip_acct = broker.get_account()
                    skip_positions = broker.get_positions()
                    record_filter_skip(
                        skip_reason=skip_reason,
                        trigger_reason=reason,
                        would_be_allocation=rec.allocation.model_dump(),
                        actual_positions=[
                            {"ticker": p.ticker, "qty": p.qty,
                             "market_value": p.market_value}
                            for p in skip_positions
                        ],
                        equity_at_skip=skip_acct.equity,
                        stats=stats,
                        deltas={t: round(d, 2) for t, d in deltas.items()},
                        prev_rec_id=state.last_rec_id,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("record_filter_skip failed: %s", e)
                if session:
                    session.log("opinion_drift_skip", {
                        "reason": reason,
                        "skip_reason": skip_reason,
                        "avg_drift_pp": stats.get("avg_drift", 0),
                        "max_delta_pp": stats.get("max_delta", 0),
                        "max_delta_ticker": stats.get("max_delta_ticker", "?"),
                        "n_tickers": stats.get("n_tickers", 0),
                        "threshold_pp": cfg.opinion_drift_threshold_pct,
                        "max_avg_drift_pp": cfg.max_avg_drift_pct,
                        "max_single_delta_pp": cfg.max_single_delta_pct,
                        "deltas": {t: round(d, 2) for t, d in deltas.items()},
                    })
                # Stable opinion doesn't mean the book matches the target -
                # earlier orders may have failed or been cooldown-blocked.
                # Fall through to drift-vs-book against the prior rec.
                # avg-drift-too-high / max-delta-unjustified are noise skips
                # and still bail here.
                if skip_reason == "barely-moved" and prev_rec is not None:
                    rec = prev_rec
                    regenerated = False
                else:
                    _log_tick_cost()
                    return TickResult(
                        tick_at=tick_at, rec_id=state.last_rec_id,
                        regenerated_rec=False, plan_count=0,
                        submitted=[], equity=broker.get_account().equity,
                    )
            else:
                rec_id = save_rec(rec)
                state.last_rec_id = rec_id
                state.last_rec_date = today
                state.frozen_picker_tickers = tickers_used
                state.pending_news_context = None
                state.last_regen_at = now
                state.baseline_prices = {}
                for p in rec.allocation.positions:
                    try:
                        state.baseline_prices[p.ticker.upper()] = float(
                            price_fetcher(p.ticker.upper())
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("baseline price capture failed for %s: %s",
                                       p.ticker, e)
                state.last_stances = {
                    s.ticker.upper(): s.stance for s in rec.technical_signals
                }
                regenerated = True
                try:
                    from agentic_investor.tools.paper_store import save_loop_state
                    save_loop_state(state.to_dict())
                except Exception as e:  # noqa: BLE001
                    logger.warning("save_loop_state failed: %s", e)
                if session:
                    session.log("regen_done", {
                        "rec_id": rec_id,
                        "targets": {p.ticker: p.weight_pct for p in rec.allocation.positions},
                        "cash_pct": rec.allocation.cash_pct,
                        "trigger": reason,
                    })
        else:
            # First-ever regen or new-day reset: no prior rec to compare
            # against, so the drift filter can't fire. Save straight through.
            rec_id = save_rec(rec)
            state.last_rec_id = rec_id
            state.last_rec_date = today
            state.frozen_picker_tickers = tickers_used
            state.pending_news_context = None
            state.last_regen_at = now
            state.baseline_prices = {}
            for p in rec.allocation.positions:
                try:
                    state.baseline_prices[p.ticker.upper()] = float(
                        price_fetcher(p.ticker.upper())
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("baseline price capture failed for %s: %s",
                                   p.ticker, e)
            state.last_stances = {
                s.ticker.upper(): s.stance for s in rec.technical_signals
            }
            regenerated = True
            try:
                from agentic_investor.tools.paper_store import save_loop_state
                save_loop_state(state.to_dict())
            except Exception as e:  # noqa: BLE001
                logger.warning("save_loop_state failed: %s", e)
            if session:
                session.log("regen_done", {
                    "rec_id": rec_id,
                    "targets": {p.ticker: p.weight_pct for p in rec.allocation.positions},
                    "cash_pct": rec.allocation.cash_pct,
                    "trigger": reason,
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

    # Cap the invested base at cfg.amount so drift/rebalance math matches
    # what the user asked to trade with (not the full account equity).
    allocation_base_for_drift = (
        min(cfg.amount, acct.equity) if cfg.amount > 0 else acct.equity
    )

    if not regenerated and not _drift_exceeds_band(
        rec, positions_dollars, allocation_base_for_drift,
        cfg.band_abs_pct, cfg.band_rel_pct,
    ):
        logger.info(
            "tick %s: no drift beyond %.1fpp - nothing to trade", tick_at, cfg.band_abs_pct
        )
        _log_tick_cost()
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

    # Deserialize state.recent_trades timestamps + prep news_batch tickers
    # for cooldown-bypass logic in compute_trade_plan.
    recent_typed: dict[str, tuple[str, datetime]] = {}
    for tk, entry in state.recent_trades.items():
        try:
            side_v, ts_v = entry
            ts_dt = datetime.fromisoformat(ts_v) if isinstance(ts_v, str) else ts_v
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=UTC)
            recent_typed[tk] = (side_v, ts_dt)
        except Exception:  # noqa: BLE001
            continue
    batch_tickers_for_cooldown = set(
        _extract_tickers_from_batch_ctx(batch_ctx)
    ) if batch_ctx else set()

    # Discipline layer inputs: recent price moves + avg entry prices per ticker.
    ticker_recent_moves: dict[str, float] = {}
    avg_entry_prices: dict[str, float] = {p.ticker.upper(): float(p.avg_entry_price)
                                          for p in positions if p.avg_entry_price}
    for t in tickers:
        baseline = state.baseline_prices.get(t)
        cur = prices.get(t)
        if baseline and cur and baseline > 0:
            ticker_recent_moves[t] = (cur / baseline - 1) * 100
    # --amount caps invested capital when the account is bigger than what
    # the user asked to trade with. If cfg.amount >= acct.equity we're
    # unconstrained (allocate against full account); otherwise cap here.
    allocation_base = min(cfg.amount, acct.equity) if cfg.amount > 0 else acct.equity
    plans = compute_trade_plan(
        rec, positions_dollars, allocation_base,
        prices=prices, min_trade_dollars=cfg.min_trade_dollars,
        recent_trades=recent_typed,
        cooldown_seconds=getattr(cfg, "cooldown_seconds", 900),
        now=now,
        news_batch_tickers=batch_tickers_for_cooldown,
        ticker_recent_moves=ticker_recent_moves,
        adverse_move_threshold_pct=cfg.adverse_move_threshold_pct,
        avg_entry_prices=avg_entry_prices,
        halt_buys_drawdown_pct=cfg.halt_buys_drawdown_pct,
        small_drawdown_hold_pct=cfg.small_drawdown_hold_pct,
        force_loss_cut_pct=cfg.force_loss_cut_pct,
    )

    if cfg.dry_run or not plans:
        _log_tick_cost()
        return TickResult(
            tick_at=tick_at, rec_id=state.last_rec_id,
            regenerated_rec=regenerated, plan_count=len(plans),
            submitted=[], equity=acct.equity,
        )

    # Cooldown on big rebalances. Direction-aware: a plan that drops as many
    # or more names than it opens (with <= 2 new) is treated as the LLM
    # cleaning up an over-diversified book and passes; adding more names
    # than it drops is the whipsaw pattern we're trying to block.
    if (
        len(plans) >= cfg.big_rebalance_min_trades
        and state.last_big_rebalance_at is not None
    ):
        secs = (now - state.last_big_rebalance_at).total_seconds()
        if secs < cfg.big_rebalance_cooldown_seconds:
            target_tickers = {p.ticker.upper() for p in rec.allocation.positions}
            current_tickers = set(positions_dollars.keys())
            n_new = len(target_tickers - current_tickers)
            n_dropped = len(current_tickers - target_tickers)
            plan_notional = sum(float(p.dollars or 0) for p in plans)
            notional_pct = (
                plan_notional / acct.equity * 100.0 if acct.equity else 0.0
            )
            is_small_reshape = (
                n_new <= n_dropped
                and n_new <= 2
                and notional_pct <= cfg.big_rebalance_max_bypass_notional_pct
            )
            if is_small_reshape:
                logger.info(
                    "tick %s: cooldown bypass (concentration reshape) "
                    "new=%d dropped=%d plan=%d notional=%.1f%%",
                    tick_at, n_new, n_dropped, len(plans), notional_pct,
                )
                if session:
                    session.log("big_rebalance_cooldown_bypass", {
                        "rec_id": state.last_rec_id,
                        "plan_count": len(plans),
                        "n_new_positions": n_new,
                        "n_dropped_positions": n_dropped,
                        "notional_pct": round(notional_pct, 2),
                        "seconds_since_last": secs,
                    })
            else:
                logger.info(
                    "tick %s: big-rebalance cooldown blocking %d trades "
                    "(last was %.0fs ago, need %ds; new=%d dropped=%d "
                    "notional=%.1f%%)",
                    tick_at, len(plans), secs,
                    cfg.big_rebalance_cooldown_seconds,
                    n_new, n_dropped, notional_pct,
                )
                if session:
                    session.log("big_rebalance_cooldown_skip", {
                        "rec_id": state.last_rec_id,
                        "plan_count": len(plans),
                        "seconds_since_last": secs,
                        "cooldown_seconds": cfg.big_rebalance_cooldown_seconds,
                        "n_new_positions": n_new,
                        "n_dropped_positions": n_dropped,
                        "notional_pct": round(notional_pct, 2),
                    })
                _log_tick_cost()
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
        # Record for temporal cooldown lookup on next tick.
        state.recent_trades[o.ticker.upper()] = (o.side, now.isoformat())
        if session:
            session.log("order_submitted", {
                "ticker": o.ticker, "side": o.side, "qty": o.qty,
                "broker_order_id": o.id, "status": o.status,
                "client_order_id": o.client_order_id,
            })
    state.orders_submitted += len(submitted)
    # Stamp the portfolio-level cooldown when this counts as a big rebalance.
    if len(submitted) >= cfg.big_rebalance_min_trades:
        state.last_big_rebalance_at = now
    # Persist post-execution state so cooldown timing + recent_trades survive
    # restart. Without this, `last_big_rebalance_at` is only saved via the
    # earlier post-regen save which runs BEFORE trade execution, so a restart
    # loses the most recent big-rebalance timestamp.
    try:
        from agentic_investor.tools.paper_store import save_loop_state as _sls
        _sls(state.to_dict())
    except Exception as e:  # noqa: BLE001
        logger.warning("save_loop_state post-execute failed: %s", e)
    _log_tick_cost()
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
        default_reaction_price_fetcher,
        drain_queue,
        ingest,
        render_batch_context,
        should_fire,
    )
    from agentic_investor.tools.news_stream import NewsStreamer
    from agentic_investor.tools.paper_store import load_loop_state

    # Restart survival: if a prior state was persisted for this account,
    # rehydrate. If last_rec_date is TODAY we skip the first-tick regen
    # (state is fresh from earlier session). Different day = new-day regen.
    persisted = load_loop_state()
    state = LoopState.from_dict(persisted) if persisted else LoopState()
    if persisted:
        logger.info(
            "restored loop state: last_rec_id=%s last_rec_date=%s ticks=%d",
            state.last_rec_id, state.last_rec_date, state.ticks_run,
        )
        if session:
            session.log("state_restored", {
                "last_rec_id": state.last_rec_id,
                "last_rec_date": state.last_rec_date,
                "ticks_run": state.ticks_run,
            })
        # Invalidate stale baselines on restart: without this, the first tick
        # after a restart compares CURRENT market prices against baselines
        # from the last session (often minutes-to-hours old), and the
        # price-move trigger fires spuriously against a stale reference.
        # Wipe them so the first successful regen re-captures fresh baselines.
        if state.baseline_prices:
            n_wiped = len(state.baseline_prices)
            state.baseline_prices = {}
            logger.info(
                "invalidated %d stale baseline prices on restart", n_wiped,
            )
            if session:
                session.log("baselines_invalidated", {"count": n_wiped})
    decision_state = DecisionState()
    event_q: _q.Queue = _q.Queue()

    # Streamer subscription: wildcard "*" in auto mode gets ALL news, then
    # filtered downstream by ticker-mention extraction against the universe.
    # Explicit tickers mode subscribes only to those (narrow, deterministic).
    if cfg.tickers:
        initial_tickers = list(cfg.tickers)
    elif cfg.auto:
        initial_tickers = ["*"]
    else:
        initial_tickers = ["SPY"]
    streamer = NewsStreamer(initial_tickers, event_queue=event_q)
    streamer.start()
    if session:
        session.log("streamer_start", {"tickers": initial_tickers})

    # Backdate the interval tick so the first loop iteration always fires a
    # tick (interval_due is immediately True). Otherwise --once + no news
    # would sit idle for `interval_seconds` before exiting.
    from datetime import timedelta as _td
    last_interval_tick = _dt.now(_UTC) - _td(seconds=cfg.interval_seconds)
    last_reconcile_at: _dt | None = None
    try:
        while True:
            try:
                clock = broker.get_clock()
            except Exception as e:  # noqa: BLE001 - network hiccup mustn't crash loop
                logger.warning("get_clock failed (%s); retrying in 15s", e)
                if session:
                    session.log("clock_error", {"error": str(e)})
                time.sleep(15)
                continue
            if not clock.is_open and not cfg.force_open:
                if session:
                    session.log("market_closed", {"next_open": clock.next_open})
                if cfg.once:
                    break
                _sleep_until(clock.next_open, now=_dt.now(_UTC))
                continue

            now = _dt.now(_UTC)
            # Poll broker for order fill updates; keeps paper_orders mirror
            # in sync with real Alpaca state. Throttled to every 60s so the
            # dashboard event feed doesn't fill with reconcile pings when
            # nothing actually changed.
            if (
                last_reconcile_at is None
                or (now - last_reconcile_at).total_seconds() >= 60
            ):
                try:
                    from agentic_investor.tools.paper_store import reconcile_orders
                    n_updated = reconcile_orders(broker)
                    if n_updated and session:
                        session.log("orders_reconciled", {"updated": n_updated})
                except Exception as e:  # noqa: BLE001
                    logger.warning("reconcile_orders failed: %s", e)
                last_reconcile_at = now
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

            # Extra triggers layered on top of news + interval:
            # (a) price-move: any held ticker moved > threshold from baseline
            # (b) force-regen: last regen was more than force_regen_seconds ago
            # (c) technical-stance: any held ticker's technical stance flipped
            price_ctx = ""
            price_fire = False
            if not fire and state.baseline_prices:
                try:
                    from agentic_investor.tools.paper_broker import get_latest_price
                    cur_prices = {}
                    for t in state.baseline_prices:
                        try:
                            cur_prices[t] = get_latest_price(t)
                        except Exception:  # noqa: BLE001
                            pass
                    hit, moves = _price_move_trigger(
                        state.baseline_prices, cur_prices,
                        cfg.price_move_threshold_pct,
                    )
                    if hit:
                        moved = []
                        for t, m in moves.items():
                            if abs(m) >= cfg.price_move_threshold_pct:
                                base = state.baseline_prices[t]
                                cur = cur_prices[t]
                                moved.append(
                                    f"{t}: {m:+.2f}% (${base:.2f} -> ${cur:.2f})"
                                )
                        price_ctx = "Price-move alert since last decision:\n" + "\n".join(
                            f"- {line}" for line in moved
                        )
                        price_fire = True
                        reason = "price-move"
                        if session:
                            session.log("price_move_trigger", {"moves": moves})
                except Exception as e:  # noqa: BLE001
                    logger.warning("price-move check failed: %s", e)

            force_fire = False
            if not fire and not price_fire and cfg.force_regen_seconds > 0:
                if state.last_regen_at is None:
                    force_fire = False  # first regen handled by is_new_day path
                else:
                    age = (now - state.last_regen_at).total_seconds()
                    if age >= cfg.force_regen_seconds:
                        force_fire = True
                        reason = "force-regen"
                        if session:
                            session.log("force_regen_trigger", {
                                "seconds_since_last": int(age),
                            })

            stance_fire = False
            stance_ctx = ""
            if (not fire and not price_fire and not force_fire
                    and cfg.enable_technical_change_trigger
                    and state.last_stances):
                try:
                    from agentic_investor.agents.technical import analyze_technical
                    from agentic_investor.tools.market import get_market_snapshot
                    cur_stances = {}
                    for t in state.last_stances:
                        try:
                            snap = get_market_snapshot(t)
                            sig = analyze_technical(snap)
                            cur_stances[t] = sig.stance
                        except Exception:  # noqa: BLE001
                            pass
                    hit, changes = _technical_stance_changed(
                        state.last_stances, cur_stances,
                    )
                    if hit:
                        stance_ctx = "Technical stance change since last decision:\n" + "\n".join(
                            f"- {t}: {c}" for t, c in changes.items()
                        )
                        stance_fire = True
                        reason = "technical-change"
                        if session:
                            session.log("technical_change_trigger", {"changes": changes})
                except Exception as e:  # noqa: BLE001
                    logger.warning("technical-change check failed: %s", e)

            if fire or price_fire or force_fire or stance_fire or interval_due:
                if fire:
                    batch = build_batch(
                        decision_state, now,
                        reaction_price_fetcher=default_reaction_price_fetcher,
                    )
                    # P1 #3: finBERT pre-filter. Score the batch locally and
                    # skip the LLM regen if aggregate sentiment barely moved
                    # from the last fired batch. Fully opt-in via config.
                    if cfg.finbert_prefilter_enabled:
                        try:
                            from agentic_investor.orchestrator.finbert_prefilter import (
                                score_events,
                            )
                            all_events = [
                                *(t.event for t in batch.hot),
                                *(t.event for t in batch.cooked),
                            ]
                            sent = score_events(all_events)
                            if sent is not None:
                                prev = state.last_finbert_score
                                if prev is not None:
                                    delta = abs(sent.score - prev)
                                    if delta < cfg.finbert_min_delta:
                                        logger.info(
                                            "finBERT pre-filter skip: sentiment "
                                            "delta %.3f < %.3f (score=%.3f, "
                                            "%d headlines)",
                                            delta, cfg.finbert_min_delta,
                                            sent.score, sent.n_headlines,
                                        )
                                        if session:
                                            session.log("finbert_skip", {
                                                "reason": reason,
                                                "score": sent.score,
                                                "prev_score": prev,
                                                "delta": delta,
                                                "threshold": cfg.finbert_min_delta,
                                                "n_headlines": sent.n_headlines,
                                            })
                                        # Consume the batch (already drained
                                        # via build_batch), skip firing.
                                        state.pending_news_context = None
                                        time.sleep(poll_seconds)
                                        continue
                                state.last_finbert_score = sent.score
                        except Exception as e:  # noqa: BLE001
                            logger.debug("finBERT prefilter error: %s", e)
                    ctx = render_batch_context(batch)
                    state.pending_news_context = ctx or None
                    if session:
                        cooked_with_reaction = sum(
                            1 for c in batch.cooked if c.reaction_pct is not None
                        )
                        session.log("decision_moment", {
                            "reason": reason,
                            **batch.summary(),
                            "context_chars": len(ctx),
                            "cooked_with_reaction": cooked_with_reaction,
                        })
                elif price_fire or stance_fire:
                    # Non-news trigger: forge a minimal batch_context so the
                    # allocator prompt sees WHY it's being re-invoked.
                    state.pending_news_context = "\n\n".join(
                        c for c in (price_ctx, stance_ctx) if c
                    ) or None
                    if session:
                        session.log("decision_moment", {
                            "reason": reason,
                            "context_chars": len(state.pending_news_context or ""),
                        })
                elif force_fire:
                    state.pending_news_context = (
                        "Force-regen: no news / price / technical trigger in the "
                        f"last {cfg.force_regen_seconds // 60} minutes; re-evaluate."
                    )
                    if session:
                        session.log("decision_moment", {"reason": reason})
                elif interval_due and not reason:
                    reason = "interval"
                try:
                    result = run_tick(
                        cfg, state, broker, now=now, session=session,
                        trigger=reason,
                    )
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
    from agentic_investor.tools.paper_store import load_loop_state
    persisted = load_loop_state()
    state = LoopState.from_dict(persisted) if persisted else LoopState()
    if persisted and session:
        session.log("state_restored", {
            "last_rec_id": state.last_rec_id,
            "last_rec_date": state.last_rec_date,
        })
    logger.info("paper-loop starting: %s", cfg)
    try:
        while True:
            try:
                clock = broker.get_clock()
            except Exception as e:  # noqa: BLE001 - network hiccup mustn't crash loop
                logger.warning("get_clock failed (%s); retrying in 15s", e)
                if session:
                    session.log("clock_error", {"error": str(e)})
                time.sleep(15)
                continue
            if not clock.is_open and not cfg.force_open:
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
