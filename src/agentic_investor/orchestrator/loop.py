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
    # Category-specific trade-size floors. Split by direction/context so a
    # ladder of small adds can't slip through under a single loose threshold.
    # Closes always execute regardless of size. Legacy min_trade_dollars was
    # removed after these superseded it.
    min_open_dollars: float = 200.0
    min_add_dollars: float = 500.0
    min_trim_dollars: float = 200.0

    # Triggers
    price_move_threshold_pct: float = 2.0
    force_regen_seconds: int = 30 * 60
    # Disabled: LLM-based technical stance is too noisy to trigger regens.
    # Re-enable once stance derives from deterministic indicators.
    enable_technical_change_trigger: bool = False

    # Opinion-drift filter (LLM variance skip logic). Threshold raised from
    # 3pp to 5pp after 2026-09-03 showed the tighter setting was blocking
    # legitimate small rebalances the LLM was making on real news. avg and
    # single-delta caps keep the outer bounds tight so hallucinated blowups
    # (TEAM +30pp on unrelated batches) still get caught.
    opinion_drift_threshold_pct: float = 5.0
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
    # Per-headline fast-path: when any single incoming headline scores above
    # this abs-threshold, force-close the batch window and fire the regen
    # immediately. Avoids the dilution failure where 1 strong signal averages
    # with N neutral notes in the aggregate. Only active when the prefilter
    # itself is enabled. 0.6 corresponds to a highly-confident directional
    # sentiment (finBERT rarely emits >0.6 on ambient analyst chatter).
    finbert_immediate_threshold: float = 0.6
    # Post-regen cooldown before hot-signal fast-path can fire another regen.
    # Without this, hot news arriving 5s after a regen just finished forces
    # a fresh LLM call while the pending buffer would have picked it up on
    # the next natural batch window. Coalesces rapid re-fires observed on
    # 2026-09-04 (avg ~1 regen per 90s during heavy news periods).
    finbert_hot_signal_cooldown_seconds: int = 60
    # Emit a `cost_alert` session event when the last-hour rolling LLM
    # spend crosses this threshold. Steady-state Sept-4 was ~$0.20/hr;
    # anything above $0.50/hr is either a promotion storm or a bug.
    # Cheap heuristic so runaway churn is auto-flagged instead of
    # requiring us to eyeball the running cost display.
    cost_alert_per_hour_usd: float = 0.50

    # CLI override for profile.max_positions. None = use profile default (12
    # for moderate). Exposed as --max-positions so we can A/B without editing
    # a TOML.
    max_positions_override: int | None = None

    # Discipline layer (vetoes at the rebalancer boundary)
    cooldown_seconds: int = 1500             # 25min per-ticker flip lockout
    # News-convergence bypass: waive the per-ticker cooldown only when at
    # least N distinct news URLs on the same ticker landed in the window
    # below. Blocks single-note re-flips; lets convergent multi-broker
    # stories through.
    news_convergence_min_sources: int = 2
    news_convergence_window_sec: int = 900   # 15min rolling window
    force_loss_cut_pct: float = 8.0          # capital-preservation circuit breaker
    # Concentration ceiling on any BUY execution. Tighter than the profile's
    # max_single_pct proposal cap - this stops the mechanical rebalancer from
    # GROWING a book position above the ceiling even when the LLM's target
    # sits under it. Blocks the cumulative-ladder pattern where each
    # individual add clears the size floor but the sum runs the position
    # to 40%+ of NAV.
    max_add_concentration_pct: float = 25.0

    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    # Trigger materiality: news batches whose tickers don't intersect with
    # held + picker-frozen + recent-exits are dropped BEFORE they fire a
    # regen. Cuts the "barely-moved skip on non-book news" waste.
    materiality_filter_enabled: bool = True

    # Correlation-drift trigger: after each rec, snapshot the pairwise
    # correlation matrix on held tickers. On each tick, recompute and fire a
    # regen if any pair shifted by more than this threshold in absolute
    # units. 0.15 means "0.60 -> 0.75 is material."
    correlation_drift_threshold: float = 0.15
    correlation_drift_enabled: bool = True

    # Regime-change trigger: fire a regen when MacroAgent's regime label
    # transitions (bull <-> bear <-> sideways <-> high_vol).
    regime_change_trigger_enabled: bool = True

    # Suppress the picker's frozen on-deck offer for tickers we fully-exited
    # within this window. Prevents the "in-out-in same session" whipsaw where
    # the picker keeps offering back a name the LLM just decided to drop.
    # Scoped to intraday (60 min default) so overnight re-considers still work.
    picker_exit_suppress_minutes: int = 60

    # Hard cap on frozen_picker_tickers to bound prompt-size growth from
    # #104 bypass promotions. Observed 2026-09-04 the list ballooned 12 → 39
    # in 30 min, doubling per-regen prompt tokens and collapsing cache hits
    # to 0-6%. When promotion would push the list past the cap, drop the
    # oldest-promoted names first (held/originally-picked names are kept).
    max_frozen_picker_size: int = 25

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
    # {ticker: [(url, iso_ts), ...]} - distinct news URLs seen per ticker in
    # the recent past. Used to grant per-ticker cooldown bypass only when
    # >= N distinct sources converge on the same name (see #94).
    news_source_urls: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # Correlation-matrix snapshot for drift detection. Keyed by "A|B" with
    # tickers sorted alphabetically, value is the pairwise correlation
    # captured just after the last regen. Compared against a fresh compute
    # each tick to decide whether to fire a correlation-drift regen.
    last_correlation_snapshot: dict[str, float] = field(default_factory=dict)
    # Most recent macro regime label. When the fresh MacroAgent call returns
    # a different label, that transition fires a regen.
    last_regime: str | None = None
    # Staging area for news events that arrived since the last regen. On the
    # next regen decision they're either attributed to a ticker delta (moved
    # to news_effect_log) or dropped as non-actionable (barely-moved skip).
    pending_news_events: list[dict] = field(default_factory=list)
    # Per-ticker journal of news events that PROVED to move the book. Entries
    # persist across regens with a TTL safety valve. Doubles as anti-ladder
    # discipline signal in the prompt: LLM sees "SNOW: added 4 times on RBC/
    # WF/MS/UBS notes, cumulative +14sh" and self-restrains at concentration.
    news_effect_log: dict[str, list[dict]] = field(default_factory=dict)
    # Ticker -> ISO timestamp of when it was promoted to on-deck via
    # materiality_bypass_promoted. Feeds the on-deck section of the LLM
    # prompt so the model can judge staleness ("42m ago, no follow-up →
    # nominate for on_deck_purge"). Held tickers are not tracked here.
    promoted_at: dict[str, str] = field(default_factory=dict)
    # Parallel to promoted_at: ticks_run value at promotion time. Lets the
    # prompt render "promoted at tick 89 (now tick 145) = 56 ticks stale"
    # alongside the minutes-ago figure.
    promoted_at_tick: dict[str, int] = field(default_factory=dict)
    # Ticker -> price snapshot when the ticker's MOST RECENT news event
    # arrived. Rendered as "news-time $X, now $Y (±Z%)" so the LLM can
    # see whether the market already priced in the headline.
    last_news_price: dict[str, float] = field(default_factory=dict)
    # Rolling in-memory (last 1h) list of (iso_ts, cost_usd) samples used
    # to fire `cost_alert` events when spend crosses a threshold. Not
    # persisted — a fresh 1h window on restart is fine.
    cost_window: list[tuple[str, float]] = field(default_factory=list)

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
            "news_source_urls": {
                k: [list(t) for t in v] for k, v in self.news_source_urls.items()
            },
            "last_correlation_snapshot": dict(self.last_correlation_snapshot),
            "last_regime": self.last_regime,
            "pending_news_events": [dict(e) for e in self.pending_news_events],
            "news_effect_log": {
                k: [dict(e) for e in v] for k, v in self.news_effect_log.items()
            },
            "promoted_at": dict(self.promoted_at),
            "promoted_at_tick": dict(self.promoted_at_tick),
            "last_news_price": dict(self.last_news_price),
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
            news_source_urls={
                k: [(t[0], t[1]) for t in v]
                for k, v in (d.get("news_source_urls") or {}).items()
            },
            last_correlation_snapshot=dict(
                d.get("last_correlation_snapshot") or {}
            ),
            last_regime=d.get("last_regime"),
            pending_news_events=[dict(e) for e in (d.get("pending_news_events") or [])],
            news_effect_log={
                k: [dict(e) for e in v]
                for k, v in (d.get("news_effect_log") or {}).items()
            },
            promoted_at=dict(d.get("promoted_at") or {}),
            promoted_at_tick=dict(d.get("promoted_at_tick") or {}),
            last_news_price={
                k: float(v) for k, v in (d.get("last_news_price") or {}).items()
            },
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


def _stage_pending_news(state, events, now) -> None:
    """Add each incoming news event to the pending staging area.

    Each entry captures ticker, ts, source, headline, and a short summary.
    On the next regen decision the event either gets attributed to a ticker
    delta and promoted into news_effect_log, or dropped if the LLM's own
    barely-moved decision proved it non-actionable.
    """
    ts_iso = now.isoformat()
    for e in events:
        ticker = (e.ticker or "").upper()
        if not ticker:
            continue
        state.pending_news_events.append({
            "ts": ts_iso,
            "ticker": ticker,
            "source": (e.source or "")[:40],
            "headline": (e.headline or "")[:200],
            "summary": (e.summary or "")[:200],
        })


def _attribute_pending_news(state, new_rec, prev_rec, min_delta_pp: float) -> None:
    """Promote pending news to news_effect_log when the ticker's rec delta
    exceeded the threshold. Deltas are computed vs the previous rec's targets
    (or vs zero if the ticker wasn't in the prior rec). Non-attributed
    events are dropped - the LLM saw them and chose not to act.
    """
    if not state.pending_news_events:
        return
    new_targets = {
        p.ticker.upper(): float(p.weight_pct) for p in new_rec.allocation.positions
    }
    prev_targets = {}
    if prev_rec is not None:
        prev_targets = {
            p.ticker.upper(): float(p.weight_pct)
            for p in prev_rec.allocation.positions
        }
    deltas = {}
    for t in set(new_targets) | set(prev_targets):
        d = new_targets.get(t, 0.0) - prev_targets.get(t, 0.0)
        if abs(d) >= min_delta_pp:
            deltas[t] = d
    for event in state.pending_news_events:
        ticker = event["ticker"]
        delta = deltas.get(ticker)
        if delta is None:
            continue
        entry = dict(event)
        entry["delta_pp"] = round(delta, 2)
        state.news_effect_log.setdefault(ticker, []).append(entry)
    state.pending_news_events = []


def _drop_pending_news(state) -> None:
    """Clear pending events without attribution. Called when the LLM's own
    decision (opinion_drift_skip=barely-moved) proved the batch non-actionable.
    """
    state.pending_news_events = []


def _prune_news_effect_log(state, now, ttl_seconds: int = 86400) -> None:
    """Age out effect-log entries older than TTL. Safety valve so stale
    entries don't hang around forever if the LLM never revisits the ticker.
    """
    from datetime import datetime as _dt

    cutoff = now.timestamp() - ttl_seconds
    empty_keys = []
    for ticker, entries in state.news_effect_log.items():
        kept = []
        for e in entries:
            try:
                ts_dt = _dt.fromisoformat(e["ts"])
                if ts_dt.timestamp() >= cutoff:
                    kept.append(e)
            except Exception:  # noqa: BLE001
                kept.append(e)  # keep if we can't parse
        if kept:
            state.news_effect_log[ticker] = kept
        else:
            empty_keys.append(ticker)
    for k in empty_keys:
        state.news_effect_log.pop(k, None)


def _render_news_effect_log(state) -> str:
    """Compact per-ticker journal for the allocator prompt. Each ticker gets
    one line: "TKR: HH:MM source headline_short -> +Xpp | HH:MM ...".
    Empty string when no ticker has any attributed entries.
    """
    if not state.news_effect_log:
        return ""
    lines: list[str] = []
    for ticker in sorted(state.news_effect_log):
        entries = state.news_effect_log[ticker]
        if not entries:
            continue
        parts = []
        for e in entries[-6:]:  # last 6 effective entries per ticker
            ts_short = str(e.get("ts", ""))[11:16]  # "HH:MM"
            head = (e.get("headline") or "")[:70]
            delta = e.get("delta_pp")
            delta_str = f" -> {delta:+.1f}pp" if delta is not None else ""
            parts.append(f"{ts_short} {head}{delta_str}")
        lines.append(f"{ticker}: " + " | ".join(parts))
    return "\n".join(lines)


def _held_ticker_set(broker) -> set[str]:
    """Uppercase set of tickers currently held. Empty on broker error."""
    try:
        return {p.ticker.upper() for p in broker.get_positions()}
    except Exception:  # noqa: BLE001
        return set()


def _same_session_exits(broker, *, since_minutes: int) -> set[str]:
    """Uppercase tickers that were sold in the last N minutes AND currently
    show zero shares. Used to suppress the picker's on-deck offer for names
    the LLM decided to drop this session, so the same name doesn't get
    re-baited on every regen.
    """
    try:
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        from agentic_investor.tools.paper_store import recent_sold_tickers
        since = (_dt.now(_UTC) - _td(minutes=since_minutes)).isoformat()
        sold = {t.upper() for t in recent_sold_tickers(since_iso=since)}
    except Exception:  # noqa: BLE001
        return set()
    held = _held_ticker_set(broker)
    return sold - held


def _material_ticker_set(state, broker) -> set[str]:
    """Union of tickers we care about: held + picker-frozen + recent exits.

    A news batch touching only non-material tickers isn't worth burning an
    LLM regen on. Recent exits stay material for 2h so a same-session
    whipsaw-back can still be prompted by real news, but stale exits from
    yesterday no longer bloat the material set (was 24h → 2h on 2026-09-04
    to reduce noise wake-ups).
    """
    material = _held_ticker_set(broker)
    for t in (state.frozen_picker_tickers or []):
        material.add(t.upper())
    try:
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        from agentic_investor.tools.paper_store import recent_sold_tickers
        since = (_dt.now(_UTC) - _td(hours=2)).isoformat()
        for t in recent_sold_tickers(since_iso=since):
            material.add(t.upper())
    except Exception:  # noqa: BLE001 - recent-exits lookup is best-effort
        pass
    return material


_HIGH_SIGNAL_TOKENS = (
    "price target",       # "Raises Price Target", "Lowers Price Target", etc.
    "upgrades",           # explicit rating upgrade
    "downgrades",         # explicit rating downgrade
    "initiates coverage", # new analyst coverage
    "reinstates",         # reinstated coverage
)


def _is_high_signal_headline(headline: str) -> bool:
    """Return True when a headline looks like an analyst rating action.

    Materiality filter drops non-book news to avoid burning LLM budget on
    tickers we don't track. But a fresh upgrade on a ticker we don't yet
    hold IS actionable - it should reach the LLM so the picker's next
    universe pass can consider it, and so the current LLM run can flag
    the ticker in its rationale. Keyword-based classification is coarse
    but catches the analyst-note pattern that drove SNOW today.
    """
    if not headline:
        return False
    h = headline.lower()
    return any(t in h for t in _HIGH_SIGNAL_TOKENS)


def _snapshot_pairs(matrix) -> dict[str, float]:
    """Flatten a correlation DataFrame into {'A|B': corr} with A<B sorted."""
    out: dict[str, float] = {}
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                v = float(matrix.loc[a, b])
            except Exception:  # noqa: BLE001
                continue
            if v != v:  # NaN
                continue
            key = f"{a}|{b}" if a < b else f"{b}|{a}"
            out[key] = v
    return out


def _correlation_shifts(
    prev: dict[str, float],
    cur: dict[str, float],
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Pairs whose absolute correlation shifted by more than `threshold`.

    Only compares pairs present in both snapshots so a change in the held
    universe doesn't count as a shift by itself.
    """
    out: dict[str, dict[str, float]] = {}
    for key, cur_v in cur.items():
        prev_v = prev.get(key)
        if prev_v is None:
            continue
        delta = abs(cur_v - prev_v)
        if delta >= threshold:
            out[key] = {
                "prev": round(prev_v, 3),
                "cur": round(cur_v, 3),
                "delta": round(cur_v - prev_v, 3),
            }
    return out


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
    on_deck_watchlist: list[str] | None = None,
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
    if cfg.max_positions_override is not None:
        profile = profile.model_copy(update={"max_positions": cfg.max_positions_override})
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
        on_deck_watchlist=on_deck_watchlist,
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
        # Rolling-hour cost alert: keep the last 60 min of tick costs,
        # sum them, and emit a `cost_alert` event when we cross the
        # threshold. Fires once when we cross and again each subsequent
        # tick above the line (no dedup) so the alert is visible in the
        # session tail rather than easy to miss.
        if cost > 0:
            now_iso = tick_at
            state.cost_window.append((now_iso, cost))
            try:
                from datetime import UTC as _UTC
                from datetime import datetime as _dt
                from datetime import timedelta as _td
                cutoff = _dt.now(_UTC) - _td(minutes=60)
                state.cost_window = [
                    (ts, c) for ts, c in state.cost_window
                    if _dt.fromisoformat(ts.replace("Z", "+00:00")) >= cutoff
                ]
            except Exception:  # noqa: BLE001
                pass
            hourly = sum(c for _, c in state.cost_window)
            if hourly > cfg.cost_alert_per_hour_usd:
                session.log("cost_alert", {
                    "hourly_cost_usd": round(hourly, 4),
                    "threshold_usd": cfg.cost_alert_per_hour_usd,
                    "window_samples": len(state.cost_window),
                    "latest_tick_cost": round(cost, 4),
                })

    # Two-tier: regenerate the recommendation once per day OR when the event
    # loop pushed news batch context onto state; reuse otherwise.
    regenerated = False
    # Set True when we're using a prior rec because opinion_drift said
    # "barely-moved". In that mode the LLM offered no fresh conviction, so
    # only walk-back (REDUCE-direction) trades execute - blocks silent-drift
    # ADD trades from sneaking through when the LLM's target drifts upward
    # imperceptibly across several skipped regens.
    fall_through_reduce_only = False
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
        # Suppress tickers we fully-exited within the last N min so the picker
        # doesn't keep offering the same name back and the LLM doesn't keep
        # re-considering it. Root cause of the ZS/TRV in-out-in whipsaws
        # observed 2026-09-04.
        if pre_picked and cfg.picker_exit_suppress_minutes > 0:
            suppressed = _same_session_exits(
                broker, since_minutes=cfg.picker_exit_suppress_minutes,
            )
            if suppressed:
                filtered = [t for t in pre_picked if t.upper() not in suppressed]
                if len(filtered) != len(pre_picked) and session:
                    session.log("picker_exit_suppressed", {
                        "dropped": sorted(
                            t.upper() for t in pre_picked
                            if t.upper() in suppressed
                        ),
                        "n_kept": len(filtered),
                        "window_min": cfg.picker_exit_suppress_minutes,
                    })
                pre_picked = filtered or None

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
                session.log("knob_fired", {
                    "name": "pre_check", "reason": "batch-unchanged",
                })
            from agentic_investor.orchestrator.store import (
                load_recommendation as _load_prev,
            )
            rec = _load_prev(state.last_rec_id)
            regenerated = False
            tickers_used = state.frozen_picker_tickers or []
            # Refresh the cooldown clock even though we skipped the LLM
            # call: from the hot-signal fast-path's perspective this is
            # still a "decision was made this instant" event. Missing
            # this update caused the 2026-09-04 gap where hot news
            # firing 5-7s after a pre-check-skipped tick bypassed the
            # 60s cooldown.
            state.last_regen_at = now
        else:
            # Delta-form prompt anchor (non-first-day only).
            prev_rec_for_prompt = None
            if not is_new_day and state.last_rec_id is not None:
                from agentic_investor.orchestrator.store import (
                    load_recommendation as _load_for_anchor,
                )
                prev_rec_for_prompt = _load_for_anchor(state.last_rec_id)
            # Prepend the per-ticker effect journal to the batch context so
            # the LLM sees "you added SNOW 4 times already, cumulative +14sh"
            # alongside the new batch events. Doubles as anti-ladder discipline.
            journal = _render_news_effect_log(state)
            combined_ctx = batch_ctx or ""
            if journal:
                combined_ctx = (
                    "News-effect journal (past batches this session that moved the book):\n"
                    + journal
                    + ("\n\nCurrent batch:\n" + batch_ctx if batch_ctx else "")
                )
            # On-deck watchlist with staleness signals for LLM purge judgment:
            # each entry gets promoted-age + minutes-since-last-news. LLM can
            # nominate stale ones via rec.allocation.on_deck_purge.
            on_deck_now = list(state.frozen_picker_tickers or [])
            on_deck_meta: list[dict] = []
            current_tick = state.ticks_run
            for t in on_deck_now:
                up = t.upper()
                entry: dict = {"ticker": up}
                promoted_iso = state.promoted_at.get(up)
                if promoted_iso:
                    try:
                        p_dt = datetime.fromisoformat(promoted_iso)
                        if p_dt.tzinfo is None:
                            p_dt = p_dt.replace(tzinfo=UTC)
                        entry["promoted_min_ago"] = int(
                            (now - p_dt).total_seconds() / 60
                        )
                    except Exception:  # noqa: BLE001
                        pass
                p_tick = state.promoted_at_tick.get(up)
                if p_tick is not None:
                    entry["promoted_ticks_ago"] = current_tick - p_tick
                # Price-at-news vs current price so the LLM can see whether
                # the market already reacted to the story.
                news_px = state.last_news_price.get(up)
                if news_px:
                    try:
                        now_px = float(price_fetcher(up))
                        entry["news_price"] = round(news_px, 2)
                        entry["now_price"] = round(now_px, 2)
                        if news_px > 0:
                            entry["price_change_pct"] = round(
                                (now_px / news_px - 1) * 100, 2
                            )
                    except Exception:  # noqa: BLE001
                        pass
                # Last-news lookup from news_source_urls (already tracked).
                news_entries = state.news_source_urls.get(up) or []
                if news_entries:
                    last_ts = max(ts for _, ts in news_entries)
                    try:
                        n_dt = datetime.fromisoformat(last_ts)
                        if n_dt.tzinfo is None:
                            n_dt = n_dt.replace(tzinfo=UTC)
                        entry["last_news_min_ago"] = int(
                            (now - n_dt).total_seconds() / 60
                        )
                        entry["news_count"] = len(news_entries)
                    except Exception:  # noqa: BLE001
                        pass
                on_deck_meta.append(entry)
            rec, tickers_used = _generate_recommendation(
                cfg, news_batch_context=combined_ctx, pre_picked_tickers=pre_picked,
                previous_rec=prev_rec_for_prompt,
                extra_tickers=promoted or None,
                on_deck_watchlist=on_deck_meta,
            )
            state.last_batch_fingerprint = batch_fp
            # Honor the LLM's on-deck purge nominations. Never drops held
            # tickers (safety) and never drops the tickers we're about to
            # trade in this rec (they're active decisions).
            purge = getattr(rec.allocation, "on_deck_purge", None) or []
            if purge and state.frozen_picker_tickers:
                held_now = _held_ticker_set(broker)
                active_targets = {
                    p.ticker.upper() for p in rec.allocation.positions
                }
                purge_set = {t.upper() for t in purge}
                # Never purge held or actively-targeted tickers.
                effective_purge = purge_set - held_now - active_targets
                if effective_purge:
                    before = len(state.frozen_picker_tickers)
                    state.frozen_picker_tickers = [
                        t for t in state.frozen_picker_tickers
                        if t.upper() not in effective_purge
                    ]
                    if session:
                        session.log("on_deck_purge_applied", {
                            "rec_id": state.last_rec_id,
                            "purged": sorted(effective_purge),
                            "requested": sorted(purge_set),
                            "size_before": before,
                            "size_after": len(state.frozen_picker_tickers),
                        })
                        session.log("knob_fired", {
                            "name": "on_deck_purge",
                            "reason": "llm-nominated",
                        })

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
                    session.log("knob_fired", {
                        "name": "opinion_drift", "reason": skip_reason,
                    })
                # Stable opinion doesn't mean the book matches the target -
                # earlier orders may have failed or been cooldown-blocked.
                # Fall through to drift-vs-book against the prior rec.
                # avg-drift-too-high / max-delta-unjustified are noise skips
                # and still bail here.
                if skip_reason == "barely-moved" and prev_rec is not None:
                    rec = prev_rec
                    regenerated = False
                    fall_through_reduce_only = True
                    # The LLM's own decision proved the pending news was
                    # non-actionable. Drop the staging area so those events
                    # never enter the effect log.
                    _drop_pending_news(state)
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
                # Snapshot pairwise correlations on the fresh held set so the
                # drift trigger has a baseline to compare against on later ticks.
                try:
                    held_syms = {p.ticker.upper() for p in rec.allocation.positions}
                    if len(held_syms) >= 2:
                        from agentic_investor.orchestrator.correlation import (
                            compute_correlation_matrix,
                        )
                        matrix = compute_correlation_matrix(sorted(held_syms))
                        if matrix is not None:
                            state.last_correlation_snapshot = _snapshot_pairs(matrix)
                except Exception as e:  # noqa: BLE001
                    logger.debug("correlation snapshot failed: %s", e)
                regenerated = True
                # Attribute pending news to the tickers whose targets shifted
                # enough to be considered driven by this batch. Non-attributed
                # events drop; attributed ones become the persistent journal.
                _attribute_pending_news(state, rec, prev_rec, min_delta_pp=1.0)
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
                    for ev in getattr(rec, "repair_events", []) or []:
                        session.log("alloc_repair", {"rec_id": rec_id, **ev})

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
            # First rec of the day, no prev to diff against - attribute
            # pending news against a zero baseline so opening trades still
            # produce journal entries.
            _attribute_pending_news(state, rec, None, min_delta_pp=1.0)
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
                for ev in getattr(rec, "repair_events", []) or []:
                    session.log("alloc_repair", {"rec_id": rec_id, **ev})
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

    # Distinct-source count per ticker over the convergence window. Prune
    # stale entries in place so state doesn't grow unbounded.
    conv_window = int(getattr(cfg, "news_convergence_window_sec", 900))
    cutoff = now - timedelta(seconds=conv_window)
    news_source_counts: dict[str, int] = {}
    empty_keys: list[str] = []
    for ticker_up, entries in state.news_source_urls.items():
        kept: list[tuple[str, str]] = []
        for url_key, ts_iso in entries:
            try:
                ts_dt = datetime.fromisoformat(ts_iso)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=UTC)
            except Exception:  # noqa: BLE001
                continue
            if ts_dt >= cutoff:
                kept.append((url_key, ts_iso))
        if kept:
            state.news_source_urls[ticker_up] = kept
            news_source_counts[ticker_up] = len(kept)
        else:
            empty_keys.append(ticker_up)
    for k in empty_keys:
        state.news_source_urls.pop(k, None)

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
        prices=prices,
        min_open_dollars=cfg.min_open_dollars,
        min_add_dollars=cfg.min_add_dollars,
        min_trim_dollars=cfg.min_trim_dollars,
        recent_trades=recent_typed,
        cooldown_seconds=getattr(cfg, "cooldown_seconds", 900),
        now=now,
        news_batch_tickers=batch_tickers_for_cooldown,
        news_source_counts=news_source_counts,
        min_bypass_sources=int(getattr(cfg, "news_convergence_min_sources", 1)),
        ticker_recent_moves=ticker_recent_moves,
        avg_entry_prices=avg_entry_prices,
        force_loss_cut_pct=cfg.force_loss_cut_pct,
        max_add_concentration_pct=cfg.max_add_concentration_pct,
    )

    # On barely-moved fall-through the LLM offered no fresh conviction, so
    # ADDs would be silent-drift executions of accumulated small target
    # bumps across skipped regens. Keep only trims/closes.
    if fall_through_reduce_only and plans:
        n_before = len(plans)
        plans = [p for p in plans if p.side == "sell"]
        n_dropped = n_before - len(plans)
        if n_dropped and session:
            session.log("fall_through_add_blocked", {
                "rec_id": state.last_rec_id,
                "n_dropped": n_dropped,
                "n_kept": len(plans),
            })
            session.log("knob_fired", {
                "name": "fall_through", "reason": "reduce_only",
            })

    # Walk-back re-derives from `positions()`, which reports shares OWNED but
    # not shares HELD_FOR_ORDERS. When a prior tick's sell is still in flight
    # the walk-back "sees" full qty and re-submits an identical sell that
    # Alpaca rejects with insufficient qty. Drop any (ticker, side) already
    # covered by an open order.
    if fall_through_reduce_only and plans:
        try:
            open_orders = broker.list_orders(status="open")
        except Exception:  # noqa: BLE001
            open_orders = []
        in_flight = {(o.ticker.upper(), o.side) for o in open_orders}
        n_before = len(plans)
        plans = [p for p in plans if (p.ticker.upper(), p.side) not in in_flight]
        n_dropped = n_before - len(plans)
        if n_dropped and session:
            session.log("fall_through_in_flight_dropped", {
                "rec_id": state.last_rec_id,
                "n_dropped": n_dropped,
                "n_kept": len(plans),
            })
            session.log("knob_fired", {
                "name": "fall_through", "reason": "in_flight",
            })

    if not plans and session and regenerated:
        # A fresh rec produced zero trades because every diff sits under the
        # band. Log what was suppressed so silent no-ops on new-position
        # proposals (e.g. CRM 5% target -> 0% held == 5pp == band edge) don't
        # disappear from the audit trail.
        allocation_base_for_diag = allocation_base if allocation_base else 1.0
        suppressed = []
        for pos in rec.allocation.positions:
            tk = pos.ticker.upper()
            target_dollars = pos.weight_pct / 100.0 * allocation_base_for_diag
            current_dollars = positions_dollars.get(tk, 0.0)
            delta_dollars = target_dollars - current_dollars
            delta_pp = pos.weight_pct - (
                current_dollars / allocation_base_for_diag * 100.0
                if allocation_base_for_diag > 0 else 0.0
            )
            if abs(delta_pp) >= 1.0:
                suppressed.append({
                    "ticker": tk,
                    "target_pct": round(pos.weight_pct, 2),
                    "delta_pp": round(delta_pp, 2),
                    "delta_dollars": round(delta_dollars, 2),
                })
        if suppressed:
            session.log("rebalance_skipped_all_below_band", {
                "rec_id": state.last_rec_id,
                "band_abs_pct": cfg.band_abs_pct,
                "band_rel_pct": cfg.band_rel_pct,
                "suppressed": suppressed,
            })

    if cfg.dry_run or not plans:
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
        # Regen attribution: capture what trigger produced trades and on
        # which tickers, so we can grep "how often does the trigger
        # ticker actually appear as the decision subject" across
        # sessions. Answers the recurring "MRK exited on PATH news"
        # pattern (trigger != decision-subject) with data.
        trigger_batch_tickers = sorted(batch_tickers_for_cooldown or set())
        decision_tickers = sorted({p.ticker.upper() for p in plans})
        overlap = set(trigger_batch_tickers) & set(decision_tickers)
        session.log("regen_attribution", {
            "rec_id": state.last_rec_id,
            "trigger": trigger,
            "trigger_tickers": trigger_batch_tickers,
            "decision_tickers": decision_tickers,
            "trigger_matches_decision": sorted(overlap),
            "n_trades": len(plans),
        })
    submitted = execute_trade_plan(
        plans, broker, rec_id=state.last_rec_id,
        stop_loss_pct=cfg.stop_loss_pct, take_profit_pct=cfg.take_profit_pct,
        day=today,
    )
    # Position rationales keyed by ticker so we can attach them to
    # order_submitted events - useful for auditing "why did the LLM buy?"
    rationale_by_ticker = {
        p.ticker.upper(): p.rationale for p in rec.allocation.positions
    }
    # Snapshot per-ticker unrealized P/L% so we can flag underwater adds
    # (LLM doubling down on a name that's already in the red). Not a
    # block - just data. If the pattern turns out to lose money over
    # weeks, we'll add discipline; for now we want the frequency.
    unrealized_pct_by_ticker = {
        p.ticker.upper(): float(p.unrealized_pl_pct) for p in positions
    }
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
            # No-news-buy audit: flag BUY trades where the ticker had no news
            # in the current batch AND no attributed effect-log entry. Not a
            # block - just observability. The LLM legitimately adds on
            # correlation-driven rebalancing, drift-back-to-target, etc.
            # Later analysis can decide if the pattern is productive.
            if o.side == "buy":
                tkr = o.ticker.upper()
                had_batch_news = tkr in batch_tickers_for_cooldown
                had_effect_entry = bool(state.news_effect_log.get(tkr))
                if not had_batch_news and not had_effect_entry:
                    session.log("no_news_buy", {
                        "ticker": tkr,
                        "qty": o.qty,
                        "rationale": (rationale_by_ticker.get(tkr) or "")[:300],
                    })
                # Underwater-add audit: flag BUY trades that add to a name
                # currently down more than 1% from average entry. Doubling
                # down on losers is a well-known LLM failure mode; we
                # observed it on IOT (-1.64%, +15sh add) today.
                pl_pct = unrealized_pct_by_ticker.get(tkr)
                if pl_pct is not None and pl_pct < -1.0:
                    session.log("underwater_add", {
                        "ticker": tkr,
                        "qty": o.qty,
                        "unrealized_pl_pct": round(pl_pct, 2),
                        "rationale": (rationale_by_ticker.get(tkr) or "")[:300],
                    })
    state.orders_submitted += len(submitted)
    # Persist post-execution state so recent_trades + trade counters survive
    # restart. The earlier post-regen save runs BEFORE trade execution.
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
    # Cap at 15 min per call so SIGINT lands quickly, floor at 1s so we don't
    # tight-spin when the broker clock lags the wall clock by <1s at the bell
    # (which shows up as dozens of market_closed events per second at 09:30).
    time.sleep(max(1.0, min(delta, 900)))


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
            hot_signal_fire = False
            if new_events and session:
                for e in new_events:
                    session.log("news_received", {
                        "ticker": e.ticker,
                        "headline": e.headline[:120],
                    })
            # Snapshot price-at-news for on-deck / held tickers so the LLM
            # can see "did the market already price this in?" when judging
            # staleness. Best-effort: single last-price wins if multiple
            # events on one ticker arrive in the same drain.
            if new_events:
                try:
                    from agentic_investor.tools.paper_broker import get_latest_price
                except Exception:  # noqa: BLE001
                    get_latest_price = None
                tracked = set(t.upper() for t in (state.frozen_picker_tickers or []))
                tracked |= _held_ticker_set(broker)
                seen_now: set[str] = set()
                if get_latest_price is not None:
                    for e in new_events:
                        tk = (e.ticker or "").upper()
                        if not tk or tk in seen_now or tk not in tracked:
                            continue
                        try:
                            state.last_news_price[tk] = float(get_latest_price(tk))
                            seen_now.add(tk)
                        except Exception:  # noqa: BLE001
                            pass
            # Per-headline finBERT fast-path: if any single arriving headline
            # scores above the immediate threshold, force-close the batch
            # window and fire NOW instead of waiting for the aggregate to
            # possibly dilute the signal. Only active when finbert prefilter
            # is enabled (uses the same pipeline).
            if new_events and cfg.finbert_prefilter_enabled:
                try:
                    from agentic_investor.orchestrator.finbert_prefilter import (
                        score_single,
                    )
                    for e in new_events:
                        headline = str(getattr(e, "headline", "") or "")
                        summary = str(getattr(e, "summary", "") or "")[:200]
                        combined = f"{headline}. {summary}" if summary else headline
                        score = score_single(combined)
                        if score is None:
                            continue
                        if abs(score) >= cfg.finbert_immediate_threshold:
                            # Cooldown: if a regen just fired recently, don't
                            # force another one — event is already in the
                            # pending buffer and will be picked up by the
                            # next natural batch closure. Coalesces the
                            # "hot news arrives 5s after regen completed"
                            # pattern.
                            recent_regen = (
                                state.last_regen_at is not None
                                and (now - state.last_regen_at).total_seconds()
                                    < cfg.finbert_hot_signal_cooldown_seconds
                            )
                            if recent_regen:
                                if session:
                                    session.log("finbert_hot_signal_deferred", {
                                        "ticker": e.ticker,
                                        "headline": headline[:120],
                                        "score": round(score, 3),
                                        "seconds_since_regen": round(
                                            (now - state.last_regen_at).total_seconds(),
                                            1,
                                        ),
                                        "cooldown_seconds":
                                            cfg.finbert_hot_signal_cooldown_seconds,
                                    })
                                break  # still one hot per drain
                            hot_signal_fire = True
                            if session:
                                session.log("finbert_hot_signal", {
                                    "ticker": e.ticker,
                                    "headline": headline[:120],
                                    "score": round(score, 3),
                                    "threshold": cfg.finbert_immediate_threshold,
                                })
                                session.log("knob_fired", {
                                    "name": "finbert_prefilter",
                                    "reason": "hot-signal-immediate",
                                })
                            break  # one hot headline is enough
                except Exception as ex:  # noqa: BLE001
                    logger.debug("finbert per-headline fast-path error: %s", ex)
            # Track distinct news URLs per ticker for the convergence bypass.
            # Same URL fanned out across many tickers still counts as one
            # signal per ticker.
            if new_events:
                ts_iso = now.isoformat()
                for e in new_events:
                    key = (e.ticker or "").upper()
                    if not key:
                        continue
                    url_key = e.url or e.headline[:80]
                    entries = state.news_source_urls.setdefault(key, [])
                    if not any(u == url_key for u, _ in entries):
                        entries.append((url_key, ts_iso))
                # Also stage them for effect-log attribution on the next
                # regen decision.
                _stage_pending_news(state, new_events, now)
            _prune_news_effect_log(state, now)
            ingest(decision_state, new_events, now)

            fire, reason = should_fire(decision_state, now)
            # Per-headline hot-signal fast-path overrides the batch-window
            # closure: force fire now with an explicit reason so the
            # aggregate finBERT prefilter below doesn't dilute+skip the
            # signal we just proved was hot.
            if hot_signal_fire:
                fire = True
                reason = "finbert-hot-headline"

            # Materiality gate: drop news-batch fires whose tickers don't
            # intersect with the tickers we care about (held + on-deck +
            # recent exits). Non-material news mostly produced barely-moved
            # skips - burn LLM budget, learn nothing. A high-signal analyst
            # action on a non-material ticker bypasses the drop so we don't
            # miss actionable upgrades on names outside the current universe.
            if fire and cfg.materiality_filter_enabled and decision_state.unprocessed:
                material = _material_ticker_set(state, broker)
                batch_tickers = {
                    (e.ticker or "").upper() for e in decision_state.unprocessed
                }
                overlap = batch_tickers & material
                if not overlap:
                    high_signal_events = [
                        e for e in decision_state.unprocessed
                        if _is_high_signal_headline(e.headline or "")
                    ]
                    if high_signal_events:
                        # Promote high-signal non-material tickers to on-deck
                        # instead of firing a full regen. Keeps the discovery
                        # pipeline alive (LLM will see them on its next
                        # naturally-triggered regen) without burning $0.006
                        # per ambient PT-change headline. Fixes the leak that
                        # dropped MRK on unrelated PATH news at 11:55 today.
                        promoted_tickers = sorted({
                            (e.ticker or "").upper()
                            for e in high_signal_events
                        })
                        # Append to frozen_picker_tickers so subsequent regens
                        # include them via pre_picked. Cap to keep prompt bounded.
                        if state.frozen_picker_tickers is None:
                            state.frozen_picker_tickers = []
                        existing = {t.upper() for t in state.frozen_picker_tickers}
                        # Whitelist filter: micro-caps that pass the news
                        # bypass keyword check but aren't in the S&P
                        # actionable set often error every regen at the
                        # yfinance boundary (XTND, VBNK, PDEX, ...). Same
                        # gate the 0w news-body promotion path already uses.
                        from agentic_investor.universes import is_actionable_ticker
                        newly_promoted = [
                            t for t in promoted_tickers
                            if t not in existing and is_actionable_ticker(t)
                        ]
                        for t in newly_promoted:
                            state.frozen_picker_tickers.append(t)
                        # Record promotion timestamps + tick-count for the
                        # newly added tickers. LLM sees both time-ago and
                        # tick-delta in the on-deck section so it can judge
                        # staleness in whichever framing is more meaningful.
                        now_iso = now.isoformat()
                        current_tick = state.ticks_run
                        for t in newly_promoted:
                            state.promoted_at[t.upper()] = now_iso
                            state.promoted_at_tick[t.upper()] = current_tick
                        # Enforce the hard cap: if the list overflows, drop
                        # oldest-promoted first. Held tickers and originally
                        # picked names are earlier in the list so they survive
                        # via FIFO ordering (append-at-end promotion).
                        dropped_by_cap = []
                        cap = cfg.max_frozen_picker_size
                        held_now = _held_ticker_set(broker)
                        while len(state.frozen_picker_tickers) > cap:
                            # Drop from index 0 (oldest) BUT skip held tickers
                            # — those are the real book, never drop them.
                            drop_idx = None
                            for i, t in enumerate(state.frozen_picker_tickers):
                                if t.upper() not in held_now:
                                    drop_idx = i
                                    break
                            if drop_idx is None:
                                break  # nothing to drop (all held)
                            dropped_by_cap.append(
                                state.frozen_picker_tickers.pop(drop_idx)
                            )
                        if session:
                            session.log("materiality_bypass_promoted", {
                                "reason": "high_signal_non_material",
                                "tickers": promoted_tickers,
                                "newly_added": newly_promoted,
                                "n_events": len(high_signal_events),
                                "cap_dropped": dropped_by_cap,
                                "cap": cap,
                                "size": len(state.frozen_picker_tickers),
                            })
                            session.log("knob_fired", {
                                "name": "materiality",
                                "reason": "promoted_to_on_deck",
                            })
                        # Keep the news in decision_state.unprocessed so it
                        # renders in the next natural batch context. Don't
                        # fire the regen now — that's the whole point.
                        # Advance last_fire_at so the next iteration doesn't
                        # immediately re-trigger the natural fire path on
                        # the same buffer.
                        fire = False
                        reason = ""
                        decision_state.last_fire_at = now
                    else:
                        if session:
                            session.log("materiality_skip", {
                                "batch_tickers": sorted(batch_tickers),
                                "material_count": len(material),
                            })
                            session.log("knob_fired", {
                                "name": "materiality", "reason": "non-material",
                            })
                        # Clear the batch so it doesn't accumulate and eventually
                        # fire on its own without any material follow-up.
                        decision_state.unprocessed = []
                        decision_state.last_fire_at = now
                        fire = False
                        reason = ""
            interval_due = (
                (now - last_interval_tick).total_seconds() >= cfg.interval_seconds
            )

            # Extra triggers layered on top of news + interval:
            # (a) price-move: any held ticker moved > threshold from baseline
            # (b) correlation-drift: pairwise correlation shifted materially
            # (c) regime-change: macro regime label flipped since last regen
            # (d) force-regen: last regen was more than force_regen_seconds ago
            # (e) technical-stance: any held ticker's technical stance flipped
            price_ctx = ""
            price_fire = False
            if state.baseline_prices:
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
                        if not fire:
                            reason = "price-move"
                        if session:
                            session.log("price_move_trigger", {"moves": moves})
                except Exception as e:  # noqa: BLE001
                    logger.warning("price-move check failed: %s", e)

            corr_fire = False
            if (not fire and not price_fire and cfg.correlation_drift_enabled
                    and state.last_correlation_snapshot):
                try:
                    held_syms = _held_ticker_set(broker)
                    if len(held_syms) >= 2:
                        from agentic_investor.orchestrator.correlation import (
                            compute_correlation_matrix,
                        )
                        matrix = compute_correlation_matrix(sorted(held_syms))
                        if matrix is not None:
                            cur_pairs = _snapshot_pairs(matrix)
                            shifts = _correlation_shifts(
                                state.last_correlation_snapshot, cur_pairs,
                                cfg.correlation_drift_threshold,
                            )
                            if shifts:
                                corr_fire = True
                                reason = "correlation-drift"
                                if session:
                                    session.log("correlation_drift_trigger", {
                                        "shifts": shifts,
                                        "threshold": cfg.correlation_drift_threshold,
                                    })
                except Exception as e:  # noqa: BLE001
                    logger.warning("correlation-drift check failed: %s", e)

            regime_fire = False
            if (not fire and not price_fire and not corr_fire
                    and cfg.regime_change_trigger_enabled):
                try:
                    from agentic_investor.agents.macro import analyze_macro
                    macro = analyze_macro()
                    prev_regime = state.last_regime
                    cur_regime = macro.regime
                    if (prev_regime is not None
                            and cur_regime != prev_regime
                            and cur_regime != "unknown"):
                        regime_fire = True
                        reason = "regime-change"
                        if session:
                            session.log("regime_change_trigger", {
                                "from": prev_regime, "to": cur_regime,
                            })
                    state.last_regime = cur_regime
                except Exception as e:  # noqa: BLE001
                    logger.warning("regime-change check failed: %s", e)

            force_fire = False
            if (not fire and not price_fire and not corr_fire and not regime_fire
                    and cfg.force_regen_seconds > 0):
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
            if (not fire and not price_fire and not corr_fire and not regime_fire
                    and not force_fire
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

            if (fire or price_fire or corr_fire or regime_fire
                    or force_fire or stance_fire or interval_due):
                if fire:
                    batch = build_batch(
                        decision_state, now,
                        reaction_price_fetcher=default_reaction_price_fetcher,
                    )
                    # P1 #3: finBERT pre-filter. Score the batch locally and
                    # skip the LLM regen if aggregate sentiment barely moved
                    # from the last fired batch. Fully opt-in via config.
                    # Skip the aggregate check entirely when a hot per-headline
                    # signal already forced the fire — the whole point of the
                    # fast-path is to not let dilution suppress the trigger.
                    if cfg.finbert_prefilter_enabled and not hot_signal_fire:
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
                                            session.log("knob_fired", {
                                                "name": "finbert_prefilter",
                                                "reason": "sentiment-flat",
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
