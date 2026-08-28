"""Micro-batched decision engine for event-driven paper trading.

Turns a stream of NewsEvents into "decision moments" where the LLM is fired
with mixed-age context:

- HOT news:    <2 min old, no market reaction yet (act on headline)
- COOKED news: ~15 min old, has news_reaction_pct (confirm/adjust)
- STALE news:  >60 min old, background only

Batching rules (in order of precedence):
1. Micro-batch window: 60 sec accumulation after first HOT news
2. Cook timer: 15 min after news arrival triggers its "cooked" review
3. Merge: if a cook timer fires within 60s of an already-scheduled decision,
   they collapse into one LLM call
4. Global cap: minimum 30 sec between any two LLM fires
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentic_investor.tools.news_stream import NewsEvent

logger = logging.getLogger(__name__)


HOT_MAX_AGE_SEC = 120       # <2 min = HOT
COOKED_MIN_AGE_SEC = 15 * 60  # >=15 min = COOKED
STALE_MIN_AGE_SEC = 60 * 60   # >=60 min = STALE (background only)


@dataclass
class TaggedNews:
    event: NewsEvent
    age_seconds: float
    tag: str  # "HOT" | "COOKED" | "STALE"
    reaction_pct: float | None = None  # only for COOKED


@dataclass
class DecisionBatch:
    """One decision moment - the news snapshot the LLM sees."""

    fire_at: str
    hot: list[TaggedNews] = field(default_factory=list)
    cooked: list[TaggedNews] = field(default_factory=list)
    stale: list[TaggedNews] = field(default_factory=list)

    def all_tickers(self) -> set[str]:
        return {t.event.ticker for t in (*self.hot, *self.cooked, *self.stale)}

    def summary(self) -> dict:
        return {
            "hot": len(self.hot),
            "cooked": len(self.cooked),
            "stale": len(self.stale),
            "tickers": sorted(self.all_tickers()),
        }


@dataclass
class DecisionState:
    """Per-ticker tracking of when news arrived and when we last processed it."""

    unprocessed: list[NewsEvent] = field(default_factory=list)
    last_fire_at: datetime | None = None
    last_batch_window_started: datetime | None = None


def _tag(event: NewsEvent, now: datetime) -> TaggedNews:
    ts = datetime.fromisoformat(str(event.published_at).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age = (now - ts).total_seconds()
    if age < HOT_MAX_AGE_SEC:
        tag = "HOT"
    elif age >= STALE_MIN_AGE_SEC:
        tag = "STALE"
    elif age >= COOKED_MIN_AGE_SEC:
        tag = "COOKED"
    else:
        tag = "HOT"  # 2-15 min: still fresh enough to be actionable
    return TaggedNews(event=event, age_seconds=age, tag=tag)


def drain_queue(q: queue.Queue) -> list[NewsEvent]:
    """Non-blocking drain of everything currently in the queue."""
    events: list[NewsEvent] = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            return events


def should_fire(
    state: DecisionState,
    now: datetime,
    *,
    batch_window_sec: int = 60,
    cooked_check_sec: int = COOKED_MIN_AGE_SEC,
    min_fire_gap_sec: int = 30,
) -> tuple[bool, str]:
    """Should we fire a decision moment now? Returns (fire?, reason)."""
    if state.last_fire_at is not None:
        gap = (now - state.last_fire_at).total_seconds()
        if gap < min_fire_gap_sec:
            return False, f"global-cap-{gap:.0f}s"

    if not state.unprocessed:
        return False, "no-unprocessed-news"

    # Any COOKED-age event forces a fire (we've held the news long enough).
    for e in state.unprocessed:
        ts = datetime.fromisoformat(str(e.published_at).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if (now - ts).total_seconds() >= cooked_check_sec:
            return True, "cooked-news-ready"

    # Micro-batch: if oldest unprocessed news is >= batch_window_sec old
    # (measured from when we FIRST saw batch news, not from event publish time),
    # fire the batch.
    if state.last_batch_window_started is not None:
        window_age = (now - state.last_batch_window_started).total_seconds()
        if window_age >= batch_window_sec:
            return True, "batch-window-closed"

    return False, "waiting-for-window"


def build_batch(
    state: DecisionState,
    now: datetime,
    *,
    reaction_price_fetcher=None,  # ticker -> (price_now, price_at_news_ts) -> float | None
) -> DecisionBatch:
    """Consume unprocessed news into a tagged batch and reset state."""
    batch = DecisionBatch(fire_at=now.isoformat())
    for e in state.unprocessed:
        tagged = _tag(e, now)
        # Best-effort: attach news_reaction_pct for COOKED events.
        if tagged.tag == "COOKED" and reaction_price_fetcher is not None:
            try:
                tagged.reaction_pct = reaction_price_fetcher(e)
            except Exception as ex:  # noqa: BLE001
                logger.warning("reaction pct failed for %s: %s", e.ticker, ex)
        if tagged.tag == "HOT":
            batch.hot.append(tagged)
        elif tagged.tag == "COOKED":
            batch.cooked.append(tagged)
        else:
            batch.stale.append(tagged)
    state.unprocessed = []
    state.last_batch_window_started = None
    state.last_fire_at = now
    return batch


def render_batch_context(batch: DecisionBatch) -> str:
    """Render a DecisionBatch into a text block for the allocator prompt.

    Format is deliberately compact so it drops into the existing prompt without
    ballooning tokens. Each line names the tag, ticker, headline snippet, age,
    and news_reaction_pct where available.
    """
    lines: list[str] = []
    for group_name, group in (("HOT", batch.hot), ("COOKED", batch.cooked), ("STALE", batch.stale)):
        for item in group:
            e = item.event
            headline = (e.headline or "").strip()[:120]
            age_min = int(item.age_seconds // 60)
            reaction = (
                f", reaction={item.reaction_pct:+.2f}%"
                if item.reaction_pct is not None
                else ""
            )
            lines.append(
                f"- [{group_name}] {e.ticker}  age={age_min}m{reaction}: {headline}"
            )
    return "\n".join(lines) if lines else ""


def default_reaction_price_fetcher(event: NewsEvent) -> float | None:
    """Real news_reaction_pct: intraday price change from published_at to now.

    Uses 1-minute yfinance bars (available for the last 7 days). Returns None
    when we can't compute a reaction - the LLM prompt handles missing values
    gracefully (COOKED entries without reaction data just omit the field).
    """
    try:
        import pandas as pd

        from agentic_investor.tools.market import fetch_ohlcv

        ts = datetime.fromisoformat(str(event.published_at).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        df = fetch_ohlcv(event.ticker, period="7d", interval="1m")
        if df.empty or len(df) < 2:
            return None
        # yfinance 1m bars are timezone-aware; normalize both sides.
        if df.index.tz is None:
            idx = df.index.tz_localize(UTC)
        else:
            idx = df.index.tz_convert(UTC)
        ts_pd = pd.Timestamp(ts)
        mask = idx >= ts_pd
        if not mask.any():
            return None
        price_at_news = float(df["Close"].iloc[mask.argmax()])
        current = float(df["Close"].iloc[-1])
        if price_at_news <= 0:
            return None
        return round((current / price_at_news - 1) * 100, 2)
    except Exception as e:  # noqa: BLE001 - reaction is best-effort
        logger.warning("reaction pct failed for %s: %s", event.ticker, e)
        return None


def ingest(state: DecisionState, events: list[NewsEvent], now: datetime) -> None:
    """Accept new events into the state; open batch window if first in a while."""
    if not events:
        return
    for e in events:
        state.unprocessed.append(e)
    if state.last_batch_window_started is None:
        state.last_batch_window_started = now
