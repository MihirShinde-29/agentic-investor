"""finBERT sentiment pre-filter (P1 #3).

Runs a local finBERT model over the headlines in a news batch and returns an
aggregate sentiment score in [-1, 1]. The event loop uses this to skip
regens when the aggregate sentiment barely moved from the last regen -
cheap heuristic that avoids firing the (expensive) LLM allocator on quiet
news that isn't going to change conviction.

The model is loaded lazily on first use and cached module-level so repeated
calls don't repay the ~440MB warm-up cost. Fully opt-in: if the caller
doesn't enable it, no download happens.

Model: ProsusAI/finbert — 3 classes (positive/negative/neutral).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from agentic_investor.tools.news_stream import NewsEvent

logger = logging.getLogger(__name__)


_pipeline_lock = threading.Lock()
_pipeline: object | None = None
_pipeline_failed = False


def _get_pipeline():
    """Lazy-load the finBERT pipeline. Returns None on any failure."""
    global _pipeline, _pipeline_failed
    if _pipeline is not None or _pipeline_failed:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None or _pipeline_failed:
            return _pipeline
        try:
            from transformers import pipeline

            _pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
            )
            logger.info("finBERT pipeline loaded (ProsusAI/finbert)")
        except Exception as e:  # noqa: BLE001
            logger.warning("finBERT unavailable, prefilter disabled: %s", e)
            _pipeline_failed = True
            _pipeline = None
    return _pipeline


@dataclass
class SentimentScore:
    """Aggregate sentiment across a batch. score in [-1, 1]."""

    n_headlines: int
    score: float
    n_positive: int
    n_negative: int
    n_neutral: int


def _label_to_signed(label: str) -> float:
    lower = label.lower()
    if lower.startswith("pos"):
        return 1.0
    if lower.startswith("neg"):
        return -1.0
    return 0.0


def score_headlines(headlines: list[str]) -> SentimentScore | None:
    """Score a list of headlines. Returns None if finBERT isn't available."""
    pipe = _get_pipeline()
    if pipe is None or not headlines:
        return None
    try:
        results = pipe(headlines, truncation=True, max_length=128)  # type: ignore[misc]
    except Exception as e:  # noqa: BLE001
        logger.warning("finBERT scoring failed: %s", e)
        return None

    total = 0.0
    n_pos = n_neg = n_neu = 0
    for row in results:
        # pipeline with top_k=None returns list-of-list-of-dicts.
        if isinstance(row, list):
            entries = row
        else:
            entries = [row]
        # weighted score = sum(prob * signed_label) over classes
        weighted = 0.0
        top_signed = 0
        top_score = -1.0
        for entry in entries:
            label = str(entry.get("label", ""))
            score = float(entry.get("score", 0.0))
            weighted += score * _label_to_signed(label)
            if score > top_score:
                top_score = score
                top_signed = _label_to_signed(label)
        total += weighted
        if top_signed > 0:
            n_pos += 1
        elif top_signed < 0:
            n_neg += 1
        else:
            n_neu += 1

    return SentimentScore(
        n_headlines=len(headlines),
        score=total / len(headlines),
        n_positive=n_pos,
        n_negative=n_neg,
        n_neutral=n_neu,
    )


def score_single(headline: str) -> float | None:
    """Return signed sentiment for one headline in [-1, 1], or None if the
    pipeline isn't available. Cheap enough (~50ms cpu) to run inline as
    news arrives so hot signals can force-fire the LLM without waiting
    for the batch window to close.
    """
    pipe = _get_pipeline()
    if pipe is None or not headline:
        return None
    try:
        row = pipe([headline], truncation=True, max_length=128)[0]  # type: ignore[misc]
    except Exception as e:  # noqa: BLE001
        logger.warning("finBERT single-headline scoring failed: %s", e)
        return None
    entries = row if isinstance(row, list) else [row]
    weighted = 0.0
    for entry in entries:
        weighted += float(entry.get("score", 0.0)) * _label_to_signed(
            str(entry.get("label", ""))
        )
    return weighted


def score_events(events: list[NewsEvent]) -> SentimentScore | None:
    """Score a batch of NewsEvents (uses headline + summary snippet)."""
    if not events:
        return None
    lines = []
    for e in events:
        headline = str(getattr(e, "headline", "")).strip()
        summary = str(getattr(e, "summary", "")).strip()[:200]
        combined = f"{headline}. {summary}" if summary else headline
        if combined:
            lines.append(combined)
    return score_headlines(lines)
