"""Tests for the finBERT sentiment pre-filter.

We don't load the actual model in tests (440MB download + slow init).
Instead we monkey-patch the lazy pipeline getter to a deterministic stub.
"""

from __future__ import annotations

import pytest

from agentic_investor.orchestrator import finbert_prefilter as fp


class _StubPipeline:
    """Returns per-headline top-k class scores like the real transformers pipeline."""

    def __init__(self, mapping: dict[str, str]):
        # mapping: substring -> "positive" | "negative" | "neutral"
        self.mapping = mapping

    def __call__(self, headlines, truncation=True, max_length=128):
        out = []
        for h in headlines:
            label = "neutral"
            for k, v in self.mapping.items():
                if k.lower() in h.lower():
                    label = v
                    break
            # Emulate top_k=None: list of {label, score} per class.
            scores = {"positive": 0.05, "negative": 0.05, "neutral": 0.05}
            scores[label] = 0.9
            out.append(
                [
                    {"label": lbl, "score": s} for lbl, s in scores.items()
                ]
            )
        return out


@pytest.fixture(autouse=True)
def _reset_pipeline(monkeypatch):
    monkeypatch.setattr(fp, "_pipeline", None)
    monkeypatch.setattr(fp, "_pipeline_failed", False)
    yield


def test_score_headlines_returns_none_when_pipeline_unavailable(monkeypatch):
    monkeypatch.setattr(fp, "_get_pipeline", lambda: None)
    assert fp.score_headlines(["anything"]) is None


def test_score_headlines_positive_bias(monkeypatch):
    monkeypatch.setattr(
        fp, "_get_pipeline",
        lambda: _StubPipeline({"beats": "positive", "misses": "negative"}),
    )
    s = fp.score_headlines([
        "AAPL beats earnings estimates",
        "NVDA beats revenue guidance",
        "TSLA holds steady",  # -> neutral
    ])
    assert s is not None
    assert s.n_headlines == 3
    assert s.n_positive == 2
    assert s.n_negative == 0
    assert s.n_neutral == 1
    assert s.score > 0


def test_score_headlines_negative_bias(monkeypatch):
    monkeypatch.setattr(
        fp, "_get_pipeline",
        lambda: _StubPipeline({"misses": "negative"}),
    )
    s = fp.score_headlines(["AAPL misses q4 estimates", "NVDA misses guide"])
    assert s is not None
    assert s.n_negative == 2
    assert s.score < 0


def test_score_headlines_empty_input():
    assert fp.score_headlines([]) is None


def test_score_events_uses_headline_plus_summary(monkeypatch):
    from agentic_investor.tools.news_stream import NewsEvent

    monkeypatch.setattr(
        fp, "_get_pipeline",
        lambda: _StubPipeline({"beats": "positive"}),
    )
    ev = NewsEvent(
        ticker="AAPL", headline="Apple beats revenue",
        summary="Strong iPhone sales drove the beat.",
        published_at="2026-08-31T10:00:00Z",
        received_at="2026-08-31T10:00:01Z",
    )
    s = fp.score_events([ev])
    assert s is not None
    assert s.n_headlines == 1
    assert s.n_positive == 1
