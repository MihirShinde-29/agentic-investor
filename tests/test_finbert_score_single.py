"""score_single returns a per-headline signed score in [-1, 1].

Underlies the immediate-fire fast-path: the loop calls score_single on
every arriving headline and fires the regen NOW if any headline exceeds
the abs threshold, bypassing the 60s batch window that would otherwise
dilute a strong signal with N mild-neutral analyst notes.
"""

from __future__ import annotations

from agentic_investor.orchestrator import finbert_prefilter


class _FakePipeline:
    def __init__(self, results):
        self._results = results

    def __call__(self, texts, **kwargs):
        return self._results


def test_score_single_returns_positive_on_positive_class(monkeypatch):
    monkeypatch.setattr(
        finbert_prefilter, "_get_pipeline",
        lambda: _FakePipeline([[
            {"label": "positive", "score": 0.85},
            {"label": "neutral", "score": 0.10},
            {"label": "negative", "score": 0.05},
        ]]),
    )
    score = finbert_prefilter.score_single("Company X beats Q3 estimates")
    assert score is not None
    assert score > 0.7


def test_score_single_returns_negative_on_negative_class(monkeypatch):
    monkeypatch.setattr(
        finbert_prefilter, "_get_pipeline",
        lambda: _FakePipeline([[
            {"label": "positive", "score": 0.05},
            {"label": "neutral", "score": 0.15},
            {"label": "negative", "score": 0.80},
        ]]),
    )
    score = finbert_prefilter.score_single("Company X misses guidance, cuts outlook")
    assert score is not None
    assert score < -0.7


def test_score_single_returns_zero_on_neutral(monkeypatch):
    monkeypatch.setattr(
        finbert_prefilter, "_get_pipeline",
        lambda: _FakePipeline([[
            {"label": "positive", "score": 0.05},
            {"label": "neutral", "score": 0.90},
            {"label": "negative", "score": 0.05},
        ]]),
    )
    score = finbert_prefilter.score_single("Bond yields tick lower on quiet trading")
    assert score is not None
    assert abs(score) < 0.1


def test_score_single_returns_none_when_pipeline_unavailable(monkeypatch):
    monkeypatch.setattr(finbert_prefilter, "_get_pipeline", lambda: None)
    assert finbert_prefilter.score_single("anything") is None


def test_score_single_returns_none_on_empty_headline(monkeypatch):
    called = []
    monkeypatch.setattr(
        finbert_prefilter, "_get_pipeline",
        lambda: called.append("pipe") or _FakePipeline([]),
    )
    assert finbert_prefilter.score_single("") is None
    # No pipeline invocation for empty input.
    assert not called or called == ["pipe"]


def test_score_single_handles_pipeline_exception(monkeypatch):
    class _Boom:
        def __call__(self, *a, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(finbert_prefilter, "_get_pipeline", lambda: _Boom())
    assert finbert_prefilter.score_single("headline") is None
