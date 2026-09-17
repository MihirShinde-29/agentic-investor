"""Tests for the shared ml-service subprocess.

Uses FastAPI's TestClient with eager_load=False so tests don't pull the
real transformer weights. Model-loading is stubbed to a small fake so we
can exercise the endpoint contracts (shape, error paths, health).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentic_investor.services import ml_service


class _FakePipeline:
    """Stand-in for transformers.pipeline("text-classification", ...).

    Returns a list-of-lists that matches the top_k=None shape:
    one list per input, each with the three finbert classes.
    """

    def __call__(self, headlines, *, truncation=True, padding=None, max_length=128):
        return [
            [
                {"label": "positive", "score": 0.6},
                {"label": "negative", "score": 0.2},
                {"label": "neutral", "score": 0.2},
            ]
            for _ in headlines
        ]


class _FakeEmbedModel:
    def encode(self, texts, *, show_progress_bar=False, normalize_embeddings=True):
        import numpy as np
        return np.array([[float(i), 0.5, -0.5] for i, _ in enumerate(texts)])


def _install_fakes(monkeypatch):
    fake_finbert = _FakePipeline()
    fake_embed = _FakeEmbedModel()
    monkeypatch.setattr(ml_service, "_finbert", fake_finbert)
    monkeypatch.setattr(ml_service, "_embed_model", fake_embed)
    monkeypatch.setattr(ml_service, "_finbert_load_error", None)
    monkeypatch.setattr(ml_service, "_embed_load_error", None)
    monkeypatch.setattr(ml_service, "_get_finbert", lambda: fake_finbert)
    monkeypatch.setattr(ml_service, "_get_embed_model", lambda: fake_embed)


def _client() -> TestClient:
    return TestClient(ml_service.create_app(eager_load=False))


def test_health_reports_ready_when_models_loaded(monkeypatch):
    _install_fakes(monkeypatch)
    with _client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["finbert_ready"] is True
    assert body["embed_ready"] is True


def test_health_reports_not_ready_when_unloaded(monkeypatch):
    """Both models absent from module globals -> health reports False.
    The service still returns 200 (it's up) but the flags let the
    supervisor decide whether to hold arm launches back.
    """
    monkeypatch.setattr(ml_service, "_finbert", None)
    monkeypatch.setattr(ml_service, "_embed_model", None)
    monkeypatch.setattr(ml_service, "_get_finbert", lambda: None)
    monkeypatch.setattr(ml_service, "_get_embed_model", lambda: None)
    with _client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["finbert_ready"] is False
    assert body["embed_ready"] is False


def test_finbert_returns_per_headline_scores(monkeypatch):
    _install_fakes(monkeypatch)
    with _client() as c:
        r = c.post("/finbert", json={
            "headlines": ["Apple beats earnings", "Fed hikes rates"],
        })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    labels = {e["label"] for e in body["results"][0]}
    assert labels == {"positive", "negative", "neutral"}


def test_finbert_empty_input_returns_empty_list(monkeypatch):
    _install_fakes(monkeypatch)
    with _client() as c:
        r = c.post("/finbert", json={"headlines": []})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_finbert_503_when_unloaded(monkeypatch):
    monkeypatch.setattr(ml_service, "_finbert", None)
    monkeypatch.setattr(ml_service, "_finbert_load_error", "torch missing")
    monkeypatch.setattr(ml_service, "_get_finbert", lambda: None)
    with _client() as c:
        r = c.post("/finbert", json={"headlines": ["x"]})
    assert r.status_code == 503
    assert "torch missing" in r.json()["detail"]


def test_embed_returns_vector_per_text(monkeypatch):
    _install_fakes(monkeypatch)
    with _client() as c:
        r = c.post("/embed", json={"texts": ["hello", "world"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["embeddings"]) == 2
    assert body["embeddings"][0] == [0.0, 0.5, -0.5]
    assert body["embeddings"][1] == [1.0, 0.5, -0.5]


def test_embed_empty_input_returns_empty_list(monkeypatch):
    _install_fakes(monkeypatch)
    with _client() as c:
        r = c.post("/embed", json={"texts": []})
    assert r.status_code == 200
    assert r.json()["embeddings"] == []


def test_embed_503_when_unloaded(monkeypatch):
    monkeypatch.setattr(ml_service, "_embed_model", None)
    monkeypatch.setattr(ml_service, "_embed_load_error", "hf blocked")
    monkeypatch.setattr(ml_service, "_get_embed_model", lambda: None)
    with _client() as c:
        r = c.post("/embed", json={"texts": ["x"]})
    assert r.status_code == 503
    assert "hf blocked" in r.json()["detail"]
