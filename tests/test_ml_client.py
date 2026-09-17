"""Tests for the ml-service HTTP client.

Uses httpx.MockTransport so no real service subprocess is needed.
Verifies the fallback contract: any HTTP failure or unset env var
returns None so callers fall back to local model loading.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx

from agentic_investor.tools import ml_client


def _install_mock_transport(monkeypatch, handler):
    """Rebuild ml_client's shared httpx.Client so it uses the mock."""
    ml_client._reset_state_for_tests()
    fake = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(ml_client, "_client", fake)


def test_no_url_returns_none_immediately(monkeypatch):
    monkeypatch.delenv(ml_client._URL_ENV, raising=False)
    ml_client._reset_state_for_tests()
    assert ml_client.finbert_scores(["hi"]) is None
    assert ml_client.embed_texts(["hi"]) is None


def test_empty_input_shortcircuits(monkeypatch):
    monkeypatch.setenv(ml_client._URL_ENV, "http://127.0.0.1:9999")

    def _boom(request):
        raise AssertionError("should not have been called for empty input")

    _install_mock_transport(monkeypatch, _boom)
    assert ml_client.finbert_scores([]) == []
    assert ml_client.embed_texts([]) == []


def test_finbert_happy_path(monkeypatch):
    monkeypatch.setenv(ml_client._URL_ENV, "http://127.0.0.1:9999")

    def _handler(request):
        assert request.url.path == "/finbert"
        return httpx.Response(200, json={
            "results": [
                [{"label": "positive", "score": 0.9},
                 {"label": "negative", "score": 0.05},
                 {"label": "neutral", "score": 0.05}],
            ],
        })

    _install_mock_transport(monkeypatch, _handler)
    out = ml_client.finbert_scores(["good news"])
    assert out is not None
    assert len(out) == 1
    labels = [e["label"] for e in out[0]]
    assert "positive" in labels


def test_embed_happy_path(monkeypatch):
    monkeypatch.setenv(ml_client._URL_ENV, "http://127.0.0.1:9999")

    def _handler(request):
        assert request.url.path == "/embed"
        return httpx.Response(200, json={
            "embeddings": [[0.1, 0.2, 0.3]],
        })

    _install_mock_transport(monkeypatch, _handler)
    out = ml_client.embed_texts(["hello"])
    assert out == [[0.1, 0.2, 0.3]]


def test_http_500_returns_none_and_notifies_once(monkeypatch, caplog):
    monkeypatch.setenv(ml_client._URL_ENV, "http://127.0.0.1:9999")

    def _handler(request):
        return httpx.Response(500, json={"detail": "boom"})

    _install_mock_transport(monkeypatch, _handler)
    import logging
    with caplog.at_level(logging.INFO, logger=ml_client.__name__):
        assert ml_client.embed_texts(["x"]) is None
        # Second call: no additional log line (once-per-process notification).
        assert ml_client.embed_texts(["y"]) is None
    fallback_lines = [
        r for r in caplog.records
        if "ml-service unavailable" in r.getMessage()
    ]
    assert len(fallback_lines) == 1, (
        f"expected exactly one fallback notification, got {len(fallback_lines)}"
    )


def test_connection_refused_returns_none(monkeypatch):
    monkeypatch.setenv(ml_client._URL_ENV, "http://127.0.0.1:9999")

    def _handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _install_mock_transport(monkeypatch, _handler)
    assert ml_client.embed_texts(["x"]) is None
    assert ml_client.finbert_scores(["x"]) is None


def test_url_trailing_slash_tolerated(monkeypatch):
    monkeypatch.setenv(ml_client._URL_ENV, "http://127.0.0.1:9999/")
    seen_paths = []

    def _handler(request):
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"embeddings": [[0.0]]})

    _install_mock_transport(monkeypatch, _handler)
    ml_client.embed_texts(["x"])
    # Path should be /embed exactly, not //embed
    assert seen_paths == ["/embed"]


def test_service_url_helper(monkeypatch):
    monkeypatch.delenv(ml_client._URL_ENV, raising=False)
    assert ml_client.service_url() is None
    monkeypatch.setenv(ml_client._URL_ENV, "  http://x:1  ")
    assert ml_client.service_url() == "http://x:1"


def test_wait_for_healthy_success(monkeypatch):
    """wait_for_healthy returns True once /health reports both models ready."""
    call_count = {"n": 0}

    def _health_handler(request):
        call_count["n"] += 1
        ready = call_count["n"] >= 2
        return httpx.Response(200, json={
            "status": "ok",
            "finbert_ready": ready,
            "embed_ready": ready,
        })

    with patch(
        "httpx.get",
        side_effect=lambda url, timeout=None: _health_handler(
            httpx.Request("GET", url),
        ),
    ):
        assert ml_client.wait_for_healthy(
            "http://127.0.0.1:9999", timeout_sec=5.0,
        )


def test_wait_for_healthy_times_out(monkeypatch):
    """Returns False if /health never reports both models ready."""

    def _never_ready(request):
        return httpx.Response(200, json={
            "status": "ok",
            "finbert_ready": False,
            "embed_ready": False,
        })

    with patch(
        "httpx.get",
        side_effect=lambda url, timeout=None: _never_ready(
            httpx.Request("GET", url),
        ),
    ):
        assert not ml_client.wait_for_healthy(
            "http://127.0.0.1:9999", timeout_sec=1.5,
        )
