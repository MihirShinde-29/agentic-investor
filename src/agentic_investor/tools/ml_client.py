"""HTTP client for the shared paper-ml-service.

Arms use this when `AGENTIC_ML_SERVICE_URL` is set. When it isn't (tests,
standalone paper-loop runs, or the service is unreachable) callers get
None back and fall back to loading models locally in their own process.

Keeps the request/response schemas in sync with `services/ml_service.py`
by construction: both import the same pydantic models is tempting, but
importing FastAPI into arms just to read a schema doubles their import
cost. Instead we use plain httpx + typed helpers here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from threading import Lock

import httpx

logger = logging.getLogger(__name__)


_URL_ENV = "AGENTIC_ML_SERVICE_URL"
_REQUEST_TIMEOUT_SEC = 10.0

# Once a call fails, log at INFO once (not every subsequent call) so
# arms don't spam the log while the service is down.
_fallback_lock = Lock()
_fallback_notified = False


def service_url() -> str | None:
    """Return the configured URL or None."""
    url = os.environ.get(_URL_ENV, "").strip()
    return url or None


def _notify_fallback_once(reason: str) -> None:
    global _fallback_notified
    with _fallback_lock:
        if _fallback_notified:
            return
        _fallback_notified = True
    logger.info("ml-service unavailable, falling back to local models: %s", reason)


_client_lock = Lock()
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(timeout=_REQUEST_TIMEOUT_SEC)
        return _client


def finbert_scores(headlines: Sequence[str]) -> list[list[dict]] | None:
    """POST /finbert. Return per-headline list of {label, score} dicts,
    or None on any failure. Callers do their own aggregation (see
    orchestrator/finbert_prefilter.py) so this wrapper stays dumb.
    """
    url = service_url()
    if not url:
        return None
    if not headlines:
        return []
    try:
        r = _get_client().post(
            f"{url.rstrip('/')}/finbert",
            json={"headlines": list(headlines)},
        )
        r.raise_for_status()
    except (httpx.HTTPError, httpx.RemoteProtocolError) as e:
        _notify_fallback_once(f"finbert: {type(e).__name__}: {e}")
        return None
    payload = r.json()
    return [
        [
            {"label": e["label"], "score": e["score"]}
            for e in row
        ]
        for row in payload.get("results", [])
    ]


def embed_texts(texts: Sequence[str]) -> list[list[float]] | None:
    """POST /embed. Return per-text embedding vector, or None on failure."""
    url = service_url()
    if not url:
        return None
    if not texts:
        return []
    try:
        r = _get_client().post(
            f"{url.rstrip('/')}/embed",
            json={"texts": list(texts)},
        )
        r.raise_for_status()
    except (httpx.HTTPError, httpx.RemoteProtocolError) as e:
        _notify_fallback_once(f"embed: {type(e).__name__}: {e}")
        return None
    return list(r.json().get("embeddings", []))


def wait_for_healthy(url: str, timeout_sec: float = 60.0) -> bool:
    """Block up to `timeout_sec` waiting for GET /health to return 200
    with both models ready. Supervisor calls this before spawning arms
    so arm boots don't race against an unwarm service.

    Returns True if ready, False on timeout.
    """
    import time
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = httpx.get(f"{url.rstrip('/')}/health", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                if data.get("finbert_ready") and data.get("embed_ready"):
                    return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def _reset_state_for_tests() -> None:
    """Tests re-arm the once-per-process fallback warning."""
    global _fallback_notified, _client
    with _fallback_lock:
        _fallback_notified = False
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # noqa: BLE001
                pass
            _client = None
