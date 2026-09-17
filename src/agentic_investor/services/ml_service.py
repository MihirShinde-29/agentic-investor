"""Shared inference service: finBERT sentiment + sentence-transformer embeddings.

One subprocess spawned by paper-experiment, siblings to paper-news-bus /
paper-price-bus. Arms consume it over localhost HTTP so each arm doesn't
load its own ~530 MB copy of the two models (3 arms x 530 MB = 1.6 GB
duplicated on a machine where physical RAM is the binding constraint).

Motivated by 2026-09-17: with three model-heavy arm processes plus
Chrome plus Windows plus everything else, the box tipped into
memory-pressure reap. Consolidating both models into one process saves
~1 GB across the fleet and cuts per-arm boot from ~20 s to ~5 s.

Endpoints:
  POST /finbert    -> raw pipeline output per headline (label + prob)
  POST /embed      -> raw sentence-transformer output per text (list[float])
  GET  /health     -> {status, finbert_ready, embed_ready}

Arms fall back to loading the models locally if the service is
unreachable or set env var is absent - see tools/ml_client.py.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- models: lazy-load with a lock -----------------------------------------
#
# Startup does load these eagerly (see `create_app` below) so /health can
# report ready state truthfully. The lazy path also handles the case
# where a caller hits an endpoint before startup finishes.

_finbert_lock = threading.Lock()
_finbert = None
_finbert_load_error: str | None = None

_embed_lock = threading.Lock()
_embed_model = None
_embed_load_error: str | None = None


def _get_finbert():
    global _finbert, _finbert_load_error
    if _finbert is not None or _finbert_load_error is not None:
        return _finbert
    with _finbert_lock:
        if _finbert is not None or _finbert_load_error is not None:
            return _finbert
        try:
            from transformers import pipeline
            _finbert = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
            )
            logger.info("finBERT pipeline loaded (ProsusAI/finbert)")
        except Exception as e:  # noqa: BLE001
            _finbert_load_error = f"{type(e).__name__}: {e}"
            logger.warning("finBERT unavailable: %s", _finbert_load_error)
    return _finbert


def _get_embed_model():
    global _embed_model, _embed_load_error
    if _embed_model is not None or _embed_load_error is not None:
        return _embed_model
    with _embed_lock:
        if _embed_model is not None or _embed_load_error is not None:
            return _embed_model
        try:
            from sentence_transformers import SentenceTransformer

            from agentic_investor.config import get_settings
            s = get_settings()
            _embed_model = SentenceTransformer(s.embedding_model)
            logger.info("embed model loaded (%s)", s.embedding_model)
        except Exception as e:  # noqa: BLE001
            _embed_load_error = f"{type(e).__name__}: {e}"
            logger.warning("embed model unavailable: %s", _embed_load_error)
    return _embed_model


# --- request/response schemas ----------------------------------------------

class FinbertRequest(BaseModel):
    headlines: list[str]


class FinbertEntry(BaseModel):
    label: str
    score: float


class FinbertResponse(BaseModel):
    # results[i] is the list of {label, score} for headlines[i]. Shape
    # mirrors what transformers.pipeline(top_k=None) returns per input.
    results: list[list[FinbertEntry]]


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class HealthResponse(BaseModel):
    status: str
    finbert_ready: bool
    embed_ready: bool
    finbert_error: str | None = None
    embed_error: str | None = None


# --- app -------------------------------------------------------------------

def create_app(*, eager_load: bool = True) -> FastAPI:
    """Return a FastAPI app. eager_load=True triggers both model loads
    at startup so /health is authoritative from the first request. Tests
    pass False to skip the slow load.
    """
    app = FastAPI(title="agentic-investor ml-service")

    @app.on_event("startup")
    def _startup() -> None:  # pragma: no cover - runtime hook
        if eager_load:
            _get_finbert()
            _get_embed_model()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            finbert_ready=_finbert is not None,
            embed_ready=_embed_model is not None,
            finbert_error=_finbert_load_error,
            embed_error=_embed_load_error,
        )

    @app.post("/finbert", response_model=FinbertResponse)
    def finbert(req: FinbertRequest) -> FinbertResponse:
        pipe = _get_finbert()
        if pipe is None:
            raise HTTPException(
                status_code=503,
                detail=f"finbert unavailable: {_finbert_load_error}",
            )
        if not req.headlines:
            return FinbertResponse(results=[])
        raw: list[Any] = pipe(req.headlines, truncation=True, max_length=128)
        # transformers.pipeline with top_k=None can return list-per-input
        # OR flat list depending on version; normalize.
        results: list[list[FinbertEntry]] = []
        for row in raw:
            entries = row if isinstance(row, list) else [row]
            results.append([
                FinbertEntry(
                    label=str(e.get("label", "")),
                    score=float(e.get("score", 0.0)),
                )
                for e in entries
            ])
        return FinbertResponse(results=results)

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        model = _get_embed_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"embed unavailable: {_embed_load_error}",
            )
        if not req.texts:
            return EmbedResponse(embeddings=[])
        vecs = model.encode(
            req.texts, show_progress_bar=False, normalize_embeddings=True,
        )
        return EmbedResponse(embeddings=vecs.tolist())

    return app


def run_ml_service(host: str = "127.0.0.1", port: int = 8765) -> int:
    """CLI entry: bind + serve until SIGINT/SIGTERM.

    Bind on 127.0.0.1 explicitly (not 0.0.0.0) so we don't trigger
    Windows firewall prompts on a fresh box.
    """
    import signal

    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    app = create_app(eager_load=True)
    config = uvicorn.Config(
        app, host=host, port=port,
        log_level="warning",  # uvicorn's INFO is per-request, noisy
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _shutdown(*_a):
        server.should_exit = True

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, AttributeError):
        pass  # Windows SIGTERM not always settable

    server.run()
    return 0
