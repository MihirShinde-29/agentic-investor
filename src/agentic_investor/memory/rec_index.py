"""Chroma index of Recommendations for retrieval-augmented allocation (M17).

One collection `recommendations`, shared across arms. Each doc carries a
`source` tag (`"historical"` for the seed corpus, `f"arm_{id}"` for live
runs) so retrieval can filter with `where={"source": {"$in": [...]}}` to
keep A/B independence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path

from agentic_investor.orchestrator.state import Recommendation

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "recommendations"


def _default_collection():
    from agentic_investor.tools.news import _get_client

    return _get_client().get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _default_embed(texts: list[str]) -> list[list[float]]:
    from agentic_investor.tools.news import _embed_text

    return _embed_text(texts)


def embed_text_for_rec(rec: Recommendation) -> str:
    """Compact text for embedding: portfolio rationale + top-3 position rationales."""
    parts: list[str] = []
    portfolio = (rec.allocation.portfolio_rationale or "").strip()
    if portfolio:
        parts.append(portfolio)
    top = sorted(
        rec.allocation.positions,
        key=lambda p: p.weight_pct,
        reverse=True,
    )[:3]
    for p in top:
        rationale = (p.rationale or "").strip()
        if rationale:
            parts.append(
                f"{p.ticker} ({p.weight_pct:.1f}%, conf={p.confidence:.2f}): "
                f"{rationale}"
            )
    return "\n".join(parts).strip()


def metadata_for_rec(
    rec: Recommendation,
    rec_id: int,
    created_at: str,
    source: str,
) -> dict:
    """Scalar-only metadata for Chroma (lists get comma-joined)."""
    positions = rec.allocation.positions
    tickers = sorted({p.ticker.upper() for p in positions})
    avg_conf = (
        sum(p.confidence for p in positions) / len(positions)
        if positions else 0.0
    )
    return {
        "rec_id": int(rec_id),
        "created_at": created_at,
        "source": source,
        "tickers": ",".join(tickers),
        "n_positions": len(positions),
        "avg_confidence": round(avg_conf, 3),
        "cash_pct": round(rec.allocation.cash_pct, 2),
        "risk": str(getattr(rec.request, "risk", "moderate") or "moderate"),
    }


def _doc_id(source: str, rec_id: int) -> str:
    return f"rec:{source}:{rec_id}"


def upsert_rec(
    rec: Recommendation,
    rec_id: int,
    created_at: str,
    source: str,
    *,
    collection=None,
    embedder: Callable[[list[str]], list[list[float]]] = _default_embed,
) -> bool:
    """Index a single rec. Returns True if written, False if skipped (empty text)."""
    text = embed_text_for_rec(rec)
    if not text:
        return False
    coll = collection if collection is not None else _default_collection()
    coll.upsert(
        ids=[_doc_id(source, rec_id)],
        embeddings=embedder([text]),
        documents=[text],
        metadatas=[metadata_for_rec(rec, rec_id, created_at, source)],
    )
    return True


def _text_from_payload(payload: dict) -> str:
    """Build the embed string directly from the JSON blob.

    Bypasses Recommendation.model_validate so schema-drifted historical rows
    (e.g. confidence=null before the field became required) still index.
    """
    alloc = payload.get("allocation") or {}
    parts: list[str] = []
    portfolio = (alloc.get("portfolio_rationale") or "").strip()
    if portfolio:
        parts.append(portfolio)
    positions = list(alloc.get("positions") or [])
    positions.sort(key=lambda p: float(p.get("weight_pct") or 0.0), reverse=True)
    for p in positions[:3]:
        rationale = (p.get("rationale") or "").strip()
        if not rationale:
            continue
        ticker = str(p.get("ticker") or "?")
        weight = float(p.get("weight_pct") or 0.0)
        conf = float(p.get("confidence") or 0.5)
        parts.append(f"{ticker} ({weight:.1f}%, conf={conf:.2f}): {rationale}")
    return "\n".join(parts).strip()


def _meta_from_payload(payload: dict, rec_id: int, created_at: str, source: str) -> dict:
    alloc = payload.get("allocation") or {}
    req = payload.get("request") or {}
    positions = list(alloc.get("positions") or [])
    tickers = sorted({str(p.get("ticker") or "").upper() for p in positions if p.get("ticker")})
    confs = [float(p.get("confidence") or 0.5) for p in positions]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return {
        "rec_id": int(rec_id),
        "created_at": created_at,
        "source": source,
        "tickers": ",".join(tickers),
        "n_positions": len(positions),
        "avg_confidence": round(avg_conf, 3),
        "cash_pct": round(float(alloc.get("cash_pct") or 0.0), 2),
        "risk": str(req.get("risk") or "moderate"),
    }


def index_historical(
    db_url: str | None = None,
    *,
    collection=None,
    embedder: Callable[[list[str]], list[list[float]]] = _default_embed,
    batch_size: int = 32,
) -> int:
    """Bulk-index every rec from the given DB (default: settings.database_url).

    Reads raw JSON blobs so schema drift in old recs doesn't block the index.
    Idempotent via `_doc_id`; re-running overwrites in place.
    """
    coll = collection if collection is not None else _default_collection()
    rows = _load_rec_blobs(db_url)
    logger.info(
        "indexing %d recommendations from %s",
        len(rows), db_url or "default DB",
    )
    n_indexed = 0
    n_skipped = 0
    batch: list[tuple[str, str, dict]] = []
    for rec_id, created_at, payload_json in rows:
        try:
            payload = json.loads(payload_json)
            text = _text_from_payload(payload)
            meta = _meta_from_payload(payload, rec_id, created_at, "historical")
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping rec %d: %s", rec_id, e)
            n_skipped += 1
            continue
        if not text:
            n_skipped += 1
            continue
        batch.append((_doc_id("historical", rec_id), text, meta))
        if len(batch) >= batch_size:
            _flush(coll, batch, embedder)
            n_indexed += len(batch)
            batch = []
    if batch:
        _flush(coll, batch, embedder)
        n_indexed += len(batch)
    logger.info(
        "indexed %d recs (skipped %d) into Chroma collection %r",
        n_indexed, n_skipped, _COLLECTION_NAME,
    )
    return n_indexed


def _load_rec_blobs(db_url: str | None) -> list[tuple[int, str, str]]:
    from agentic_investor.config import get_settings

    url = db_url or get_settings().database_url
    if not url.startswith("sqlite:///"):
        raise ValueError(f"only sqlite:/// URLs supported (got {url!r})")
    path = Path(url.removeprefix("sqlite:///"))
    with sqlite3.connect(str(path)) as conn:
        return conn.execute(
            "SELECT id, created_at, payload_json FROM recommendations ORDER BY id",
        ).fetchall()


def index_arm_rec(
    rec: Recommendation,
    rec_id: int,
    *,
    arm_id: str | None = None,
    collection=None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> bool:
    """Index a fresh live rec under source=f"arm_{id}".

    Reads AGENTIC_ARM_ID env when arm_id is not passed (solo paper-loop
    runs get arm_id="solo" so their memory is isolated from any A/B
    experiment). Honors AGENTIC_MEMORY_RAG kill-switch. Never raises -
    failure returns False and the loop moves on.
    """
    import os
    from datetime import UTC, datetime

    if os.environ.get("AGENTIC_MEMORY_RAG", "1") != "1":
        return False
    resolved_arm = arm_id or os.environ.get("AGENTIC_ARM_ID") or "solo"
    try:
        return upsert_rec(
            rec,
            rec_id=rec_id,
            created_at=datetime.now(UTC).isoformat(),
            source=f"arm_{resolved_arm}",
            collection=collection,
            embedder=embedder if embedder is not None else _default_embed,
        )
    except Exception as e:  # noqa: BLE001 - ingestion failure never blocks trading
        logger.debug("memory ingest failed for rec %d: %s", rec_id, e)
        return False


def _flush(coll, batch: Iterable[tuple[str, str, dict]], embedder) -> None:
    items = list(batch)
    coll.upsert(
        ids=[i for i, _, _ in items],
        embeddings=embedder([t for _, t, _ in items]),
        documents=[t for _, t, _ in items],
        metadatas=[m for _, _, m in items],
    )
