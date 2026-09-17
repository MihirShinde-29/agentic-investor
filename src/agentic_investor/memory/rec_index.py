"""sqlite-vec index of Recommendations for retrieval-augmented allocation (M17).

One shared store (`settings.rec_store_path`, default `./.rec_store.db`).
Each doc carries a `source` tag (`"historical"` for the seed corpus,
`f"arm_{id}"` for live arm runs) so retrieval can filter with
`WHERE source IN (...)` and keep A/B independence.

Replaces the chromadb version - see `store.py` module docstring and
docs/INTERVIEW_NOTES.md B14 for why we swapped.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from collections.abc import Callable
from pathlib import Path

from agentic_investor.memory.store import EMBED_DIM, get_connection
from agentic_investor.orchestrator.state import Recommendation

logger = logging.getLogger(__name__)


def _default_embed(texts: list[str]) -> list[list[float]]:
    from agentic_investor.tools.news import _embed_text

    return _embed_text(texts)


def _default_connection() -> sqlite3.Connection:
    return get_connection()


def _pack_vector(vec: list[float]) -> bytes:
    """Encode one float32 vector as the byte string vec0 wants."""
    if len(vec) != EMBED_DIM:
        raise ValueError(f"embedding dim {len(vec)} != expected {EMBED_DIM}")
    return struct.pack(f"{EMBED_DIM}f", *vec)


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
    """Row payload for the `recs` table (native SQL types)."""
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


def _upsert_row(
    conn: sqlite3.Connection,
    meta: dict,
    text: str,
    embedding: list[float],
    *,
    db_url: str | None = None,
) -> None:
    """Write both tables in one transaction; on conflict, replace.

    vec0's INSERT OR REPLACE requires deleting the old vec row first when
    the rowid already exists (its virtual-table implementation doesn't
    honor sqlite's normal REPLACE semantics for the vector column).
    """
    rec_id = int(meta["rec_id"])
    packed = _pack_vector(embedding)
    with conn:  # BEGIN/COMMIT
        conn.execute(
            """
            INSERT INTO recs (
                rec_id, source, created_at, tickers, text,
                n_positions, avg_confidence, cash_pct, risk, db_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rec_id) DO UPDATE SET
                source = excluded.source,
                created_at = excluded.created_at,
                tickers = excluded.tickers,
                text = excluded.text,
                n_positions = excluded.n_positions,
                avg_confidence = excluded.avg_confidence,
                cash_pct = excluded.cash_pct,
                risk = excluded.risk,
                db_url = excluded.db_url
            """,
            (
                rec_id, meta["source"], meta["created_at"],
                meta.get("tickers", ""), text,
                meta.get("n_positions", 0),
                meta.get("avg_confidence", 0.0),
                meta.get("cash_pct", 0.0),
                meta.get("risk", "moderate"),
                db_url,
            ),
        )
        conn.execute("DELETE FROM vec_recs WHERE rowid = ?", (rec_id,))
        conn.execute(
            "INSERT INTO vec_recs (rowid, embedding) VALUES (?, ?)",
            (rec_id, packed),
        )


def upsert_rec(
    rec: Recommendation,
    rec_id: int,
    created_at: str,
    source: str,
    *,
    conn: sqlite3.Connection | None = None,
    embedder: Callable[[list[str]], list[list[float]]] = _default_embed,
) -> bool:
    """Index a single rec. Returns True if written, False if skipped (empty text)."""
    text = embed_text_for_rec(rec)
    if not text:
        return False
    connection = conn if conn is not None else _default_connection()
    embedding = embedder([text])[0]
    meta = metadata_for_rec(rec, rec_id, created_at, source)
    _upsert_row(connection, meta, text, embedding)
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
    conn: sqlite3.Connection | None = None,
    embedder: Callable[[list[str]], list[list[float]]] = _default_embed,
    batch_size: int = 32,
) -> int:
    """Bulk-index every rec from the given DB (default: settings.database_url).

    Reads raw JSON blobs so schema drift in old recs doesn't block the index.
    Idempotent: re-running overwrites in place because rec_id is the primary key.
    """
    connection = conn if conn is not None else _default_connection()
    rows = _load_rec_blobs(db_url)
    logger.info(
        "indexing %d recommendations from %s",
        len(rows), db_url or "default DB",
    )
    n_indexed = 0
    n_skipped = 0
    batch: list[tuple[str, dict]] = []

    def _flush(items: list[tuple[str, dict]]) -> int:
        if not items:
            return 0
        embeddings = embedder([t for t, _ in items])
        for (text, meta), emb in zip(items, embeddings, strict=True):
            _upsert_row(connection, meta, text, emb)
        return len(items)

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
        batch.append((text, meta))
        if len(batch) >= batch_size:
            n_indexed += _flush(batch)
            batch = []
    n_indexed += _flush(batch)
    logger.info(
        "indexed %d recs (skipped %d) into rec store",
        n_indexed, n_skipped,
    )
    return n_indexed


def _load_rec_blobs(db_url: str | None) -> list[tuple[int, str, str]]:
    from agentic_investor.config import get_settings

    url = db_url or get_settings().database_url
    if not url.startswith("sqlite:///"):
        raise ValueError(f"only sqlite:/// URLs supported (got {url!r})")
    path = Path(url.removeprefix("sqlite:///"))
    with sqlite3.connect(str(path)) as source:
        return source.execute(
            "SELECT id, created_at, payload_json FROM recommendations ORDER BY id",
        ).fetchall()


def index_arm_rec(
    rec: Recommendation,
    rec_id: int,
    *,
    arm_id: str | None = None,
    conn: sqlite3.Connection | None = None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> bool:
    """Index a fresh live rec under source=f"arm_{id}".

    Reads AGENTIC_ARM_ID env when arm_id is not passed (solo paper-loop
    runs get arm_id="solo" so their memory is isolated from any A/B
    experiment). Also stashes the current DATABASE_URL in the row so a
    later memory-outcomes sweep knows which SQLite to read snapshots
    from - arm subprocesses each have their own DB path.

    Honors AGENTIC_MEMORY_RAG kill-switch. Never raises - failure
    returns False and the loop moves on.
    """
    from datetime import UTC, datetime

    from agentic_investor.config import get_settings
    from agentic_investor.flags import flags

    if not flags.MEMORY_RAG:
        return False
    resolved_arm = arm_id or flags.ARM_ID
    try:
        source = f"arm_{resolved_arm}"
        created_at = datetime.now(UTC).isoformat()
        text = embed_text_for_rec(rec)
        if not text:
            return False
        connection = conn if conn is not None else _default_connection()
        emb_fn = embedder if embedder is not None else _default_embed
        embedding = emb_fn([text])[0]
        meta = metadata_for_rec(rec, rec_id, created_at, source)
        _upsert_row(connection, meta, text, embedding,
                    db_url=get_settings().database_url)
        return True
    except Exception as e:  # noqa: BLE001 - ingestion failure never blocks trading
        logger.debug("memory ingest failed for rec %d: %s", rec_id, e)
        return False
