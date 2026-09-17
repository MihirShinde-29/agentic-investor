"""A/B-safe retrieval over the recommendations sqlite-vec index.

Arm X only ever sees docs with source ∈ {"historical", "arm_X"}. Cross-arm
leaks would poison the A/B experiment - arm B's live decisions must not
influence what arm A retrieves as "similar past reasoning". The filter is
mandatory; there is no code path that queries without it.

Same signatures as the earlier chromadb-backed retrieval so callers in
`orchestrator/graph.py._similar_precedents_block` don't move.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from agentic_investor.memory.rec_index import _default_connection, _default_embed
from agentic_investor.memory.store import EMBED_DIM


def _unsentinel(v) -> float | None:
    """NULL passthrough; sqlite gives us Python None directly, no
    -9999.0 sentinel dance like chroma needed. Kept as a helper so
    the RetrievedRec fields stay Optional[float] and callers don't
    need to change.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


@dataclass(frozen=True)
class RetrievedRec:
    rec_id: int
    source: str
    created_at: str
    tickers: list[str]
    similarity: float
    text: str
    n_positions: int
    avg_confidence: float
    risk: str
    outcome_pl_pct_15m: float | None
    outcome_pl_pct_60m: float | None
    outcome_pl_pct_1d: float | None
    outcome_pl_pct_1w: float | None

    def to_prompt_line(self, max_text: int = 220) -> str:
        """Compact one-line rendering for injection into the allocator prompt.

        Trailing hint flags partial trajectories so the LLM can weight
        a not-fully-matured precedent accordingly.
        """
        date = self.created_at[:10] if self.created_at else "?"
        head = f"[{date}] {','.join(self.tickers) or '?'}"
        traj_parts: list[str] = []
        n_present = 0
        for horizon in ("15m", "60m", "1d", "1w"):
            val = getattr(self, f"outcome_pl_pct_{horizon}")
            if val is not None:
                traj_parts.append(f"{horizon} {val:+.2f}%")
                n_present += 1
        if not traj_parts:
            trajectory = "no outcome yet"
        elif n_present < 4:
            trajectory = " -> ".join(traj_parts) + " (partial)"
        else:
            trajectory = " -> ".join(traj_parts)
        text = (self.text or "").replace("\n", " ").strip()
        if len(text) > max_text:
            text = text[: max_text - 1] + "…"
        return f"{head} [{trajectory}] {text}"

    def _tiebreak_score(self) -> float:
        """Longest-available horizon anchors the tiebreak.

        Prefer 1w > 1d > 60m > 15m so mature outcomes dominate; a rec
        with only 15m data still uses its immediate reaction rather
        than being flattened to 0.
        """
        for h in ("1w", "1d", "60m", "15m"):
            v = getattr(self, f"outcome_pl_pct_{h}")
            if v is not None:
                return v
        return 0.0


def _pack_vector(vec: list[float]) -> bytes:
    if len(vec) != EMBED_DIM:
        raise ValueError(f"embedding dim {len(vec)} != expected {EMBED_DIM}")
    return struct.pack(f"{EMBED_DIM}f", *vec)


def retrieve_similar(
    query_text: str,
    arm_id: str,
    *,
    k: int = 4,
    include_historical: bool = True,
    max_age_days: int | None = None,
    conn: sqlite3.Connection | None = None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[RetrievedRec]:
    """Top-K past recs semantically similar to query_text, scoped to this arm.

    Filters `source ∈ {"historical", f"arm_{arm_id}"}` unconditionally. When
    `include_historical=False`, drops "historical" so only the arm's own live
    reasoning is retrieved (useful for detecting drift late in a session).
    """
    if not query_text or not query_text.strip():
        return []
    connection = conn if conn is not None else _default_connection()
    emb = embedder if embedder is not None else _default_embed

    sources: list[str] = []
    if include_historical:
        sources.append("historical")
    sources.append(f"arm_{arm_id}")
    placeholders = ",".join("?" for _ in sources)

    # Over-fetch when age-filtering client-side; sqlite's WHERE + vec MATCH
    # composes cleanly but we still want extra headroom so a well-scoring
    # but stale hit doesn't push a fresh one out of the top-K.
    fetch_k = k * 4 if max_age_days is not None else k
    # Route via recorded_now so a bit-exact replay uses the same
    # cutoff as the recording (task #155). No-op when replay is off.
    from agentic_investor.orchestrator.recorder import recorded_now
    cutoff_iso = (
        (recorded_now() - timedelta(days=max_age_days)).isoformat()
        if max_age_days is not None else None
    )

    query_embedding = _pack_vector(emb([query_text])[0])

    sql = f"""
        SELECT
            r.rec_id, r.source, r.created_at, r.tickers, r.text,
            r.n_positions, r.avg_confidence, r.risk,
            r.outcome_pl_pct_15m, r.outcome_pl_pct_60m,
            r.outcome_pl_pct_1d, r.outcome_pl_pct_1w,
            v.distance
        FROM vec_recs v
        JOIN recs r ON r.rec_id = v.rowid
        WHERE v.embedding MATCH ?
          AND v.k = ?
          AND r.source IN ({placeholders})
        ORDER BY v.distance
    """
    rows = connection.execute(sql, (query_embedding, fetch_k, *sources)).fetchall()

    out: list[RetrievedRec] = []
    for row in rows:
        (rec_id, source, created_at, tickers_str, text,
         n_positions, avg_confidence, risk,
         pl_15m, pl_60m, pl_1d, pl_1w, dist) = row
        if cutoff_iso is not None and created_at and created_at < cutoff_iso:
            continue
        # sqlite-vec returns L2 or cosine distance depending on the vec0
        # table declaration. We default to L2 in `store.py`; convert to a
        # bounded similarity for callers that treated the chroma value
        # (1 - cos_sim) the same way. This is monotonic and preserves
        # ordering; the absolute number is comparable within a query.
        similarity = round(1.0 / (1.0 + float(dist)), 4)
        tickers = [t for t in (tickers_str or "").split(",") if t]
        out.append(RetrievedRec(
            rec_id=int(rec_id),
            source=str(source or ""),
            created_at=str(created_at or ""),
            tickers=tickers,
            similarity=similarity,
            text=text or "",
            n_positions=int(n_positions or 0),
            avg_confidence=float(avg_confidence or 0.0),
            risk=str(risk or "moderate"),
            outcome_pl_pct_15m=_unsentinel(pl_15m),
            outcome_pl_pct_60m=_unsentinel(pl_60m),
            outcome_pl_pct_1d=_unsentinel(pl_1d),
            outcome_pl_pct_1w=_unsentinel(pl_1w),
        ))
    # Nudge successful precedents ahead of failed ones on near-ties in
    # similarity. Uses 1d outcome as the anchor (highest coverage +
    # meaningful signal per M17.B distribution).
    out.sort(key=lambda r: (-r.similarity, -r._tiebreak_score()))
    return out[:k]
