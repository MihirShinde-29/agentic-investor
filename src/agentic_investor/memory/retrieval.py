"""A/B-safe retrieval over the recommendations Chroma index.

Arm X only ever sees docs with source ∈ {"historical", "arm_X"}. Cross-arm
leaks would poison the A/B experiment - arm B's live decisions must not
influence what arm A retrieves as "similar past reasoning". The filter is
mandatory; there is no code path that queries without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from agentic_investor.memory.rec_index import _default_collection, _default_embed

_SENTINEL = -9999.0  # matches memory.outcomes.attach_outcomes_to_index


def _unsentinel(v) -> float | None:
    if v is None or v == _SENTINEL:
        return None
    return float(v)


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


def retrieve_similar(
    query_text: str,
    arm_id: str,
    *,
    k: int = 4,
    include_historical: bool = True,
    max_age_days: int | None = None,
    collection=None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[RetrievedRec]:
    """Top-K past recs semantically similar to query_text, scoped to this arm.

    Filters `source ∈ {"historical", f"arm_{arm_id}"}` unconditionally. When
    `include_historical=False`, drops "historical" so only the arm's own live
    reasoning is retrieved (useful for detecting drift late in a session).
    """
    if not query_text or not query_text.strip():
        return []
    coll = collection if collection is not None else _default_collection()
    emb = embedder if embedder is not None else _default_embed

    sources: list[str] = []
    if include_historical:
        sources.append("historical")
    sources.append(f"arm_{arm_id}")

    # Chroma's $gt only accepts numeric metadata; created_at is ISO string
    # so we over-fetch and age-filter client-side.
    fetch_k = k * 4 if max_age_days is not None else k
    cutoff_iso = (
        (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat()
        if max_age_days is not None else None
    )
    res = coll.query(
        query_embeddings=emb([query_text]),
        n_results=fetch_k,
        where={"source": {"$in": sources}},
        include=["metadatas", "documents", "distances"],
    )
    metas = (res.get("metadatas") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[RetrievedRec] = []
    for meta, text, dist in zip(metas, docs, dists, strict=False):
        created_at = str(meta.get("created_at") or "")
        if cutoff_iso is not None and created_at and created_at < cutoff_iso:
            continue
        # Cosine distance in Chroma is (1 - cos_sim); invert for similarity.
        similarity = round(1.0 - float(dist), 4)
        tickers_str = str(meta.get("tickers") or "")
        tickers = [t for t in tickers_str.split(",") if t]
        out.append(RetrievedRec(
            rec_id=int(meta.get("rec_id") or 0),
            source=str(meta.get("source") or ""),
            created_at=str(meta.get("created_at") or ""),
            tickers=tickers,
            similarity=similarity,
            text=text or "",
            n_positions=int(meta.get("n_positions") or 0),
            avg_confidence=float(meta.get("avg_confidence") or 0.0),
            risk=str(meta.get("risk") or "moderate"),
            outcome_pl_pct_15m=_unsentinel(meta.get("outcome_pl_pct_15m")),
            outcome_pl_pct_60m=_unsentinel(meta.get("outcome_pl_pct_60m")),
            outcome_pl_pct_1d=_unsentinel(meta.get("outcome_pl_pct_1d")),
            outcome_pl_pct_1w=_unsentinel(meta.get("outcome_pl_pct_1w")),
        ))
    # Nudge successful precedents ahead of failed ones on near-ties in
    # similarity. Uses 1d outcome as the anchor (highest coverage +
    # meaningful signal per M17.B distribution).
    out.sort(key=lambda r: (-r.similarity, -r._tiebreak_score()))
    return out[:k]
