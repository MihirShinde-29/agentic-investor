"""A/B-safe retrieval over the recommendations Chroma index (M17.C)."""

from __future__ import annotations

import chromadb
import pytest

from agentic_investor.memory.retrieval import RetrievedRec, retrieve_similar


def _fake_embedder(texts):
    return [
        [
            (hash(t) & 0xff) / 255.0,
            ((hash(t) >> 8) & 0xff) / 255.0,
            ((hash(t) >> 16) & 0xff) / 255.0,
            ((hash(t) >> 24) & 0xff) / 255.0,
        ]
        for t in texts
    ]


def _tmp_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return client.get_or_create_collection(
        name="recommendations", metadata={"hnsw:space": "cosine"},
    )


def _seed_doc(
    coll, doc_id: str, source: str, rec_id: int, text: str,
    tickers: str = "AAPL", created_at: str = "2026-09-01T10:00:00+00:00",
    outcome_1d: float | None = None,
):
    meta = {
        "rec_id": rec_id,
        "source": source,
        "created_at": created_at,
        "tickers": tickers,
        "n_positions": len(tickers.split(",")),
        "avg_confidence": 0.7,
        "cash_pct": 0.0,
        "risk": "moderate",
    }
    if outcome_1d is not None:
        meta["outcome_pl_pct_1d"] = outcome_1d
    coll.upsert(
        ids=[doc_id],
        embeddings=_fake_embedder([text]),
        documents=[text],
        metadatas=[meta],
    )


def test_returns_empty_for_empty_query(tmp_path):
    coll = _tmp_collection(tmp_path)
    _seed_doc(coll, "rec:historical:1", "historical", 1, "some text")
    assert retrieve_similar("", "A", collection=coll, embedder=_fake_embedder) == []
    assert retrieve_similar("   ", "A", collection=coll, embedder=_fake_embedder) == []


def test_arm_a_cannot_see_arm_b_recs(tmp_path):
    """The core A/B invariant - identical text, different source, must not leak."""
    coll = _tmp_collection(tmp_path)
    _seed_doc(coll, "rec:arm_B:1", "arm_B", 1, "identical query text here")
    _seed_doc(coll, "rec:arm_C:1", "arm_C", 1, "identical query text here")
    results = retrieve_similar(
        "identical query text here", "A",
        k=10, collection=coll, embedder=_fake_embedder,
    )
    sources = {r.source for r in results}
    assert "arm_B" not in sources
    assert "arm_C" not in sources


def test_arm_a_sees_own_and_historical(tmp_path):
    coll = _tmp_collection(tmp_path)
    _seed_doc(coll, "rec:historical:1", "historical", 1, "shared knowledge")
    _seed_doc(coll, "rec:arm_A:1", "arm_A", 1, "shared knowledge")
    _seed_doc(coll, "rec:arm_B:1", "arm_B", 1, "shared knowledge")
    results = retrieve_similar(
        "shared knowledge", "A",
        k=10, collection=coll, embedder=_fake_embedder,
    )
    sources = {r.source for r in results}
    assert sources == {"historical", "arm_A"}


def test_include_historical_false_returns_only_own_arm(tmp_path):
    coll = _tmp_collection(tmp_path)
    _seed_doc(coll, "rec:historical:1", "historical", 1, "text")
    _seed_doc(coll, "rec:arm_A:1", "arm_A", 1, "text")
    results = retrieve_similar(
        "text", "A", k=10, include_historical=False,
        collection=coll, embedder=_fake_embedder,
    )
    assert {r.source for r in results} == {"arm_A"}


def test_topk_respected(tmp_path):
    coll = _tmp_collection(tmp_path)
    for i in range(10):
        _seed_doc(coll, f"rec:historical:{i}", "historical", i, f"doc {i}")
    results = retrieve_similar(
        "doc 5", "A", k=3, collection=coll, embedder=_fake_embedder,
    )
    assert len(results) == 3


def test_sentinel_outcomes_render_as_none(tmp_path):
    coll = _tmp_collection(tmp_path)
    coll.upsert(
        ids=["rec:historical:1"],
        embeddings=_fake_embedder(["text"]),
        documents=["text"],
        metadatas=[{
            "rec_id": 1, "source": "historical",
            "created_at": "2026-09-01T10:00:00+00:00",
            "tickers": "AAPL", "n_positions": 1,
            "avg_confidence": 0.7, "cash_pct": 0.0, "risk": "moderate",
            "outcome_pl_pct_15m": -9999.0,
            "outcome_pl_pct_60m": 0.5,
            "outcome_pl_pct_1d": -9999.0,
            "outcome_pl_pct_1w": -9999.0,
        }],
    )
    results = retrieve_similar(
        "text", "A", k=1, collection=coll, embedder=_fake_embedder,
    )
    r = results[0]
    assert r.outcome_pl_pct_15m is None
    assert r.outcome_pl_pct_60m == 0.5
    assert r.outcome_pl_pct_1d is None
    assert r.outcome_pl_pct_1w is None


def test_successful_ranked_above_failed_on_tie(tmp_path):
    """When similarity is identical, higher 1d outcome ranks first."""
    coll = _tmp_collection(tmp_path)
    _seed_doc(coll, "rec:historical:1", "historical", 1, "duplicate", outcome_1d=-2.5)
    _seed_doc(coll, "rec:historical:2", "historical", 2, "duplicate", outcome_1d=+3.1)
    results = retrieve_similar(
        "duplicate", "A", k=2, collection=coll, embedder=_fake_embedder,
    )
    # Same doc text -> identical distance -> tiebreak by outcome_1d DESC
    assert results[0].outcome_pl_pct_1d == 3.1
    assert results[1].outcome_pl_pct_1d == -2.5


def test_similarity_is_one_minus_cosine_distance(tmp_path):
    coll = _tmp_collection(tmp_path)
    _seed_doc(coll, "rec:historical:1", "historical", 1, "hello world")
    results = retrieve_similar(
        "hello world", "A", k=1, collection=coll, embedder=_fake_embedder,
    )
    # Identical text (same embedding) -> distance ~0 -> similarity ~1
    assert results[0].similarity > 0.99


def test_to_prompt_line_includes_trajectory_and_text(tmp_path):
    r = RetrievedRec(
        rec_id=1, source="historical",
        created_at="2026-09-01T10:00:00+00:00",
        tickers=["AAPL", "MSFT"],
        similarity=0.87,
        text="balanced tech allocation with cash buffer",
        n_positions=2, avg_confidence=0.7, risk="moderate",
        outcome_pl_pct_15m=0.02, outcome_pl_pct_60m=-0.01,
        outcome_pl_pct_1d=0.45, outcome_pl_pct_1w=None,
    )
    line = r.to_prompt_line()
    assert "2026-09-01" in line
    assert "AAPL,MSFT" in line
    assert "15m +0.02%" in line
    assert "60m -0.01%" in line
    assert "1d +0.45%" in line
    assert "1w" not in line  # skipped because None
    assert "balanced tech allocation" in line


def test_to_prompt_line_no_outcome_data(tmp_path):
    r = RetrievedRec(
        rec_id=1, source="arm_A",
        created_at="2026-09-05T10:00:00+00:00",
        tickers=["TSLA"], similarity=0.7,
        text="fresh rec no bars yet",
        n_positions=1, avg_confidence=0.6, risk="moderate",
        outcome_pl_pct_15m=None, outcome_pl_pct_60m=None,
        outcome_pl_pct_1d=None, outcome_pl_pct_1w=None,
    )
    line = r.to_prompt_line()
    assert "no outcome" in line


def test_max_age_days_filters_stale_recs(tmp_path):
    from datetime import UTC, datetime, timedelta

    coll = _tmp_collection(tmp_path)
    old = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    recent = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    _seed_doc(coll, "rec:historical:1", "historical", 1, "text", created_at=old)
    _seed_doc(coll, "rec:historical:2", "historical", 2, "text", created_at=recent)
    results = retrieve_similar(
        "text", "A", k=10, max_age_days=30,
        collection=coll, embedder=_fake_embedder,
    )
    ids = {r.rec_id for r in results}
    assert 2 in ids
    assert 1 not in ids


def test_empty_collection_returns_empty(tmp_path):
    coll = _tmp_collection(tmp_path)
    results = retrieve_similar("x", "A", collection=coll, embedder=_fake_embedder)
    assert results == []


@pytest.mark.parametrize("arm_id", ["A", "B", "C"])
def test_isolation_holds_for_every_arm(tmp_path, arm_id):
    """Parameterized: no matter which arm queries, it only sees own + historical."""
    coll = _tmp_collection(tmp_path)
    for other in ["A", "B", "C"]:
        _seed_doc(coll, f"rec:arm_{other}:1", f"arm_{other}", 1, "x")
    _seed_doc(coll, "rec:historical:1", "historical", 1, "x")
    results = retrieve_similar(
        "x", arm_id, k=10, collection=coll, embedder=_fake_embedder,
    )
    for r in results:
        assert r.source in ("historical", f"arm_{arm_id}")
