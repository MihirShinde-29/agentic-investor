"""Historical Recommendation indexer for M17."""

from __future__ import annotations

import chromadb

from agentic_investor.memory.rec_index import (
    embed_text_for_rec,
    index_historical,
    metadata_for_rec,
    upsert_rec,
)
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    Recommendation,
)
from agentic_investor.orchestrator.store import save_recommendation


def _fake_embedder(texts):
    # Deterministic 4-dim vectors from string hashes so tests don't
    # pull down sentence-transformers weights.
    return [
        [
            (hash(t) & 0xff) / 255.0,
            ((hash(t) >> 8) & 0xff) / 255.0,
            ((hash(t) >> 16) & 0xff) / 255.0,
            ((hash(t) >> 24) & 0xff) / 255.0,
        ]
        for t in texts
    ]


def _make_rec(
    tickers: list[str],
    rationales: list[str] | None = None,
    portfolio_rationale: str = "diversified across three sectors",
) -> Recommendation:
    rationales = rationales or [f"{t} thesis" for t in tickers]
    positions = [
        Position(
            ticker=t,
            weight_pct=round(100.0 / max(len(tickers), 1), 2),
            dollars=1000.0,
            rationale=r,
            confidence=0.6,
        )
        for t, r in zip(tickers, rationales, strict=False)
    ]
    return Recommendation(
        request=OrchestratorRequest(
            tickers=tickers,
            amount=10_000.0,
            risk="moderate",
        ),
        allocation=Allocation(
            positions=positions,
            cash_pct=0.0,
            cash_dollars=0.0,
            portfolio_rationale=portfolio_rationale,
        ),
    )


def _tmp_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return client.get_or_create_collection(
        name="recommendations", metadata={"hnsw:space": "cosine"},
    )


def test_embed_text_uses_portfolio_and_top_positions():
    rec = _make_rec(
        ["AAPL", "MSFT", "NVDA", "TSLA"],
        rationales=["A rat", "M rat", "N rat", "T rat"],
        portfolio_rationale="global tech tilt",
    )
    text = embed_text_for_rec(rec)
    assert "global tech tilt" in text
    # Top-3 by weight (all equal weight here, order stable) — the 4th
    # position must NOT appear so we cap prompt bloat per-rec.
    assert "A rat" in text or "M rat" in text or "N rat" in text
    assert "T rat" not in text


def test_metadata_is_all_scalars():
    rec = _make_rec(["AAPL", "MSFT"])
    meta = metadata_for_rec(rec, rec_id=42, created_at="2026-09-01T10:00:00", source="historical")
    for k, v in meta.items():
        assert isinstance(v, (str, int, float, bool)), f"{k}={v!r} is not scalar"
    assert meta["tickers"] == "AAPL,MSFT"
    assert meta["source"] == "historical"
    assert meta["rec_id"] == 42


def test_upsert_writes_one_doc(tmp_path):
    coll = _tmp_collection(tmp_path)
    rec = _make_rec(["AAPL"])
    wrote = upsert_rec(
        rec, rec_id=1, created_at="2026-09-01T10:00:00",
        source="historical", collection=coll, embedder=_fake_embedder,
    )
    assert wrote is True
    assert coll.count() == 1


def test_upsert_skips_empty_text(tmp_path):
    coll = _tmp_collection(tmp_path)
    rec = _make_rec(["AAPL"], rationales=[""], portfolio_rationale="")
    wrote = upsert_rec(
        rec, rec_id=1, created_at="2026-09-01T10:00:00",
        source="historical", collection=coll, embedder=_fake_embedder,
    )
    assert wrote is False
    assert coll.count() == 0


def test_upsert_is_idempotent(tmp_path):
    coll = _tmp_collection(tmp_path)
    rec = _make_rec(["AAPL"])
    for _ in range(3):
        upsert_rec(
            rec, rec_id=1, created_at="2026-09-01T10:00:00",
            source="historical", collection=coll, embedder=_fake_embedder,
        )
    assert coll.count() == 1


def test_index_historical_reads_all_recs(tmp_path):
    db = tmp_path / "seed.db"
    db_url = f"sqlite:///{db}"
    for i in range(5):
        save_recommendation(_make_rec([f"T{i}"]), url=db_url)
    coll = _tmp_collection(tmp_path)
    n = index_historical(db_url=db_url, collection=coll, embedder=_fake_embedder)
    assert n == 5
    assert coll.count() == 5


def test_index_historical_tags_source_historical(tmp_path):
    db = tmp_path / "seed.db"
    db_url = f"sqlite:///{db}"
    save_recommendation(_make_rec(["AAPL"]), url=db_url)
    coll = _tmp_collection(tmp_path)
    index_historical(db_url=db_url, collection=coll, embedder=_fake_embedder)
    res = coll.get()
    assert all(m["source"] == "historical" for m in res["metadatas"])


def test_index_historical_is_idempotent(tmp_path):
    db = tmp_path / "seed.db"
    db_url = f"sqlite:///{db}"
    for i in range(3):
        save_recommendation(_make_rec([f"T{i}"]), url=db_url)
    coll = _tmp_collection(tmp_path)
    index_historical(db_url=db_url, collection=coll, embedder=_fake_embedder)
    index_historical(db_url=db_url, collection=coll, embedder=_fake_embedder)
    assert coll.count() == 3


def test_arm_source_writes_distinct_doc_id(tmp_path):
    """Same rec_id with different source must produce distinct Chroma docs."""
    coll = _tmp_collection(tmp_path)
    rec = _make_rec(["AAPL"])
    upsert_rec(rec, 1, "2026-09-01T10:00:00", "historical",
               collection=coll, embedder=_fake_embedder)
    upsert_rec(rec, 1, "2026-09-01T10:00:00", "arm_A",
               collection=coll, embedder=_fake_embedder)
    assert coll.count() == 2


def test_index_arm_rec_tags_source_from_env(tmp_path, monkeypatch):
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.setenv("AGENTIC_ARM_ID", "B")
    coll = _tmp_collection(tmp_path)
    ok = index_arm_rec(
        _make_rec(["NVDA"]), rec_id=42,
        collection=coll, embedder=_fake_embedder,
    )
    assert ok is True
    res = coll.get()
    assert res["metadatas"][0]["source"] == "arm_B"


def test_index_arm_rec_defaults_to_solo(tmp_path, monkeypatch):
    """Unset AGENTIC_ARM_ID = solo tag so single-arm loops still get memory."""
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.delenv("AGENTIC_ARM_ID", raising=False)
    coll = _tmp_collection(tmp_path)
    index_arm_rec(
        _make_rec(["AAPL"]), rec_id=1,
        collection=coll, embedder=_fake_embedder,
    )
    assert coll.get()["metadatas"][0]["source"] == "arm_solo"


def test_index_arm_rec_skips_when_disabled(tmp_path, monkeypatch):
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "0")
    coll = _tmp_collection(tmp_path)
    ok = index_arm_rec(
        _make_rec(["AAPL"]), rec_id=1,
        collection=coll, embedder=_fake_embedder,
    )
    assert ok is False
    assert coll.count() == 0


def test_index_arm_rec_stashes_db_url_in_metadata(tmp_path, monkeypatch):
    """The db_url metadata is what lets a later memory-outcomes sweep
    read snapshots from the right arm SQLite - critical for arms whose
    DATABASE_URL differs from the caller's default."""
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///out/experiments/exp1/A.db")
    coll = _tmp_collection(tmp_path)
    index_arm_rec(
        _make_rec(["AAPL"]), rec_id=99,
        collection=coll, embedder=_fake_embedder,
    )
    meta = coll.get(ids=["rec:arm_A:99"])["metadatas"][0]
    assert meta["db_url"] == "sqlite:///out/experiments/exp1/A.db"


def test_index_arm_rec_never_raises_on_failure(tmp_path, monkeypatch):
    """A chroma outage during ingestion cannot take down the loop."""
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")

    class _BustedColl:
        def upsert(self, *a, **kw):
            raise RuntimeError("simulated chroma outage")

    # Would raise if not caught; returning False is the contract.
    assert index_arm_rec(
        _make_rec(["AAPL"]), rec_id=1,
        collection=_BustedColl(), embedder=_fake_embedder,
    ) is False
