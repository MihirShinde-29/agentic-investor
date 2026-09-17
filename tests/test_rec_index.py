"""Historical Recommendation indexer for M17 (sqlite-vec store)."""

from __future__ import annotations

from agentic_investor.memory.rec_index import (
    embed_text_for_rec,
    index_historical,
    metadata_for_rec,
    upsert_rec,
)
from agentic_investor.memory.store import EMBED_DIM, open_memory_conn
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    Recommendation,
)
from agentic_investor.orchestrator.store import save_recommendation


def _fake_embedder(texts):
    """Deterministic 384-dim vectors (mostly zeros) so tests don't pull
    sentence-transformers weights."""
    out = []
    for t in texts:
        h = hash(t)
        vec = [0.0] * EMBED_DIM
        vec[0] = (h & 0xff) / 255.0
        vec[1] = ((h >> 8) & 0xff) / 255.0
        vec[2] = ((h >> 16) & 0xff) / 255.0
        vec[3] = ((h >> 24) & 0xff) / 255.0
        out.append(vec)
    return out


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


def _row_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM recs").fetchone()[0]


def _vec_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM vec_recs").fetchone()[0]


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


def test_upsert_writes_one_doc():
    conn = open_memory_conn()
    rec = _make_rec(["AAPL"])
    wrote = upsert_rec(
        rec, rec_id=1, created_at="2026-09-01T10:00:00",
        source="historical", conn=conn, embedder=_fake_embedder,
    )
    assert wrote is True
    assert _row_count(conn) == 1
    assert _vec_count(conn) == 1


def test_upsert_skips_empty_text():
    conn = open_memory_conn()
    rec = _make_rec(["AAPL"], rationales=[""], portfolio_rationale="")
    wrote = upsert_rec(
        rec, rec_id=1, created_at="2026-09-01T10:00:00",
        source="historical", conn=conn, embedder=_fake_embedder,
    )
    assert wrote is False
    assert _row_count(conn) == 0


def test_upsert_is_idempotent():
    conn = open_memory_conn()
    rec = _make_rec(["AAPL"])
    for _ in range(3):
        upsert_rec(
            rec, rec_id=1, created_at="2026-09-01T10:00:00",
            source="historical", conn=conn, embedder=_fake_embedder,
        )
    assert _row_count(conn) == 1
    assert _vec_count(conn) == 1


def test_index_historical_reads_all_recs(tmp_path):
    db = tmp_path / "seed.db"
    db_url = f"sqlite:///{db}"
    for i in range(5):
        save_recommendation(_make_rec([f"T{i}"]), url=db_url)
    conn = open_memory_conn()
    n = index_historical(db_url=db_url, conn=conn, embedder=_fake_embedder)
    assert n == 5
    assert _row_count(conn) == 5


def test_index_historical_tags_source_historical(tmp_path):
    db = tmp_path / "seed.db"
    db_url = f"sqlite:///{db}"
    save_recommendation(_make_rec(["AAPL"]), url=db_url)
    conn = open_memory_conn()
    index_historical(db_url=db_url, conn=conn, embedder=_fake_embedder)
    sources = [r[0] for r in conn.execute("SELECT source FROM recs").fetchall()]
    assert all(s == "historical" for s in sources)


def test_index_historical_is_idempotent(tmp_path):
    db = tmp_path / "seed.db"
    db_url = f"sqlite:///{db}"
    for i in range(3):
        save_recommendation(_make_rec([f"T{i}"]), url=db_url)
    conn = open_memory_conn()
    index_historical(db_url=db_url, conn=conn, embedder=_fake_embedder)
    index_historical(db_url=db_url, conn=conn, embedder=_fake_embedder)
    assert _row_count(conn) == 3


def test_arm_source_replaces_by_rec_id_not_by_source():
    """sqlite-vec is keyed on rec_id alone; historical + arm_A rec_id=1
    collide unlike the earlier chroma composite doc-id. This is a
    behavior change worth documenting: arm live recs OVERWRITE historical
    entries with the same rec_id. Not a problem in practice because arm
    rec_ids come from arm-specific sqlite paths (each arm has its own
    autoincrement counter), so IDs don't collide across sources in the
    real system.
    """
    conn = open_memory_conn()
    rec = _make_rec(["AAPL"])
    upsert_rec(rec, 1, "2026-09-01T10:00:00", "historical",
               conn=conn, embedder=_fake_embedder)
    upsert_rec(rec, 1, "2026-09-01T10:00:00", "arm_A",
               conn=conn, embedder=_fake_embedder)
    # Second upsert overwrites; source reflects the latest write.
    row = conn.execute("SELECT source FROM recs WHERE rec_id = 1").fetchone()
    assert row[0] == "arm_A"
    assert _row_count(conn) == 1


def test_index_arm_rec_tags_source_from_env(monkeypatch):
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.setenv("AGENTIC_ARM_ID", "B")
    conn = open_memory_conn()
    ok = index_arm_rec(
        _make_rec(["NVDA"]), rec_id=42,
        conn=conn, embedder=_fake_embedder,
    )
    assert ok is True
    row = conn.execute("SELECT source FROM recs WHERE rec_id = 42").fetchone()
    assert row[0] == "arm_B"


def test_index_arm_rec_defaults_to_solo(monkeypatch):
    """Unset AGENTIC_ARM_ID = solo tag so single-arm loops still get memory."""
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.delenv("AGENTIC_ARM_ID", raising=False)
    conn = open_memory_conn()
    index_arm_rec(
        _make_rec(["AAPL"]), rec_id=1,
        conn=conn, embedder=_fake_embedder,
    )
    row = conn.execute("SELECT source FROM recs WHERE rec_id = 1").fetchone()
    assert row[0] == "arm_solo"


def test_index_arm_rec_skips_when_disabled(monkeypatch):
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "0")
    conn = open_memory_conn()
    ok = index_arm_rec(
        _make_rec(["AAPL"]), rec_id=1,
        conn=conn, embedder=_fake_embedder,
    )
    assert ok is False
    assert _row_count(conn) == 0


def test_index_arm_rec_stashes_db_url_in_row(monkeypatch):
    """db_url on the recs row is what lets a later memory-outcomes sweep
    read snapshots from the right arm SQLite - critical for arms whose
    DATABASE_URL differs from the caller's default."""
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///out/experiments/exp1/A.db")
    conn = open_memory_conn()
    index_arm_rec(
        _make_rec(["AAPL"]), rec_id=99,
        conn=conn, embedder=_fake_embedder,
    )
    row = conn.execute("SELECT db_url FROM recs WHERE rec_id = 99").fetchone()
    assert row[0] == "sqlite:///out/experiments/exp1/A.db"


def test_index_arm_rec_never_raises_on_failure(monkeypatch):
    """A store outage during ingestion cannot take down the loop."""
    from agentic_investor.memory.rec_index import index_arm_rec

    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")

    class _BustedConn:
        def execute(self, *a, **kw):
            raise RuntimeError("simulated store outage")

        def __enter__(self):
            raise RuntimeError("simulated store outage")

        def __exit__(self, *a):
            return False

    # Would raise if not caught; returning False is the contract.
    assert index_arm_rec(
        _make_rec(["AAPL"]), rec_id=1,
        conn=_BustedConn(), embedder=_fake_embedder,
    ) is False
