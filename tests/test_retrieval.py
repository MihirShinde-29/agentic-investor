"""A/B-safe retrieval over the sqlite-vec recommendations index (M17.C).

Post-2026-09-17 migration: fixtures now build an in-memory sqlite-vec DB
instead of a Chroma tempdir. Assertions on ordering/isolation are
identical; the sentinel-outcome test flipped to NULL-passthrough because
the sqlite schema stores NULL directly (no -9999.0 sentinel dance).
"""

from __future__ import annotations

import pytest

from agentic_investor.memory.rec_index import _pack_vector
from agentic_investor.memory.retrieval import RetrievedRec, retrieve_similar
from agentic_investor.memory.store import EMBED_DIM, open_memory_conn


def _fake_embedder(texts):
    """Deterministic 384-dim vector per text (mostly zeros; first 4 slots
    carry the hashed signature so identical texts collide to the same
    embedding and distinct texts diverge).
    """
    out: list[list[float]] = []
    for t in texts:
        h = hash(t)
        vec = [0.0] * EMBED_DIM
        vec[0] = (h & 0xff) / 255.0
        vec[1] = ((h >> 8) & 0xff) / 255.0
        vec[2] = ((h >> 16) & 0xff) / 255.0
        vec[3] = ((h >> 24) & 0xff) / 255.0
        out.append(vec)
    return out


def _tmp_conn():
    return open_memory_conn()


def _seed_row(
    conn, source: str, rec_id: int, text: str,
    tickers: str = "AAPL", created_at: str = "2026-09-01T10:00:00+00:00",
    outcome_15m: float | None = None,
    outcome_60m: float | None = None,
    outcome_1d: float | None = None,
    outcome_1w: float | None = None,
):
    conn.execute(
        """
        INSERT OR REPLACE INTO recs (
            rec_id, source, created_at, tickers, text,
            n_positions, avg_confidence, cash_pct, risk,
            outcome_pl_pct_15m, outcome_pl_pct_60m,
            outcome_pl_pct_1d, outcome_pl_pct_1w
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rec_id, source, created_at, tickers, text,
            len(tickers.split(",")), 0.7, 0.0, "moderate",
            outcome_15m, outcome_60m, outcome_1d, outcome_1w,
        ),
    )
    conn.execute("DELETE FROM vec_recs WHERE rowid = ?", (rec_id,))
    conn.execute(
        "INSERT INTO vec_recs (rowid, embedding) VALUES (?, ?)",
        (rec_id, _pack_vector(_fake_embedder([text])[0])),
    )


def test_returns_empty_for_empty_query():
    conn = _tmp_conn()
    _seed_row(conn, "historical", 1, "some text")
    assert retrieve_similar("", "A", conn=conn, embedder=_fake_embedder) == []
    assert retrieve_similar("   ", "A", conn=conn, embedder=_fake_embedder) == []


def test_arm_a_cannot_see_arm_b_recs():
    """The core A/B invariant - identical text, different source, must not leak."""
    conn = _tmp_conn()
    _seed_row(conn, "arm_B", 1, "identical query text here")
    _seed_row(conn, "arm_C", 2, "identical query text here")
    results = retrieve_similar(
        "identical query text here", "A",
        k=10, conn=conn, embedder=_fake_embedder,
    )
    sources = {r.source for r in results}
    assert "arm_B" not in sources
    assert "arm_C" not in sources


def test_arm_a_sees_own_and_historical():
    conn = _tmp_conn()
    _seed_row(conn, "historical", 1, "shared knowledge")
    _seed_row(conn, "arm_A", 2, "shared knowledge")
    _seed_row(conn, "arm_B", 3, "shared knowledge")
    results = retrieve_similar(
        "shared knowledge", "A",
        k=10, conn=conn, embedder=_fake_embedder,
    )
    sources = {r.source for r in results}
    assert sources == {"historical", "arm_A"}


def test_include_historical_false_returns_only_own_arm():
    conn = _tmp_conn()
    _seed_row(conn, "historical", 1, "text")
    _seed_row(conn, "arm_A", 2, "text")
    results = retrieve_similar(
        "text", "A", k=10, include_historical=False,
        conn=conn, embedder=_fake_embedder,
    )
    assert {r.source for r in results} == {"arm_A"}


def test_topk_respected():
    conn = _tmp_conn()
    for i in range(10):
        _seed_row(conn, "historical", i, f"doc {i}")
    results = retrieve_similar(
        "doc 5", "A", k=3, conn=conn, embedder=_fake_embedder,
    )
    assert len(results) == 3


def test_null_outcomes_render_as_none():
    """Unripe horizons are stored as NULL; retrieved as Python None."""
    conn = _tmp_conn()
    _seed_row(
        conn, "historical", 1, "text",
        outcome_15m=None, outcome_60m=0.5,
        outcome_1d=None, outcome_1w=None,
    )
    results = retrieve_similar(
        "text", "A", k=1, conn=conn, embedder=_fake_embedder,
    )
    r = results[0]
    assert r.outcome_pl_pct_15m is None
    assert r.outcome_pl_pct_60m == 0.5
    assert r.outcome_pl_pct_1d is None
    assert r.outcome_pl_pct_1w is None


def test_successful_ranked_above_failed_on_tie():
    """When similarity is identical, higher 1d outcome ranks first."""
    conn = _tmp_conn()
    _seed_row(conn, "historical", 1, "duplicate", outcome_1d=-2.5)
    _seed_row(conn, "historical", 2, "duplicate", outcome_1d=+3.1)
    results = retrieve_similar(
        "duplicate", "A", k=2, conn=conn, embedder=_fake_embedder,
    )
    # Same doc text -> identical distance -> tiebreak by outcome_1d DESC
    assert results[0].outcome_pl_pct_1d == 3.1
    assert results[1].outcome_pl_pct_1d == -2.5


def test_similarity_is_monotonic_in_distance():
    """Exact match -> distance ~0 -> similarity near 1."""
    conn = _tmp_conn()
    _seed_row(conn, "historical", 1, "hello world")
    results = retrieve_similar(
        "hello world", "A", k=1, conn=conn, embedder=_fake_embedder,
    )
    # Identical embedding => L2 distance 0 => similarity == 1.0 in our
    # 1 / (1 + d) mapping.
    assert results[0].similarity == pytest.approx(1.0)


def test_to_prompt_line_includes_trajectory_and_text():
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


def test_to_prompt_line_no_outcome_data():
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
    assert "no outcome yet" in line


def test_to_prompt_line_partial_trajectory_marked():
    r = RetrievedRec(
        rec_id=1, source="arm_A",
        created_at="2026-09-05T10:00:00+00:00",
        tickers=["TSLA"], similarity=0.7,
        text="partially matured", n_positions=1,
        avg_confidence=0.6, risk="moderate",
        outcome_pl_pct_15m=0.5, outcome_pl_pct_60m=None,
        outcome_pl_pct_1d=None, outcome_pl_pct_1w=None,
    )
    line = r.to_prompt_line()
    assert "(partial)" in line
    assert "15m +0.50%" in line


def test_tiebreak_cascades_to_longest_available_horizon():
    """When 1d is None but 60m is present, 60m anchors the tiebreak
    (not treated as 0.0). Recent important recs don't lose to older
    slightly-positive ones just because 1d hasn't matured yet."""
    conn = _tmp_conn()
    _seed_row(conn, "historical", 1, "same",
              created_at="2026-09-01T10:00:00+00:00", outcome_1d=-0.5)
    _seed_row(conn, "historical", 2, "same",
              created_at="2026-09-05T10:00:00+00:00", outcome_60m=0.8)
    results = retrieve_similar(
        "same", "A", k=2, conn=conn, embedder=_fake_embedder,
    )
    # rec 2 (60m +0.8 = tiebreak +0.8) beats rec 1 (1d -0.5 = tiebreak -0.5)
    assert results[0].rec_id == 2
    assert results[1].rec_id == 1


def test_max_age_days_filters_stale_recs():
    from datetime import UTC, datetime, timedelta

    conn = _tmp_conn()
    old = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    recent = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    _seed_row(conn, "historical", 1, "text", created_at=old)
    _seed_row(conn, "historical", 2, "text", created_at=recent)
    results = retrieve_similar(
        "text", "A", k=10, max_age_days=30,
        conn=conn, embedder=_fake_embedder,
    )
    ids = {r.rec_id for r in results}
    assert 2 in ids
    assert 1 not in ids


def test_empty_collection_returns_empty():
    conn = _tmp_conn()
    results = retrieve_similar("x", "A", conn=conn, embedder=_fake_embedder)
    assert results == []


@pytest.mark.parametrize("arm_id", ["A", "B", "C"])
def test_isolation_holds_for_every_arm(arm_id):
    """Parameterized: no matter which arm queries, it only sees own + historical."""
    conn = _tmp_conn()
    for i, other in enumerate(["A", "B", "C"], start=1):
        _seed_row(conn, f"arm_{other}", i, "x")
    _seed_row(conn, "historical", 4, "x")
    results = retrieve_similar(
        "x", arm_id, k=10, conn=conn, embedder=_fake_embedder,
    )
    for r in results:
        assert r.source in ("historical", f"arm_{arm_id}")
