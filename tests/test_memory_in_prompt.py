"""M17.D: retrieved precedents wired into the allocator prompt (sqlite-vec)."""

from __future__ import annotations

import pytest

from agentic_investor.memory.rec_index import _pack_vector
from agentic_investor.memory.store import EMBED_DIM, open_memory_conn
from agentic_investor.orchestrator.graph import _messages
from agentic_investor.orchestrator.state import (
    OrchestratorRequest,
)


def _fake_embedder(texts):
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


def _seed_row(conn, rec_id, source, text, tickers, created_at, outcome_1d=None):
    conn.execute(
        """
        INSERT INTO recs (
            rec_id, source, created_at, tickers, text,
            n_positions, avg_confidence, cash_pct, risk,
            outcome_pl_pct_1d
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (rec_id, source, created_at, tickers, text,
         len(tickers.split(",")), 0.7, 15.0, "moderate", outcome_1d),
    )
    conn.execute("DELETE FROM vec_recs WHERE rowid = ?", (rec_id,))
    conn.execute(
        "INSERT INTO vec_recs (rowid, embedding) VALUES (?, ?)",
        (rec_id, _pack_vector(_fake_embedder([text])[0])),
    )


@pytest.fixture
def seeded_store(monkeypatch):
    """A recs store with one obviously-relevant historical rec."""
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    import agentic_investor.memory.rec_index as rec_index_mod
    import agentic_investor.memory.retrieval as retrieval_mod
    import agentic_investor.tools.news as news_mod

    conn = open_memory_conn()
    monkeypatch.setattr(rec_index_mod, "_default_connection", lambda: conn)
    monkeypatch.setattr(rec_index_mod, "_default_embed", _fake_embedder)
    monkeypatch.setattr(retrieval_mod, "_default_connection", lambda: conn)
    monkeypatch.setattr(retrieval_mod, "_default_embed", _fake_embedder)
    # Also stub the news embedder so any accidental reload doesn't touch HF.
    monkeypatch.setattr(news_mod, "_embed_text", _fake_embedder)

    _seed_row(
        conn, rec_id=1, source="historical",
        text="Apple earnings defensive tech allocation cash buffer",
        tickers="AAPL,MSFT",
        created_at="2026-08-15T10:00:00+00:00",
        outcome_1d=0.42,
    )
    return conn


def _base_state(**overrides):
    req = OrchestratorRequest(
        tickers=["AAPL", "MSFT"], amount=100_000.0, risk="moderate",
    )
    state: dict = {
        "request": req,
        "technical_signals": [],
        "news_signals": [],
        "market_snapshots": {},
        "news_batch_context": (
            "Apple earnings beat guidance, defensive positioning."
        ),
    }
    state.update(overrides)
    return state


def test_section_11_renders_when_enabled_and_docs_exist(seeded_store, monkeypatch):
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    msgs = _messages(_base_state())
    user_content = msgs[1]["content"]
    fast_tail = user_content[1]["text"]
    assert "## 11. Similar past decisions" in fast_tail
    assert "AAPL,MSFT" in fast_tail
    assert "1d +0.42%" in fast_tail


def test_section_11_lives_in_fast_tail_not_cached_slow_prefix(
    seeded_store, monkeypatch,
):
    """Retrieval output changes every regen; if it leaked into slow_prefix
    the cache_control marker would tank the hit rate."""
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    msgs = _messages(_base_state())
    slow_prefix = msgs[1]["content"][0]["text"]
    fast_tail = msgs[1]["content"][1]["text"]
    assert "## 11." not in slow_prefix
    assert "## 11." in fast_tail


def test_disabled_via_env_flag_skips_retrieval(seeded_store, monkeypatch):
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "0")
    msgs = _messages(_base_state())
    fast_tail = msgs[1]["content"][1]["text"]
    assert "## 11." not in fast_tail
    assert "Similar past decisions" not in fast_tail


def test_no_batch_no_holdings_still_queries_on_risk_profile(
    seeded_store, monkeypatch,
):
    """Even a bare tick with no news + no prior alloc should query on risk
    profile; retrieval doesn't crash on minimal state."""
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    msgs = _messages(_base_state(news_batch_context=""))
    fast_tail = msgs[1]["content"][1]["text"]
    # Section 11 present because query text = "Risk profile: moderate, target ..."
    assert "## 11." in fast_tail


def test_retrieval_failure_never_blocks_prompt_build(seeded_store, monkeypatch):
    """A bug in memory.retrieval must never take down the allocator."""
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")

    def _boom(*a, **kw):
        raise RuntimeError("simulated store outage")

    import agentic_investor.memory.retrieval as retrieval_mod
    monkeypatch.setattr(retrieval_mod, "retrieve_similar", _boom)

    msgs = _messages(_base_state())
    fast_tail = msgs[1]["content"][1]["text"]
    # Section 11 gracefully absent; sections 1-8 still there.
    assert "## 11." not in fast_tail
    assert "## 1. Request" in msgs[1]["content"][0]["text"]


def test_memory_rag_k_env_override_takes_effect(seeded_store, monkeypatch):
    """AGENTIC_MEMORY_RAG_K env lets us A/B-test retrieval k per arm
    without a LoopConfig knob. Verify effective k = 8 not the default 4."""
    for i in range(2, 8):
        _seed_row(
            seeded_store, rec_id=i, source="historical",
            text=f"Apple defensive tech alloc variant {i}",
            tickers="AAPL,MSFT",
            created_at=f"2026-08-{10 + i:02d}T10:00:00+00:00",
        )
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.setenv("AGENTIC_MEMORY_RAG_K", "8")
    msgs = _messages(_base_state())
    fast_tail = msgs[1]["content"][1]["text"]
    section = fast_tail.split("## 11. Similar past decisions")[-1]
    line_count = section.count("[2026-")
    assert line_count >= 5, (
        f"expected k=8 retrieved lines in section 11, got {line_count}. "
        "AGENTIC_MEMORY_RAG_K env may not be respected."
    )


def test_arm_id_env_scopes_retrieval(seeded_store, monkeypatch):
    """Retrieval respects AGENTIC_ARM_ID for source filtering."""
    _seed_row(
        seeded_store, rec_id=100, source="arm_B",
        text="arm B live rec on tech",
        tickers="GOOGL",
        created_at="2026-09-05T10:00:00+00:00",
    )
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    msgs = _messages(_base_state())
    fast_tail = msgs[1]["content"][1]["text"]
    # Arm A view: sees historical (AAPL,MSFT rec), never arm_B's GOOGL rec.
    assert "GOOGL" not in fast_tail
    assert "AAPL,MSFT" in fast_tail
