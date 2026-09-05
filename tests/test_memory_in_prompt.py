"""M17.D: retrieved precedents wired into the allocator prompt."""

from __future__ import annotations

import chromadb
import pytest

from agentic_investor.orchestrator.graph import _messages
from agentic_investor.orchestrator.state import (
    OrchestratorRequest,
)


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


@pytest.fixture
def seeded_chroma(tmp_path, monkeypatch):
    """A recommendations collection with one obviously-relevant historical rec."""
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    # Point the module-level Chroma factory at a fresh dir.
    import agentic_investor.memory.rec_index as rec_index_mod
    import agentic_investor.memory.retrieval as retrieval_mod
    import agentic_investor.tools.news as news_mod

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    coll = client.get_or_create_collection(
        name="recommendations", metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(rec_index_mod, "_default_collection", lambda: coll)
    monkeypatch.setattr(rec_index_mod, "_default_embed", _fake_embedder)
    monkeypatch.setattr(retrieval_mod, "_default_collection", lambda: coll)
    monkeypatch.setattr(retrieval_mod, "_default_embed", _fake_embedder)
    # Also stub the news embedder so any accidental reload doesn't touch HF.
    monkeypatch.setattr(news_mod, "_embed_text", _fake_embedder)

    coll.upsert(
        ids=["rec:historical:1"],
        embeddings=_fake_embedder(["Apple earnings defensive tech allocation cash buffer"]),
        documents=["Apple earnings defensive tech allocation cash buffer"],
        metadatas=[{
            "rec_id": 1, "source": "historical",
            "created_at": "2026-08-15T10:00:00+00:00",
            "tickers": "AAPL,MSFT",
            "n_positions": 2, "avg_confidence": 0.7,
            "cash_pct": 15.0, "risk": "moderate",
            "outcome_pl_pct_1d": 0.42,
        }],
    )
    return coll


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


def test_section_9_renders_when_enabled_and_docs_exist(seeded_chroma, monkeypatch):
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    msgs = _messages(_base_state())
    user_content = msgs[1]["content"]
    fast_tail = user_content[1]["text"]
    assert "## 11. Similar past decisions" in fast_tail
    assert "AAPL,MSFT" in fast_tail
    assert "1d +0.42%" in fast_tail


def test_section_9_lives_in_fast_tail_not_cached_slow_prefix(
    seeded_chroma, monkeypatch,
):
    """Retrieval output changes every regen; if it leaked into slow_prefix
    the cache_control marker would tank the hit rate."""
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    msgs = _messages(_base_state())
    slow_prefix = msgs[1]["content"][0]["text"]
    fast_tail = msgs[1]["content"][1]["text"]
    # Section header lives in fast_tail; USER_PREAMBLE (in slow_prefix)
    # mentions "9. Similar past decisions" as a TOC entry - that's fine
    # because the TOC never changes, only the rendered CONTENT is volatile.
    assert "## 11." not in slow_prefix
    assert "## 11." in fast_tail


def test_disabled_via_env_flag_skips_retrieval(seeded_chroma, monkeypatch):
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "0")
    msgs = _messages(_base_state())
    fast_tail = msgs[1]["content"][1]["text"]
    assert "## 11." not in fast_tail
    assert "Similar past decisions" not in fast_tail


def test_no_batch_no_holdings_still_queries_on_risk_profile(
    seeded_chroma, monkeypatch,
):
    """Even a bare tick with no news + no prior alloc should query on risk
    profile; retrieval doesn't crash on minimal state."""
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    msgs = _messages(_base_state(news_batch_context=""))
    fast_tail = msgs[1]["content"][1]["text"]
    # Section 9 present because query text = "Risk profile: moderate, target ..."
    assert "## 11." in fast_tail


def test_retrieval_failure_never_blocks_prompt_build(seeded_chroma, monkeypatch):
    """A bug in memory.retrieval must never take down the allocator."""
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")

    def _boom(*a, **kw):
        raise RuntimeError("simulated chroma outage")

    import agentic_investor.memory.retrieval as retrieval_mod
    monkeypatch.setattr(retrieval_mod, "retrieve_similar", _boom)

    msgs = _messages(_base_state())
    fast_tail = msgs[1]["content"][1]["text"]
    # Section 9 gracefully absent; sections 1-8 still there.
    assert "## 11." not in fast_tail
    assert "## 1. Request" in msgs[1]["content"][0]["text"]


def test_arm_id_env_scopes_retrieval(seeded_chroma, monkeypatch):
    """Retrieval respects AGENTIC_ARM_ID for source filtering."""
    seeded_chroma.upsert(
        ids=["rec:arm_B:1"],
        embeddings=_fake_embedder(["arm B live rec on tech"]),
        documents=["arm B live rec on tech"],
        metadatas=[{
            "rec_id": 1, "source": "arm_B",
            "created_at": "2026-09-05T10:00:00+00:00",
            "tickers": "GOOGL", "n_positions": 1,
            "avg_confidence": 0.8, "cash_pct": 5.0, "risk": "moderate",
        }],
    )
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    monkeypatch.setenv("AGENTIC_ARM_ID", "A")
    msgs = _messages(_base_state())
    fast_tail = msgs[1]["content"][1]["text"]
    # Arm A view: sees historical (AAPL,MSFT rec), never arm_B's GOOGL rec.
    assert "GOOGL" not in fast_tail
    assert "AAPL,MSFT" in fast_tail
