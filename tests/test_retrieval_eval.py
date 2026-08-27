"""Tests for L1 RAG retrieval evals. Fake embedder keeps the suite offline."""

import math

from agentic_investor.eval.retrieval import (
    hit_at_k,
    load_cases,
    load_fixtures,
    mrr,
    ndcg_at_k,
    recall_at_k,
    run_retrieval_eval,
)

# Metric functions - known-answer tests


def test_hit_at_k_true_when_relevant_in_top_k():
    assert hit_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0
    # Relevant at rank 4 but k=3 -> miss
    assert hit_at_k(["a", "b", "c", "z"], {"z"}, k=3) == 0.0


def test_recall_at_k_counts_relevant_found():
    # 2 of 3 relevant found in top-5
    assert recall_at_k(["a", "b", "c", "d", "e"], {"a", "b", "z"}, k=5) == 2 / 3
    # All relevant found
    assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0
    # No relevant given returns 0
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_mrr_reciprocal_of_first_relevant_rank():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0
    assert mrr(["x", "a", "b"], {"a"}) == 0.5
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


def test_ndcg_perfect_when_relevant_ranked_first():
    # 2 relevant, both at ranks 1 and 2 -> ideal ordering -> NDCG = 1
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_ndcg_lower_when_relevant_ranked_later():
    # relevant at rank 3 vs rank 1: dcg / idcg
    val = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    # dcg = 1/log2(4) = 0.5; idcg (1 relevant total) = 1/log2(2) = 1.0
    assert val == 0.5
    # Sanity: less than perfect
    assert val < 1.0


def test_ndcg_zero_when_no_relevant_retrieved():
    assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


# Fixture + case loading


def test_load_bundled_fixtures():
    fixtures = load_fixtures()
    assert len(fixtures) >= 10
    # Should span multiple tickers
    tickers = {f.ticker for f in fixtures}
    assert {"NVDA", "AAPL", "MSFT"} <= tickers


def test_load_bundled_cases():
    cases = load_cases()
    assert len(cases) >= 5
    for c in cases:
        assert c.ticker
        assert c.query
        assert c.relevant_ids
        # Every relevant id references a real fixture
    fixture_ids = {f.id for f in load_fixtures()}
    for c in cases:
        for rid in c.relevant_ids:
            assert rid in fixture_ids, f"case {c.id} refs missing fixture {rid}"


# End-to-end with a fake embedder


def _keyword_embedder(texts: list[str]) -> list[list[float]]:
    """Map text to a 6-dim vector by topic keyword hits. Enough to make ranking
    deterministic without loading sentence-transformers.
    """
    keywords = ["earn", "guidance", "product", "regulat", "china", "azure"]
    out = []
    for t in texts:
        low = t.lower()
        vec = [1.0 if kw in low else 0.0 for kw in keywords]
        # Normalize to unit length so cosine similarity behaves.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def test_run_retrieval_eval_end_to_end_with_fake_embedder():
    report = run_retrieval_eval(k=5, embedder=_keyword_embedder)

    assert report.n_cases >= 5
    assert report.n_fixtures >= 10
    # Even a coarse keyword embedder should hit on most cases.
    assert report.aggregate.hit_at_k >= 0.5
    # Every per-case retrieved list is at most k
    for cr in report.per_case:
        assert len(cr.retrieved_ids) <= 5
