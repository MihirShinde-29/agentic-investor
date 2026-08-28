"""L1 RAG retrieval evals: measure the news RAG against a golden query set.

Loads hand-curated fixture articles into an ephemeral Chroma collection, runs
each query, and scores against ground-truth relevant IDs using Hit@k, Recall@k,
MRR, and NDCG@k. Fully offline once fixtures exist: no live news-API calls.
The real sentence-transformers embedder is used by default so scores reflect
production quality; tests inject a fake embedder for speed.
"""

import math
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import chromadb
from pydantic import BaseModel, Field

from agentic_investor.tools import news
from agentic_investor.tools.news import NewsArticle

DEFAULT_FIXTURES = Path(__file__).parent / "datasets" / "news_fixtures.jsonl"
DEFAULT_CASES = Path(__file__).parent / "datasets" / "news_cases.jsonl"


class RetrievalCase(BaseModel):
    id: str
    ticker: str
    query: str
    relevant_ids: list[str]
    notes: str = ""


class RetrievalMetrics(BaseModel):
    hit_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


class CaseResult(BaseModel):
    case_id: str
    query: str
    ticker: str
    k: int
    retrieved_ids: list[str]
    relevant_ids: list[str]
    metrics: RetrievalMetrics


class RetrievalEvalReport(BaseModel):
    k: int
    n_cases: int
    n_fixtures: int
    aggregate: RetrievalMetrics
    per_case: list[CaseResult] = Field(default_factory=list)


def load_fixtures(path: str | Path = DEFAULT_FIXTURES) -> list[NewsArticle]:
    lines = Path(path).read_text().splitlines()
    return [NewsArticle.model_validate_json(line) for line in lines if line.strip()]


def load_cases(path: str | Path = DEFAULT_CASES) -> list[RetrievalCase]:
    lines = Path(path).read_text().splitlines()
    return [RetrievalCase.model_validate_json(line) for line in lines if line.strip()]


# Metric functions (unit-testable with pure Python inputs).


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(rid in relevant for rid in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    # Binary relevance: 1 if in relevant_ids else 0. Discount by log2(rank+1).
    dcg = sum(
        (1.0 if rid in relevant else 0.0) / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved[:k], start=1)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _score(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> RetrievalMetrics:
    return RetrievalMetrics(
        hit_at_k=round(hit_at_k(retrieved_ids, relevant_ids, k), 3),
        recall_at_k=round(recall_at_k(retrieved_ids, relevant_ids, k), 3),
        mrr=round(mrr(retrieved_ids, relevant_ids), 3),
        ndcg_at_k=round(ndcg_at_k(retrieved_ids, relevant_ids, k), 3),
    )


def _ephemeral_collection():
    # Unique name per call so parallel runs and repeated eval invocations do
    # not share state (Chroma EphemeralClient is a per-process singleton).
    import uuid

    return chromadb.EphemeralClient().get_or_create_collection(
        name=f"eval_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )


def run_retrieval_eval(
    *,
    k: int = 5,
    fixtures_path: str | Path = DEFAULT_FIXTURES,
    cases_path: str | Path = DEFAULT_CASES,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> RetrievalEvalReport:
    """Grade the news RAG against the golden set. Returns aggregate + per-case."""
    fixtures = load_fixtures(fixtures_path)
    cases = load_cases(cases_path)

    embed = embedder if embedder is not None else news._embed_text
    coll = _ephemeral_collection()
    news.upsert_news_articles(fixtures, collection=coll, embedder=embed)

    per_case: list[CaseResult] = []
    for case in cases:
        retrieved = news.retrieve_news(
            case.ticker, case.query, k=k, collection=coll, embedder=embed
        )
        retrieved_ids = [a.id for a in retrieved]
        metrics = _score(retrieved_ids, set(case.relevant_ids), k)
        per_case.append(
            CaseResult(
                case_id=case.id,
                query=case.query,
                ticker=case.ticker,
                k=k,
                retrieved_ids=retrieved_ids,
                relevant_ids=case.relevant_ids,
                metrics=metrics,
            )
        )

    aggregate = RetrievalMetrics(
        hit_at_k=round(mean(c.metrics.hit_at_k for c in per_case), 3) if per_case else 0.0,
        recall_at_k=round(mean(c.metrics.recall_at_k for c in per_case), 3) if per_case else 0.0,
        mrr=round(mean(c.metrics.mrr for c in per_case), 3) if per_case else 0.0,
        ndcg_at_k=round(mean(c.metrics.ndcg_at_k for c in per_case), 3) if per_case else 0.0,
    )

    return RetrievalEvalReport(
        k=k,
        n_cases=len(cases),
        n_fixtures=len(fixtures),
        aggregate=aggregate,
        per_case=per_case,
    )
