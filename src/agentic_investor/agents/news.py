"""News-Sentiment Agent: RAG over recent news with grounded citations.

The LLM cites by 1-based index into the retrieved articles; URL resolution
happens in code so a hallucinated source URL is structurally impossible.
"""

from typing import Literal

from pydantic import BaseModel, Field

from agentic_investor.llm.client import structured_complete
from agentic_investor.tools.news import (
    NewsArticle,
    fetch_company_news,
    retrieve_news,
    upsert_news_articles,
)

Stance = Literal["bullish", "neutral", "bearish"]


class Citation(BaseModel):
    headline: str
    url: str
    source: str


class _NewsSignalDraft(BaseModel):
    """What the LLM produces. Cite-by-index only; we resolve to real URLs after."""

    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    citation_ids: list[int] = Field(
        default_factory=list,
        description="1-based indices into the retrieved articles you cited",
    )


class NewsSignal(BaseModel):
    """Public output. Citations are resolved from the retrieved articles, not the LLM."""

    ticker: str
    stance: Stance
    confidence: float
    reasoning: str
    citations: list[Citation]


SYSTEM = """\
You are a disciplined news-sentiment agent. Given a set of recent news articles
about a single stock (each labeled [n]), classify near-term sentiment. Ground
every claim in the provided articles by listing the [n] indices you used in
citation_ids. If evidence is thin or conflicting, prefer 'neutral' with lower
confidence. Do not invent facts. Confidence is your calibrated probability the
stance is correct, in [0, 1]."""

DEFAULT_QUERY = (
    "material recent news likely to move near-term stock sentiment: "
    "earnings, guidance, revenue, deals, product launches, lawsuits, "
    "leadership changes, regulatory actions"
)


def _format_context(articles: list[NewsArticle]) -> str:
    if not articles:
        return "(no recent articles found)"
    lines = []
    for i, a in enumerate(articles, start=1):
        date = a.published_at[:10] if a.published_at else "n/a"
        summary = a.summary[:400]  # cap to keep the prompt tight
        lines.append(f"[{i}] ({a.source}, {date}) {a.headline}\n    {summary}")
    return "\n\n".join(lines)


def _messages(ticker: str, articles: list[NewsArticle]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Ticker: {ticker}\n\n"
                f"Recent news:\n{_format_context(articles)}\n\n"
                "Classify near-term sentiment for this ticker."
            ),
        },
    ]


def _resolve_citations(ids: list[int], articles: list[NewsArticle]) -> list[Citation]:
    out: list[Citation] = []
    seen: set[str] = set()
    for i in ids:
        if 1 <= i <= len(articles):
            a = articles[i - 1]
            if a.url in seen:
                continue
            seen.add(a.url)
            out.append(Citation(headline=a.headline, url=a.url, source=a.source))
    return out


def analyze_news(
    ticker: str,
    *,
    refresh: bool = True,
    days: int = 7,
    k: int = 6,
    query: str = DEFAULT_QUERY,
    model: str | None = None,
) -> NewsSignal:
    if refresh:
        upsert_news_articles(fetch_company_news(ticker, days=days))
    top = retrieve_news(ticker, query, k=k)

    draft = structured_complete(_NewsSignalDraft, _messages(ticker, top), model=model)
    return NewsSignal(
        ticker=ticker.upper(),
        stance=draft.stance,
        confidence=draft.confidence,
        reasoning=draft.reasoning,
        citations=_resolve_citations(draft.citation_ids, top),
    )
