"""LangGraph orchestrator: fan out to agents, allocate, then validate.

Three nodes:
  gather_signals  runs both agents across all tickers in parallel via a
                  ThreadPoolExecutor. A failed agent for one ticker is
                  logged and skipped, not fatal.
  allocate        summarizes the collected signals and asks an allocator LLM
                  for an Allocation. The Allocation model validates that
                  weights sum to 100, and instructor auto-retries on failure.
  validate        checks risk-tier rules (max single, cash floor). Kept as a
                  separate node so a conditional retry-back-to-allocate loop
                  is a one-line change later.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph

from agentic_investor.agents.news import NewsSignal, analyze_news
from agentic_investor.agents.technical import TechnicalSignal, analyze_ticker
from agentic_investor.llm.client import structured_complete
from agentic_investor.orchestrator.state import (
    RISK_RULES,
    Allocation,
    GraphState,
    OrchestratorRequest,
    Recommendation,
    check_risk_rules,
)

logger = logging.getLogger(__name__)

ALLOCATOR_SYSTEM = """\
You are a disciplined portfolio allocator. Given per-ticker signals from a
technical-analysis agent and a news-sentiment agent, plus the user's amount,
risk tolerance, and target, produce a paper-portfolio allocation.

Hard rules your output MUST satisfy:
- All weights, including cash_pct, must sum to 100.
- Risk caps:
  * conservative: no single position > 20%, cash_pct >= 20.
  * moderate:     no single position > 35%, cash_pct >= 10.
  * aggressive:   no single position > 50%, cash_pct >= 0.
- For each position, dollars = amount * weight_pct / 100; same for cash.

How to reason:
- Bigger weight where technical and news agents agree with higher confidence.
- Smaller or zero weight when signals conflict or evidence is thin.
- If most signals are neutral or bearish, lean on cash.
- In each position rationale, cite the specific stances and drivers you used.
- In portfolio_rationale, summarize how the mix fits the risk band and target.
"""


def _summarize_signals(tech: list[TechnicalSignal], news: list[NewsSignal]) -> str:
    by_ticker: dict[str, dict] = {}
    for t in tech:
        by_ticker.setdefault(t.ticker, {})["technical"] = t.model_dump(
            include={"stance", "confidence", "reasoning", "key_drivers"}
        )
    for n in news:
        by_ticker.setdefault(n.ticker, {})["news"] = n.model_dump(
            include={"stance", "confidence", "reasoning"}
        )
    return json.dumps(by_ticker, indent=2)


def gather_signals(state: GraphState) -> dict:
    req = state["request"]
    tech: list[TechnicalSignal] = []
    news: list[NewsSignal] = []
    # Cap concurrent LLM calls at 2 to stay gentle on free-tier rate limits and
    # avoid instructor's mode-registration race on the very first parallel call.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for t in req.tickers:
            futures[pool.submit(analyze_ticker, t)] = ("tech", t)
            futures[pool.submit(analyze_news, t)] = ("news", t)
        for fut, (kind, ticker) in futures.items():
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001 - degrade gracefully per ticker
                logger.warning("%s agent failed for %s: %s", kind, ticker, e)
                continue
            if kind == "tech":
                tech.append(out)
            else:
                news.append(out)
    return {"technical_signals": tech, "news_signals": news}


def _messages(state: GraphState) -> list[dict]:
    req = state["request"]
    max_single, cash_floor = RISK_RULES[req.risk]
    signals = _summarize_signals(
        state.get("technical_signals", []), state.get("news_signals", [])
    )
    return [
        {"role": "system", "content": ALLOCATOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Request:\n"
                f"  tickers: {', '.join(req.tickers)}\n"
                f"  amount:  ${req.amount:,.2f}\n"
                f"  risk:    {req.risk} "
                f"(max single {max_single:.0f}%, cash floor {cash_floor:.0f}%)\n"
                f"  target:  {req.target}\n\n"
                f"Signals (JSON, keyed by ticker):\n{signals}\n\n"
                "Produce a valid Allocation."
            ),
        },
    ]


def allocate(state: GraphState) -> dict:
    return {"allocation": structured_complete(Allocation, _messages(state))}


def validate(state: GraphState) -> dict:
    return {"violations": check_risk_rules(state["allocation"], state["request"].risk)}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("gather_signals", gather_signals)
    g.add_node("allocate", allocate)
    g.add_node("validate", validate)
    g.add_edge(START, "gather_signals")
    g.add_edge("gather_signals", "allocate")
    g.add_edge("allocate", "validate")
    g.add_edge("validate", END)
    return g.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_orchestrator(request: OrchestratorRequest) -> Recommendation:
    final = _get_graph().invoke({"request": request})
    return Recommendation(
        request=request,
        allocation=final["allocation"],
        technical_signals=final.get("technical_signals", []),
        news_signals=final.get("news_signals", []),
        violations=final.get("violations", []),
    )
