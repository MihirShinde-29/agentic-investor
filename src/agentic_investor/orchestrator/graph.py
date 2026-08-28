"""LangGraph orchestrator: fan out to agents, allocate (profile-driven), then validate.

Three nodes:
  gather_signals  fetches per-ticker MarketSnapshots in parallel, then runs
                  the technical + news agents from those snapshots (also
                  parallel). Snapshots are kept in state so non-LLM allocators
                  can use them without re-fetching. A failed agent for one
                  ticker is logged and skipped, not fatal.
  allocate        routes to the allocator chosen by StrategyProfile.allocator:
                  - "llm"          -> in-file allocator LLM call
                  - "equal_weight" / "inverse_vol" / ...  -> allocators.py
                  Guardrails (max_single, cash_floor) come from the profile
                  and are enforced by the Allocation model_validator +
                  post-hoc check_profile_rules.
  validate        checks profile guardrails and emits violations. Kept
                  separate so a conditional retry-back-to-allocate loop is a
                  one-line change later.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph

from agentic_investor.agents.news import NewsSignal, analyze_news
from agentic_investor.agents.technical import TechnicalSignal, analyze_technical
from agentic_investor.llm.client import structured_complete
from agentic_investor.orchestrator.allocators import get_allocator
from agentic_investor.orchestrator.state import (
    Allocation,
    GraphState,
    OrchestratorRequest,
    Recommendation,
    check_profile_rules,
)
from agentic_investor.orchestrator.strategy import StrategyProfile, get_preset
from agentic_investor.tools.market import MarketSnapshot, get_market_snapshot

logger = logging.getLogger(__name__)

ALLOCATOR_SYSTEM = """\
You are a disciplined portfolio allocator. Given per-ticker signals from a
technical-analysis agent and a news-sentiment agent, plus the user's amount,
risk tolerance, and target, produce a paper-portfolio allocation.

Hard rules your output MUST satisfy:
- All weights, including cash_pct, must sum to 100.
- No single position weight may exceed the profile's max_single_pct cap.
- cash_pct must be at least the profile's cash_floor_pct.
- For each position, dollars = amount * weight_pct / 100; same for cash.
- Every position MUST include a `confidence` field in [0.0, 1.0].
  Do NOT omit it. Missing confidence disables downstream risk controls.

How to reason:
- Bigger weight where technical and news agents agree with higher confidence.
- Smaller or zero weight when signals conflict or evidence is thin.
- If most signals are neutral or bearish, lean on cash.
- In each position rationale, cite the specific stances and drivers you used.
- In portfolio_rationale, summarize how the mix fits the risk band and target.
- For each position also emit `confidence` in [0.0, 1.0] reflecting how sure
  you are of that weight. High (0.8-1.0) = both agents strongly agree, thesis
  is clear. Medium (0.5-0.7) = one strong signal, one weak or missing.
  Low (0.2-0.4) = conflicting signals, forced-choice sizing. The rebalancer
  uses this to widen bands on low-confidence positions (anti-churn).
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


def _profile_from_state(state: GraphState) -> StrategyProfile:
    """Fall back to the risk-tier preset if no explicit profile was supplied."""
    profile = state.get("profile")
    if profile is not None:
        return profile
    return get_preset(state["request"].risk)


def gather_signals(state: GraphState) -> dict:
    """Fetch snapshots + run technical/news agents. Store snapshots in state so
    non-LLM allocators can use them without re-fetching prices.
    """
    req = state["request"]

    # Fetch MarketSnapshots in parallel. Higher concurrency is fine here - no LLM calls.
    snapshots: dict[str, MarketSnapshot] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(2, len(req.tickers)))) as pool:
        fetches = {pool.submit(get_market_snapshot, t): t for t in req.tickers}
        for fut, ticker in fetches.items():
            try:
                snapshots[ticker] = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad ticker mustn't sink the run
                logger.warning("snapshot failed for %s: %s", ticker, e)

    # Analyze in parallel. Cap concurrent LLM calls at 2 to stay gentle on
    # free-tier rate limits + avoid instructor's mode-registration race.
    tech: list[TechnicalSignal] = []
    news: list[NewsSignal] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for ticker, snap in snapshots.items():
            futures[pool.submit(analyze_technical, snap)] = ("tech", ticker)
            futures[pool.submit(analyze_news, ticker)] = ("news", ticker)
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

    return {"technical_signals": tech, "news_signals": news, "market_snapshots": snapshots}


def _messages(state: GraphState) -> list[dict]:
    req = state["request"]
    profile = _profile_from_state(state)
    signals = _summarize_signals(
        state.get("technical_signals", []), state.get("news_signals", [])
    )
    batch_ctx = (state.get("news_batch_context") or "").strip()
    batch_block = (
        f"\nBreaking-news events (from live stream):\n{batch_ctx}\n"
        "Weight HOT news toward scout sizing (~50% of your intended target). "
        "Weight COOKED news toward full sizing informed by news_reaction_pct: "
        "high positive reaction = already priced in, avoid chasing; flat despite "
        "bullish news = underreaction, edge remains; sharp negative reaction on "
        "bad news = thesis re-evaluation warranted.\n"
        if batch_ctx
        else ""
    )
    # Prompt anchoring: if there's a previous allocation, show it explicitly
    # and ask for delta-form thinking. Reduces the LLM's tendency to re-
    # conceive the portfolio from scratch on every regen (source of today's
    # whipsaw pattern where 5-15pp per-position shifts came from LLM noise
    # rather than real signal change).
    prev_alloc = state.get("previous_allocation")
    if prev_alloc is not None:
        prev_lines = [
            f"  {p.ticker}: {p.weight_pct:.1f}%" for p in prev_alloc.positions
        ]
        prev_lines.append(f"  cash: {prev_alloc.cash_pct:.1f}%")
        prev_block = (
            "\nCurrent allocation (your previous decision):\n"
            + "\n".join(prev_lines)
            + "\n\nPropose CHANGES from this baseline. If a ticker's thesis is "
            "unchanged since last decision, keep its weight identical (do not "
            "round or adjust by 1-2pp on noise). Only shift weights when the "
            "signals justify the move. Weight changes > 10pp should be reserved "
            "for material catalysts (earnings, breaking news specific to that "
            "ticker) and cited explicitly in the rationale.\n"
        )
    else:
        prev_block = ""
    return [
        {"role": "system", "content": ALLOCATOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Request:\n"
                f"  tickers: {', '.join(req.tickers)}\n"
                f"  amount:  ${req.amount:,.2f}\n"
                f"  risk:    {req.risk} (profile '{profile.name}': max single "
                f"{profile.max_single_pct:.0f}%, cash floor "
                f"{profile.cash_floor_pct:.0f}%)\n"
                f"  target:  {req.target}\n\n"
                f"Signals (JSON, keyed by ticker):\n{signals}\n"
                f"{prev_block}"
                f"{batch_block}\n"
                "Produce a valid Allocation."
            ),
        },
    ]


def allocate(state: GraphState) -> dict:
    """Dispatch to the allocator selected by profile.allocator."""
    profile = _profile_from_state(state)
    if profile.allocator == "llm":
        return {"allocation": structured_complete(Allocation, _messages(state))}
    allocator_fn = get_allocator(profile.allocator)
    alloc = allocator_fn(
        state["request"],
        state.get("technical_signals", []),
        state.get("news_signals", []),
        state.get("market_snapshots", {}),
        profile,
    )
    return {"allocation": alloc}


def validate(state: GraphState) -> dict:
    profile = _profile_from_state(state)
    return {"violations": check_profile_rules(
        state["allocation"], profile,
        snapshots=state.get("market_snapshots"),
    )}


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


def run_orchestrator(
    request: OrchestratorRequest,
    profile: StrategyProfile | None = None,
    *,
    news_batch_context: str | None = None,
    previous_allocation: "Allocation | None" = None,
) -> Recommendation:
    """Run the orchestrator graph. If profile is None, uses the risk-tier preset.

    news_batch_context is an optional text block from the event-driven loop
    (HOT/COOKED news events with reaction_pct); when set, it flows into the
    allocator prompt so the LLM can weight breaking news in its decision.

    previous_allocation is an optional prior rec's allocation; when set, the
    allocator prompt is anchored to it and asked to propose delta-form changes
    rather than re-conceiving the portfolio from scratch. Prevents baseline
    LLM-variance churn (whipsaw pattern observed 2026-08-28 live session).
    """
    initial: dict = {"request": request}
    if profile is not None:
        initial["profile"] = profile
    if news_batch_context:
        initial["news_batch_context"] = news_batch_context
    if previous_allocation is not None:
        initial["previous_allocation"] = previous_allocation
    final = _get_graph().invoke(initial)
    return Recommendation(
        request=request,
        allocation=final["allocation"],
        technical_signals=final.get("technical_signals", []),
        news_signals=final.get("news_signals", []),
        violations=final.get("violations", []),
    )
