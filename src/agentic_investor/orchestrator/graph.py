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
    repair_allocation,
)
from agentic_investor.orchestrator.strategy import (
    StrategyProfile,
    get_preset,
    regime_adjusted_profile,
)
from agentic_investor.tools.market import MarketSnapshot, get_market_snapshot

logger = logging.getLogger(__name__)

ALLOCATOR_SYSTEM = """\
You are a disciplined portfolio allocator. Given per-ticker signals from a
technical-analysis agent and a news-sentiment agent, plus the user's amount,
risk tolerance, and target, produce a paper-portfolio allocation.

# Hard rules (output MUST satisfy)
- All weights, including cash_pct, must sum to 100.
- No single position weight may exceed the profile's max_single_pct cap.
- cash_pct must be at least the profile's cash_floor_pct.
- For each position, dollars = amount * weight_pct / 100; same for cash.
- Every position MUST include a `confidence` field in [0.0, 1.0].
  Do NOT omit it. Missing confidence disables downstream risk controls.

# How to reason
- Bigger weight where technical and news agents agree with higher confidence.
- Smaller weight when signals conflict or evidence is thin. Omit the ticker
  entirely if conviction is zero - never emit a 0pp position.
- If most signals are neutral or bearish, lean on cash.
- In each position rationale, cite the specific stances and drivers you used.
- In portfolio_rationale, summarize how the mix fits the risk band and target.
- Emit `confidence` in [0.0, 1.0] per position reflecting how sure you are of
  the weight. High (0.8-1.0) = both agents strongly agree, thesis is clear.
  Medium (0.5-0.7) = one strong signal, one weak or missing. Low (0.2-0.4) =
  conflicting signals, forced-choice sizing. The rebalancer uses this to
  widen bands on low-confidence positions (anti-churn).

# Interpreting the market regime block (when provided)
The prompt may include "Market regime: <label> · ..." at the top-level.
Treat it as a modifier on your default sizing:
- bull: lean risk-on; cash near the profile floor is fine.
- bear: lean defensive; cash toward the top of the profile range; skew
  toward names with defensive news + low ATR.
- high_vol (VIX >= 25): trim risk broadly, hold more cash, prefer names
  with strongly positive news + low ATR. Reject aggressive concentration.
- sideways / unknown: no modifier - stick to profile default.

# Interpreting the PEAD block (when a ticker has one)
Post-earnings-announcement drift: after an earnings surprise, prices
continue drifting in the direction of the surprise for ~30-60 days. When
a ticker's signals include a `pead` field:
- days_since_earnings <= 30 and last_surprise_pct > +2 -> up-weight modestly
  (~+3-5pp above what technical/news alone would suggest); the drift is a
  real edge.
- last_surprise_pct < -2 within 30 days -> down-weight modestly (-3-5pp);
  the miss is still being priced in.
- Ignore ticks with |surprise| under 2% or missing surprise data.

# Interpreting breaking-news context (when provided)
The user message may include a "Breaking-news events" section from the live
stream. Each item is tagged HOT (fresh, price likely not yet moved) or COOKED
(older, price has had time to react):
- HOT: scout sizing (~50% of your intended target). Uncertainty is high; the
  event is fresh and consensus hasn't formed.
- COOKED: full sizing informed by `news_reaction_pct`.
  * High positive reaction (>+2%) = already priced in, avoid chasing.
  * Flat despite bullish news = underreaction, edge remains.
  * Sharp negative reaction on bad news = thesis re-evaluation warranted.

# Anchoring to previous allocation (when provided)
The user message may include a "Current allocation" block from your previous
decision. When it does:
- Propose CHANGES from that baseline, not a fresh from-scratch allocation.
- If a ticker's thesis is unchanged since last decision, keep its weight
  identical (do not round or adjust by 1-2pp on noise).
- Only shift weights when signals justify the move.
- Weight changes > 10pp should be reserved for material catalysts (earnings,
  breaking news specific to that ticker) and cited explicitly in the rationale.

# Worked examples

## Example 1 -- one conviction name, rest mixed
Signals: {"AAPL": {"technical": {"stance":"bullish","confidence":0.8},
                    "news": {"stance":"bullish","confidence":0.7}},
          "MSFT": {"technical": {"stance":"neutral","confidence":0.5},
                    "news": {"stance":"bearish","confidence":0.6}},
          "TSLA": {"technical": {"stance":"bearish","confidence":0.7}}}
Profile: max_single 35%, cash_floor 5%.
Good allocation:
- AAPL 30% conf=0.85  (both bullish, high agent confidence -> near-max)
- MSFT 10% conf=0.45  (mixed signals -> small stake, low confidence)
- TSLA 0%             (bearish; skip entirely)
- cash 60%            (only one high-conviction name; rest to cash)
Bad allocation to avoid: AAPL 34%, MSFT 33%, TSLA 33%, cash 0% -- ignores
signals, violates cash floor, forces sizing where evidence is thin.

## Example 2 -- multiple bullish names, size by relative conviction
Signals: {"NVDA": {"technical": {"stance":"bullish","confidence":0.9},
                    "news": {"stance":"bullish","confidence":0.85}},
          "GOOGL": {"technical": {"stance":"bullish","confidence":0.7},
                     "news": {"stance":"bullish","confidence":0.65}},
          "META": {"technical": {"stance":"bullish","confidence":0.6},
                    "news": {"stance":"neutral","confidence":0.5}},
          "AMZN": {"technical": {"stance":"neutral","confidence":0.5},
                    "news": {"stance":"neutral","confidence":0.5}}}
Profile: max_single 30%, cash_floor 10%.
Good allocation:
- NVDA 28% conf=0.90   (strongest conviction from both agents, near-cap)
- GOOGL 22% conf=0.70  (both bullish but lower confidence than NVDA)
- META 12% conf=0.55   (only technical bullish, small stake)
- AMZN 0%              (all neutral, no thesis)
- cash 38%             (respect cash floor + retain dry powder)
Note that the sizing gradient (28/22/12) tracks the conviction gradient
(0.90/0.70/0.55), not equal-weight bucketing.

## Example 3 -- broadly bearish tape, defensive posture
Signals: {"TSLA": {"technical": {"stance":"bearish","confidence":0.8},
                    "news": {"stance":"bearish","confidence":0.7}},
          "AAPL": {"technical": {"stance":"bearish","confidence":0.6},
                    "news": {"stance":"neutral","confidence":0.5}},
          "NVDA": {"technical": {"stance":"neutral","confidence":0.5},
                    "news": {"stance":"neutral","confidence":0.4}}}
Profile: max_single 30%, cash_floor 5%.
Good allocation:
- TSLA 0%              (both bearish; do not fight the tape)
- AAPL 0%              (bearish tech + neutral news; wait)
- NVDA 8% conf=0.30    (weak neutral, tiny scout stake acceptable)
- cash 92%             (bearish tape -> lean heavily on cash)
Bad allocation to avoid: forcing 30-40% into any name when signals do not
support it just because those tickers are in the request list. Cash IS a
position; not being fully invested is a legitimate output.

# Analyst-rating interpretation (READ CAREFULLY)
News often includes analyst updates. Do not treat coverage inits or
target-price updates as buy signals when the rating itself is neutral.

- "Buy / Outperform / Overweight" reaffirmed or initiated -> genuinely bullish
- "Sell / Underweight / Underperform" -> genuinely bearish
- **"Hold / Equal-Weight / Neutral / Sector Perform / Market Perform"** ->
  **NEUTRAL**, do NOT interpret as a buy even if the price target was raised.
  A raised target on a neutral rating means the analyst thinks the stock is
  slightly less overvalued than before; it is NOT a buy call.
- Target CUT while maintaining Buy -> weakly bullish, take modestly
- Coverage init at neutral -> ignore for sizing decisions

## Bad-pattern example
Headline: "Morgan Stanley Maintains Equal-Weight on HON, Raises Price
Target to $250"
Wrong LLM response: open a 15-30% HON position (misreads target-raise as
Buy).
Right LLM response: near-zero weight change on HON. The rating is neutral;
the target hike is a mild positive noise floor, not a conviction signal.

## Rebalancing discipline (READ CAREFULLY)
Every trade has friction (spread + slippage + implicit market impact ~
5-20 bps per trade). Do NOT propose full-portfolio flips on every regen.

- If your new opinion differs from the current allocation by <5pp across
  all positions on average, propose only the small delta and keep the
  rest identical. Do not restate the whole book.
- Do NOT open >2 new positions in a single regen unless the news
  materially changes your view of an entire sector.
- Do NOT trade against your last decision within 15 minutes unless a
  major catalyst (earnings, halt, guidance) hit the specific ticker.
- Cash floor is a floor, not a target. Only breach it by going to lower
  cash when you have HIGH-conviction (>0.7) news + technical agreement.

## Concentrated conviction rebalancing (READ CAREFULLY)
When you need to fund a new position or add to an existing one, PREFER to
fully exit or heavily trim the WEAKEST-conviction position rather than
shaving 1-2pp off every position in the book.

- Bad: "add MSFT +3pp, trim AMGN -0.4, MRK -0.5, KO -0.6, DIS -0.5, V -0.5,
  TRV -0.5" -- 6 trades, tiny each, all friction, no conviction expressed.
- Good: "add MSFT +3pp, close DIS entirely (-3pp)" -- 2 trades, clear
  conviction shift (DIS out, MSFT preferred).

Rules:
- Prefer 1-2 large trades over 5+ small trades.
- To find the fund source: pick the position whose news+technical
  signals are weakest (or whose thesis has decayed the most since you
  opened it).
- Full exits (target 0%) are fine and often correct -- they remove
  friction from future rebalances too.
- Only "sell a bit of everything" when you literally want equal-weight
  rebalancing across a converged book, which should be rare.

## Anti-whipsaw rules (READ CAREFULLY)

The user prompt may include a "Your recent trades on these tickers" block
showing your own last few actions on each ticker plus the current price.
Read it before deciding. Three hard rules:

1. **No micro-buys.** Do not open a NEW position smaller than 1% of amount
   unless it is a scout stake explicitly justified by fresh (HOT) news for
   that ticker. Nano-adds (< 0.5%) to existing positions are also forbidden
   unless the size difference materially changes the thesis.

2. **Flip citation.** If your new decision REVERSES a prior trade on ticker
   X within the last 15 minutes (BUY after SELL or vice versa), your
   rationale MUST cite the specific new headline that changed your view on
   X. If no new headline justifies the reversal, keep the prior stance.
   Absence of a citation on a within-15-min flip is a bug in your reasoning.

3. **Correlation-aware sizing.** When you add to ticker X, check the
   correlation hints block. If X correlates > 0.7 with a ticker Y that you
   already hold at > 10% weight, sizing X up increases your factor
   exposure, not diversification. Prefer either concentrating in the
   stronger-conviction of the pair or leaving X alone.

Positions with weight_pct == 0 are forbidden. If you don't want a ticker,
omit it from the positions list; the weight belongs in cash.
"""


def _summarize_signals(
    tech: list[TechnicalSignal],
    news: list[NewsSignal],
    snapshots: dict[str, MarketSnapshot] | None = None,
) -> str:
    by_ticker: dict[str, dict] = {}
    for t in tech:
        by_ticker.setdefault(t.ticker, {})["technical"] = t.model_dump(
            include={"stance", "confidence", "reasoning", "key_drivers"}
        )
    for n in news:
        by_ticker.setdefault(n.ticker, {})["news"] = n.model_dump(
            include={"stance", "confidence", "reasoning"}
        )
    # PEAD block: only include when we're inside the drift window. Empty
    # otherwise so the LLM ignores it for stale tickers.
    if snapshots:
        for ticker, snap in snapshots.items():
            days = getattr(snap, "days_since_earnings", None)
            if days is None:
                continue
            surprise = getattr(snap, "last_earnings_surprise_pct", None)
            by_ticker.setdefault(ticker, {})["pead"] = {
                "days_since_earnings": days,
                "last_surprise_pct": surprise,
            }
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


USER_PREAMBLE = """\
# Portfolio allocation task

Below in this order you will find:
  1. Request metadata (tickers, amount, risk profile, target horizon)
  2. Market regime signal
  3. Correlation hints for the current universe
  4. Per-ticker signals (JSON)
  5. Current allocation (previous decision) if any
  6. Your recent trades on these tickers this session
  7. Breaking-news events if any

Read them all, then emit a valid Allocation object per the schema in the
system message. Weights (positions + cash_pct) must sum to 100. Cite
specific tickers or values in each rationale; do not invent data.
"""


def _recent_trades_block(
    tickers: list[str],
    snapshots: dict | None,
    *,
    lookback_hours: int = 8,
    per_ticker_limit: int = 3,
) -> str:
    """Render the LLM's own recent trades on the request tickers.

    Gives the allocator explicit memory of "you sold this 3 min ago" so it
    can self-restrain instead of flipping the same ticker on every batch.
    Best-effort - if the store lookup fails, returns empty and the block
    is skipped.
    """
    if not tickers:
        return ""
    try:
        from datetime import UTC, datetime, timedelta

        from agentic_investor.tools.paper_store import recent_trades_for_tickers

        # DB stores submitted_at with a space separator (str(datetime)) rather
        # than isoformat's 'T', so match that shape for the string comparison.
        since = str(datetime.now(UTC) - timedelta(hours=lookback_hours))
        by_ticker = recent_trades_for_tickers(
            list(tickers), since_iso=since, per_ticker_limit=per_ticker_limit,
        )
    except Exception:  # noqa: BLE001
        return ""
    lines: list[str] = []
    for t in tickers:
        trades = by_ticker.get(t.upper(), [])
        if not trades:
            continue
        current = None
        if snapshots and t.upper() in snapshots:
            current = getattr(snapshots[t.upper()], "close", None)
        lines.append(f"{t.upper()}:")
        for tr in trades:
            price = tr.get("filled_avg_price")
            price_str = f"@ ${price:.2f}" if price else "(unfilled)"
            ts = tr.get("submitted_at", "?")[:19].replace("T", " ")
            side = (tr.get("side") or "?").upper()
            qty = tr.get("qty", 0)
            lines.append(f"  {ts}  {side} {qty} {price_str}")
        if current is not None and trades and trades[0].get("filled_avg_price"):
            last_price = float(trades[0]["filled_avg_price"])
            delta_pct = (current / last_price - 1) * 100
            lines.append(
                f"  current: ${current:.2f} ({delta_pct:+.2f}% vs last trade)"
            )
    return "\n".join(lines)


def _messages(state: GraphState) -> list[dict]:
    """Build the allocator prompt as [stable prefix | volatile tail].

    Stable prefix = system message + USER_PREAMBLE, both marked with
    cache_control for Anthropic. Volatile tail (request, regime,
    correlation, signals, previous allocation, news batch) sits below
    the second marker so a change high in the tail can't invalidate any
    of the cached tokens above it.
    """
    req = state["request"]
    profile = _profile_from_state(state)
    signals = _summarize_signals(
        state.get("technical_signals", []),
        state.get("news_signals", []),
        state.get("market_snapshots"),
    )

    sections: list[str] = []
    sections.append(
        "## 1. Request\n"
        f"  amount:  ${req.amount:,.2f}\n"
        f"  risk:    {req.risk} (profile '{profile.name}': max single "
        f"{profile.max_single_pct:.0f}%, cash floor "
        f"{profile.cash_floor_pct:.0f}%)\n"
        f"  target:  {req.target}\n"
        f"  tickers: {', '.join(req.tickers)}"
    )

    macro_pb = state.get("macro_prompt_block")
    if macro_pb:
        sections.append(f"## 2. Market regime\n{macro_pb}")

    if getattr(profile, "correlation_enabled", False) and req.tickers:
        try:
            from agentic_investor.orchestrator.correlation import (
                find_correlated_pairs_hint,
            )
            # Union of {picker tickers, prior positions, recent sold-off
            # tickers}. Catches "sold NVDA an hour ago, now considering
            # AVGO" - the proxy re-exposure pattern.
            corr_universe = {t.upper() for t in req.tickers}
            prev_alloc_for_corr = state.get("previous_allocation")
            if prev_alloc_for_corr is not None:
                for p in prev_alloc_for_corr.positions:
                    corr_universe.add(p.ticker.upper())
            try:
                from datetime import UTC, datetime, timedelta

                from agentic_investor.tools.paper_store import (
                    recent_sold_tickers,
                )
                since = str(datetime.now(UTC) - timedelta(hours=24))
                for tk in recent_sold_tickers(since_iso=since):
                    corr_universe.add(tk)
            except Exception:  # noqa: BLE001 - hint best-effort
                pass
            pairs = find_correlated_pairs_hint(
                sorted(corr_universe),
                window_days=getattr(profile, "correlation_window_days", 60),
                threshold=getattr(profile, "max_pair_correlation", 0.7),
            )
            if pairs:
                corr_lines = [
                    f"  {a}+{b}: {corr:+.2f}" for a, b, corr in pairs[:8]
                ]
                cap = getattr(
                    profile, "max_joint_correlated_weight_pct", 50.0
                )
                sections.append(
                    "## 3. Correlation hints (60d daily returns)\n"
                    + "\n".join(corr_lines)
                    + f"\nJoint weight of any pair above the correlation "
                    f"threshold cannot exceed {cap:.0f}% -- these are one "
                    "bet under the hood, not diversification. If you want "
                    "big exposure, pick the higher-conviction name."
                )
        except Exception:  # noqa: BLE001 - hint is best-effort
            pass

    sections.append(f"## 4. Signals (JSON, keyed by ticker)\n{signals}")

    prev_alloc = state.get("previous_allocation")
    if prev_alloc is not None:
        prev_lines = [
            f"  {p.ticker}: {p.weight_pct:.1f}%" for p in prev_alloc.positions
        ]
        prev_lines.append(f"  cash: {prev_alloc.cash_pct:.1f}%")
        sections.append(
            "## 5. Current allocation (your previous decision)\n"
            + "\n".join(prev_lines)
        )

    trades_block = _recent_trades_block(req.tickers, state.get("market_snapshots"))
    if trades_block:
        sections.append(
            "## 6. Your recent trades on these tickers (this session)\n"
            + trades_block
        )

    batch_ctx = (state.get("news_batch_context") or "").strip()
    if batch_ctx:
        sections.append(
            f"## 7. Breaking-news events (from live stream)\n{batch_ctx}"
        )

    volatile = "\n\n".join(sections) + "\n\nProduce a valid Allocation."

    # cache_control marker: Anthropic caches everything up to and including
    # the marker. OpenAI ignores the field and auto-caches by prefix match.
    system_msg = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": ALLOCATOR_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": USER_PREAMBLE,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": volatile},
        ],
    }
    return [system_msg, user_msg]


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


def repair(state: GraphState) -> dict:
    """Repair pass between allocate and validate.

    Applies position-count cap + cash floor so downstream rebalancing sees
    a target that respects the profile even when the LLM doesn't.
    """
    profile = _profile_from_state(state)
    repaired, notes = repair_allocation(state["allocation"], profile)
    if notes:
        logger.info("alloc_repair: %s", "; ".join(notes))
    return {"allocation": repaired}


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
    g.add_node("repair", repair)
    g.add_node("validate", validate)
    g.add_edge(START, "gather_signals")
    g.add_edge("gather_signals", "allocate")
    g.add_edge("allocate", "repair")
    g.add_edge("repair", "validate")
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
    effective_profile = profile if profile is not None else get_preset(request.risk)
    # Macro signal is best-effort: yfinance blips fall back to the untouched
    # profile so we never take a network hiccup as a reason to widen risk.
    macro_block = ""
    macro_label = "unknown"
    try:
        from agentic_investor.agents.macro import analyze_macro

        signal = analyze_macro()
        macro_label = signal.regime
        macro_block = signal.prompt_block
        effective_profile, notes = regime_adjusted_profile(
            effective_profile, macro_label
        )
        if notes:
            logger.info("regime_adjust: %s", "; ".join(notes))
    except Exception as e:  # noqa: BLE001
        logger.debug("macro signal skipped: %s", e)

    initial: dict = {"request": request, "profile": effective_profile}
    if macro_block:
        initial["macro_prompt_block"] = macro_block
    if macro_label:
        initial["macro_regime"] = macro_label
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
