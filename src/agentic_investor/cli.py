"""Console entry point (agentic-investor).

  agentic-investor                                              show resolved config
  agentic-investor analyze AAPL NVDA                            run the technical agent
  agentic-investor recommend AAPL NVDA --amount 10000 --risk moderate
                                                                run the full orchestrator
"""

import argparse

from agentic_investor.config import get_settings


def _print_config() -> None:
    s = get_settings()
    print("Agentic Investor scaffold OK\n")
    print(f"LLM model: {s.llm_model}")
    print(f"Orchestrator: {s.orchestrator_model}")
    print(f"Embedding model: {s.embedding_model}\n")

    keys = {
        "OpenAI": s.openai_api_key,
        "Anthropic": s.anthropic_api_key,
        "Gemini": s.gemini_api_key,
        "Finnhub": s.finnhub_api_key,
    }
    for name, value in keys.items():
        print(f"{name}: {'set' if value else 'not set (mock mode)'}")


def _analyze(tickers: list[str], model: str | None) -> None:
    # Heavy imports (litellm) only when we actually analyze.
    from agentic_investor.agents.technical import analyze_ticker

    for t in tickers:
        try:
            sig = analyze_ticker(t, model=model)
        except Exception as e:  # noqa: BLE001 - surface any provider/data error cleanly
            print(f"{t.upper()}: error - {e}\n")
            continue
        print(f"{sig.ticker}: {sig.stance} (confidence {sig.confidence:.2f})")
        print(f"  {sig.reasoning}\n")


def _recommend(tickers: list[str], amount: float, risk: str, target: str) -> None:
    from agentic_investor.orchestrator.graph import run_orchestrator
    from agentic_investor.orchestrator.state import OrchestratorRequest
    from agentic_investor.orchestrator.store import save_recommendation

    req = OrchestratorRequest(
        tickers=[t.upper() for t in tickers],
        amount=amount,
        risk=risk,  # type: ignore[arg-type]  argparse choices restrict values
        target=target,
    )
    rec = run_orchestrator(req)
    rec_id = save_recommendation(rec)

    print(
        f"Recommendation #{rec_id}  "
        f"(risk: {req.risk}, ${req.amount:,.2f}, target: {req.target})\n"
    )
    for p in rec.allocation.positions:
        print(f"  {p.ticker:6} {p.weight_pct:5.1f}%  ${p.dollars:>10,.2f}   {p.rationale}")
    print(
        f"  {'CASH':6} {rec.allocation.cash_pct:5.1f}%  "
        f"${rec.allocation.cash_dollars:>10,.2f}"
    )
    print(f"\n{rec.allocation.portfolio_rationale}")
    if rec.violations:
        print("\nGuardrail violations:")
        for v in rec.violations:
            print(f"  - {v}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentic-investor")
    sub = parser.add_subparsers(dest="cmd")

    analyze = sub.add_parser("analyze", help="run the technical agent on tickers")
    analyze.add_argument("tickers", nargs="+")
    analyze.add_argument("--model", default=None, help="override the LLM model string")

    rec = sub.add_parser(
        "recommend", help="run full pipeline; persist a paper portfolio recommendation"
    )
    rec.add_argument("tickers", nargs="+")
    rec.add_argument("--amount", type=float, required=True, help="portfolio dollars to allocate")
    rec.add_argument(
        "--risk",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
    )
    rec.add_argument("--target", default="12-month growth")

    args = parser.parse_args()
    if args.cmd == "analyze":
        _analyze(args.tickers, args.model)
    elif args.cmd == "recommend":
        _recommend(args.tickers, args.amount, args.risk, args.target)
    else:
        _print_config()


if __name__ == "__main__":
    main()
