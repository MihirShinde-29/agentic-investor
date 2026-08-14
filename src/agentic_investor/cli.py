"""Console entry point (agentic-investor).

  agentic-investor                 show resolved config
  agentic-investor analyze AAPL NVDA   run the technical agent on tickers
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentic-investor")
    sub = parser.add_subparsers(dest="cmd")
    analyze = sub.add_parser("analyze", help="run the technical agent on tickers")
    analyze.add_argument("tickers", nargs="+")
    analyze.add_argument("--model", default=None, help="override the LLM model string")

    args = parser.parse_args()
    if args.cmd == "analyze":
        _analyze(args.tickers, args.model)
    else:
        _print_config()


if __name__ == "__main__":
    main()
