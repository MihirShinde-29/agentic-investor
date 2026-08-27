"""Console entry point (agentic-investor).

  agentic-investor                                              show resolved config
  agentic-investor analyze AAPL NVDA                            run the technical agent
  agentic-investor recommend AAPL NVDA --amount 10000 --risk moderate
                                                                run the full orchestrator
  agentic-investor backtest 1 --start 2024-01-01                backtest a recommendation vs SPY
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


def _backtest(
    rec_id: int,
    start: str | None,
    end: str | None,
    benchmark: str,
    plot: str | None,
    rebalance: str,
    band_abs_pct: float,
    band_rel_pct: float,
    cash_yield: float,
    tcost_bps: float,
    slippage_bps: float,
) -> None:
    from agentic_investor.eval.backtest import backtest_recommendation
    from agentic_investor.orchestrator.store import load_recommendation

    rec = load_recommendation(rec_id)
    if rec is None:
        print(f"No recommendation with id {rec_id}. Run `agentic-investor recommend ...` first.")
        return
    result = backtest_recommendation(
        rec,
        start=start,
        end=end,
        benchmark=benchmark,
        rebalance=rebalance,
        band_abs_pct=band_abs_pct,
        band_rel_pct=band_rel_pct,
        cash_yield_annual=cash_yield,
        tcost_bps=tcost_bps,
        slippage_bps=slippage_bps,
    )

    print(
        f"Backtest of recommendation #{rec_id}  ({result.start} -> {result.end}, "
        f"{result.n_days} bars, ${result.init_cash:,.0f} init)\n"
    )
    print(f"  {'':16}{'Portfolio':>12}   {benchmark:>10}")
    for label, key in [
        ("Total return %", "total_return_pct"),
        ("CAGR %", "cagr_pct"),
        ("Sharpe", "sharpe"),
        ("Max drawdown %", "max_drawdown_pct"),
        ("Volatility %", "volatility_annual_pct"),
    ]:
        p = getattr(result.portfolio, key)
        b = getattr(result.benchmark, key)
        print(f"  {label:16}{p:>12.2f}   {b:>10.2f}")
    print(f"\n  Alpha (annualized): {result.alpha_annual_pct:+.2f}%")
    print(f"  Beta vs {benchmark}:       {result.beta:.3f}")
    print(
        f"\n  Final value:   ${result.portfolio_final_value:>12,.2f}   "
        f"(vs {benchmark}: ${result.benchmark_final_value:,.2f})"
    )
    if result.n_trades:
        print(f"  Trades: {result.n_trades}   Total friction: ${result.total_costs:,.2f}")

    if plot is not None:
        from agentic_investor.eval.plots import plot_equity_curve

        out = plot_equity_curve(
            rec,
            start=start,
            end=end,
            benchmark=benchmark,
            out_path=plot,
            rec_id=rec_id,
            rebalance=rebalance,
            band_abs_pct=band_abs_pct,
            band_rel_pct=band_rel_pct,
            cash_yield_annual=cash_yield,
            tcost_bps=tcost_bps,
            slippage_bps=slippage_bps,
        )
        print(f"\n  Equity curve saved: {out}")


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

    bt = sub.add_parser("backtest", help="backtest a saved recommendation vs a benchmark")
    bt.add_argument("rec_id", type=int)
    bt.add_argument("--start", default=None, help="YYYY-MM-DD")
    bt.add_argument("--end", default=None, help="YYYY-MM-DD")
    bt.add_argument("--benchmark", default="SPY")
    bt.add_argument(
        "--plot",
        nargs="?",
        const="out/equity_curve.png",
        default=None,
        help="save equity curve PNG (default: out/equity_curve.png)",
    )
    bt.add_argument(
        "--rebalance",
        choices=["never", "monthly", "quarterly", "bands"],
        default="never",
        help="rebalance mode (default: never = buy-and-hold)",
    )
    bt.add_argument("--band-abs-pct", type=float, default=5.0,
                    help="drift threshold in percentage points for --rebalance bands (default 5)")
    bt.add_argument("--band-rel-pct", type=float, default=20.0,
                    help="relative drift threshold %% of target for bands mode (default 20)")
    bt.add_argument("--cash-yield", type=float, default=0.0,
                    help="annual risk-free yield on cash, e.g. 0.045 for 4.5%% (default 0)")
    bt.add_argument("--tcost-bps", type=float, default=0.0,
                    help="commission in basis points per trade (default 0)")
    bt.add_argument("--slippage-bps", type=float, default=0.0,
                    help="slippage in basis points per fill (default 0)")

    args = parser.parse_args()
    if args.cmd == "analyze":
        _analyze(args.tickers, args.model)
    elif args.cmd == "recommend":
        _recommend(args.tickers, args.amount, args.risk, args.target)
    elif args.cmd == "backtest":
        _backtest(
            args.rec_id,
            args.start,
            args.end,
            args.benchmark,
            args.plot,
            args.rebalance,
            args.band_abs_pct,
            args.band_rel_pct,
            args.cash_yield,
            args.tcost_bps,
            args.slippage_bps,
        )
    else:
        _print_config()


if __name__ == "__main__":
    main()
