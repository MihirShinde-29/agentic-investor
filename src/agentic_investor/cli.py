"""Console entry point (agentic-investor).

  agentic-investor                                              show resolved config
  agentic-investor analyze AAPL NVDA                            run the technical agent
  agentic-investor recommend AAPL NVDA --amount 10000 --risk moderate
                                                                run the full orchestrator
  agentic-investor recommend --auto --universe dow30 --amount 10000
                                                                AI picks tickers itself
  agentic-investor backtest 1 --start 2024-01-01                backtest a recommendation vs SPY
  agentic-investor eval-agents --n-samples 3                    L2 agent eval vs golden set
  agentic-investor eval-retrieval --k 5                         L1 RAG retrieval eval
  agentic-investor eval-report --rec-id 1 --judge               aggregate to REPORT.md
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


def _recommend(
    tickers: list[str],
    amount: float,
    risk: str,
    target: str,
    auto: bool,
    universe: str,
    top_n: int,
    exclude: list[str] | None,
) -> None:
    from agentic_investor.orchestrator.graph import run_orchestrator
    from agentic_investor.orchestrator.state import OrchestratorRequest
    from agentic_investor.orchestrator.store import save_recommendation

    if auto and tickers:
        print("Cannot combine --auto with explicit tickers.")
        return
    if not auto and not tickers:
        print("Provide tickers, or use --auto to let the AI pick from a universe.")
        return

    if auto:
        from agentic_investor.orchestrator.picker import pick_top_n
        from agentic_investor.universes import get_universe

        pool = get_universe(universe)
        excludes = {t.upper() for t in (exclude or [])}
        picks = pick_top_n(pool, top_n=top_n, exclude=excludes)
        if not picks:
            print(f"No tickers passed the picker from universe '{universe}'.")
            return
        print(
            f"\nAuto picker: {len(picks)} of {len(pool) - len(excludes & set(pool))} "
            f"scanned tickers (universe '{universe}')\n"
        )
        print(f"  {'Ticker':<8}{'Score':>7}   Reasons")
        print("  " + "-" * 80)
        for p in picks:
            reasons = ", ".join(p.reasons[:4]) or "(no notable signals)"
            print(f"  {p.ticker:<8}{p.score:>7.2f}   {reasons}")
        print()
        tickers = [p.ticker for p in picks]

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


def _eval_report(
    rec_id: int | None,
    start: str | None,
    end: str | None,
    rebalance: str,
    cash_yield: float,
    tcost_bps: float,
    slippage_bps: float,
    n_samples: int,
    judge: bool,
    judge_model: str | None,
    k: int,
    out_dir: str,
    skip: list[str] | None,
    compare_to: str | None,
) -> int:
    import sys

    from agentic_investor.eval.report import (
        compare_scorecards,
        load_baseline,
        run_full_eval,
        write_report,
    )

    resolved_judge_model = None
    if judge:
        from agentic_investor.config import get_settings
        resolved_judge_model = judge_model or get_settings().llm_model

    skip_set = set(s.upper() for s in (skip or []))
    report = run_full_eval(
        rec_id=rec_id,
        start=start,
        end=end,
        rebalance=rebalance,
        cash_yield_annual=cash_yield,
        tcost_bps=tcost_bps,
        slippage_bps=slippage_bps,
        n_samples=n_samples,
        judge_model=resolved_judge_model,
        k=k,
        skip=skip_set,
    )
    md_path, json_path = write_report(report, out_dir=out_dir)

    print(f"\nEval report  ({report.generated_at})\n")
    print(f"  {'Layer':<20}{'Status':<12}{'Summary'}")
    print("  " + "-" * 100)
    for v in report.verdicts:
        print(f"  {v.layer:<20}{v.status:<12}{v.summary}")
    print()
    print(f"  Wrote: {md_path}")
    print(f"  Wrote: {json_path}")

    if compare_to is not None:
        baseline = load_baseline(compare_to)
        result = compare_scorecards(report, baseline)
        print()
        if result.passed:
            print(f"  Regression check vs {compare_to}: PASSED (no metric dropped past tolerance)")
        else:
            print(f"  Regression check vs {compare_to}: FAILED")
            for r in result.regressions:
                print(f"    - {r}")
            sys.exit(1)
    return 0


def _eval_retrieval(k: int, fixtures: str | None, cases: str | None) -> None:
    from agentic_investor.eval.retrieval import (
        DEFAULT_CASES,
        DEFAULT_FIXTURES,
        run_retrieval_eval,
    )

    report = run_retrieval_eval(
        k=k,
        fixtures_path=fixtures or DEFAULT_FIXTURES,
        cases_path=cases or DEFAULT_CASES,
    )

    print(
        f"\nRetrieval eval  (news RAG, k={report.k}, "
        f"{report.n_cases} cases over {report.n_fixtures} fixtures)\n"
    )
    print(
        f"  {'Case':<26}{'Hit@k':>7}{'Recall@k':>10}"
        f"{'MRR':>7}{'NDCG@k':>9}   {'Retrieved (top-3)'}"
    )
    print("  " + "-" * 90)
    for cr in report.per_case:
        m = cr.metrics
        top3 = ", ".join(cr.retrieved_ids[:3]) if cr.retrieved_ids else "-"
        print(
            f"  {cr.case_id:<26}{m.hit_at_k:>7.2f}{m.recall_at_k:>10.2f}"
            f"{m.mrr:>7.2f}{m.ndcg_at_k:>9.2f}   {top3}"
        )
    print()
    a = report.aggregate
    print(f"  Aggregate  Hit@k={a.hit_at_k:.3f}  Recall@k={a.recall_at_k:.3f}"
          f"  MRR={a.mrr:.3f}  NDCG@k={a.ndcg_at_k:.3f}")


def _eval_agents(
    n_samples: int,
    model: str | None,
    cases: str | None,
    judge: bool,
    judge_model: str | None,
) -> None:
    from agentic_investor.eval.agents import DEFAULT_CASES, run_agent_eval

    resolved_judge_model = None
    if judge:
        resolved_judge_model = judge_model or model
        if resolved_judge_model is None:
            # Both are None -> use configured default via the LLM client.
            from agentic_investor.config import get_settings
            resolved_judge_model = get_settings().llm_model

    report = run_agent_eval(
        cases_path=cases or DEFAULT_CASES,
        n_samples=n_samples,
        model=model,
        judge_model=resolved_judge_model,
    )

    print(
        f"\nAgent eval  (technical, N={report.n_samples_per_case} samples/case, "
        f"{report.n_cases} cases)\n"
    )
    hdr = f"  {'Case':<28}{'Schema':>8}{'Stance':>8}{'AvgConf':>9}{'ConfOK':>8}"
    if report.avg_rationale_overall is not None:
        hdr += f"{'Rationale':>11}"
    hdr += f"   {'Stances seen'}"
    print(hdr)
    print("  " + "-" * 100)
    for r in report.per_case:
        seen = ",".join(dict.fromkeys(r.stances_seen)) or "-"
        row = (
            f"  {r.case_id:<28}"
            f"{r.schema_validity:>8.2f}{r.stance_pass_rate:>8.2f}"
            f"{r.avg_confidence:>9.2f}{'YES' if r.confidence_in_range else 'no':>8}"
        )
        if report.avg_rationale_overall is not None:
            rat = f"{r.avg_rationale_overall:.2f}/5" if r.avg_rationale_overall else "-"
            row += f"{rat:>11}"
        row += f"   {seen}"
        print(row)
    print()
    print(f"  Aggregate schema validity: {report.schema_validity:.2%}")
    print(f"  Aggregate stance pass:     {report.stance_pass_rate:.2%}")
    print(f"  Confidence-in-range rate:  {report.confidence_in_range_rate:.2%}")
    print(f"  Fully passing cases:       {report.fully_passing_cases}/{report.n_cases}")
    if report.avg_rationale_overall is not None:
        print(f"  Avg rationale overall:     {report.avg_rationale_overall:.2f}/5"
              f"  (judge model: {report.judge_model})")


def _backtest(
    rec_id: int,
    start: str | None,
    end: str | None,
    benchmark: str,
    plot: str | None,
    rebalance: str,
    band_abs_pct: float,
    band_rel_pct: float,
    band_buy_multiplier: float,
    dd_buy_pause_pct: float,
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
        band_buy_multiplier=band_buy_multiplier,
        dd_buy_pause_pct=dd_buy_pause_pct,
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
            band_buy_multiplier=band_buy_multiplier,
            dd_buy_pause_pct=dd_buy_pause_pct,
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
    rec.add_argument("tickers", nargs="*",
                     help="tickers to allocate over (omit and use --auto for AI picking)")
    rec.add_argument("--amount", type=float, required=True, help="portfolio dollars to allocate")
    rec.add_argument(
        "--risk",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
    )
    rec.add_argument("--target", default="12-month growth")
    rec.add_argument("--auto", action="store_true",
                     help="AI picks tickers itself via rules-based Selector")
    rec.add_argument("--universe", default="dow30",
                     help="universe name for --auto (dow30, sectors, mega_tech)")
    rec.add_argument("--top-n", type=int, default=5,
                     help="how many top-scored tickers to keep (default 5)")
    rec.add_argument("--exclude", nargs="*", default=None,
                     help="tickers to skip in --auto mode")

    ev = sub.add_parser("eval-agents", help="run L2 agent output evals vs golden set")
    ev.add_argument("--n-samples", type=int, default=3,
                    help="samples per case (default 3)")
    ev.add_argument("--model", default=None, help="override LLM model string")
    ev.add_argument("--cases", default=None,
                    help="path to jsonl cases file (default: bundled golden set)")
    ev.add_argument("--judge", action="store_true",
                    help="also grade rationale quality via LLM-as-judge")
    ev.add_argument("--judge-model", default=None,
                    help="model for judge (default: same as agent; use a stronger model "
                         "like gpt-4o to avoid self-preference bias)")

    er = sub.add_parser("eval-retrieval", help="run L1 RAG retrieval evals on news golden set")
    er.add_argument("--k", type=int, default=5, help="top-k for retrieval (default 5)")
    er.add_argument("--fixtures", default=None,
                    help="path to jsonl fixture articles (default: bundled)")
    er.add_argument("--cases", default=None,
                    help="path to jsonl retrieval cases (default: bundled)")

    rp = sub.add_parser("eval-report",
                        help="run L1+L2+L3, write REPORT.md + scorecard.json")
    rp.add_argument("--rec-id", type=int, default=None,
                    help="recommendation id for L3 backtest (skip L3 if omitted)")
    rp.add_argument("--start", default=None, help="backtest start YYYY-MM-DD")
    rp.add_argument("--end", default=None, help="backtest end YYYY-MM-DD")
    rp.add_argument("--rebalance",
                    choices=["never", "monthly", "quarterly", "bands"], default="never")
    rp.add_argument("--cash-yield", type=float, default=0.0)
    rp.add_argument("--tcost-bps", type=float, default=0.0)
    rp.add_argument("--slippage-bps", type=float, default=0.0)
    rp.add_argument("--n-samples", type=int, default=3,
                    help="samples per agent case (default 3)")
    rp.add_argument("--judge", action="store_true", help="run LLM-as-judge on rationales")
    rp.add_argument("--judge-model", default=None, help="model for judge")
    rp.add_argument("--k", type=int, default=5, help="top-k for retrieval")
    rp.add_argument("--out-dir", default="out/eval", help="output directory")
    rp.add_argument("--skip", nargs="*", default=None, choices=["L1", "L2", "L3"],
                    help="skip named layers, e.g. --skip L3")
    rp.add_argument("--compare-to", default=None,
                    help="path to a baseline scorecard.json; exit 1 on regression")

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
    bt.add_argument("--band-buy-multiplier", type=float, default=1.0,
                    help="widen underweight-drift thresholds (>1 = stingy buy-back; default 1)")
    bt.add_argument("--dd-buy-pause-pct", type=float, default=0.0,
                    help="skip BUY trades when portfolio DD deeper than this %% (default 0=off)")
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
        _recommend(args.tickers, args.amount, args.risk, args.target,
                   args.auto, args.universe, args.top_n, args.exclude)
    elif args.cmd == "eval-agents":
        _eval_agents(args.n_samples, args.model, args.cases, args.judge, args.judge_model)
    elif args.cmd == "eval-retrieval":
        _eval_retrieval(args.k, args.fixtures, args.cases)
    elif args.cmd == "eval-report":
        _eval_report(
            args.rec_id, args.start, args.end, args.rebalance,
            args.cash_yield, args.tcost_bps, args.slippage_bps,
            args.n_samples, args.judge, args.judge_model, args.k,
            args.out_dir, args.skip, args.compare_to,
        )
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
            args.band_buy_multiplier,
            args.dd_buy_pause_pct,
            args.cash_yield,
            args.tcost_bps,
            args.slippage_bps,
        )
    else:
        _print_config()


if __name__ == "__main__":
    main()
