"""Console entry point (agentic-investor).

  agentic-investor                                              show resolved config
  agentic-investor analyze AAPL NVDA                            run the technical agent
  agentic-investor recommend AAPL NVDA --amount 10000 --risk moderate
                                                                run the full orchestrator
  agentic-investor recommend --auto --universe dow30 --amount 10000
                                                                AI picks tickers itself
  agentic-investor backtest 1 --start 2024-01-01                backtest a recommendation vs SPY
  agentic-investor compare-strategies 1 --start 2024-01-01      same rec, all 3 presets side-by-side
  agentic-investor compare-allocators --tickers NVDA,TSLA --amount 10000
                                                                fresh rec per preset
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
        "Alpaca": s.alpaca_api_key,
    }
    for name, value in keys.items():
        print(f"{name}: {'set' if value else 'not set (mock mode)'}")


def _analyze(tickers: list[str], model: str | None) -> None:
    # Heavy imports (litellm) only when we actually analyze.
    from agentic_investor.agents.technical import analyze_ticker
    from agentic_investor.llm.client import format_call_stats, reset_call_stats

    reset_call_stats()
    for t in tickers:
        try:
            sig = analyze_ticker(t, model=model)
        except Exception as e:  # noqa: BLE001 - surface any provider/data error cleanly
            print(f"{t.upper()}: error - {e}\n")
            continue
        print(f"{sig.ticker}: {sig.stance} (confidence {sig.confidence:.2f})")
        print(f"  {sig.reasoning}\n")
    print(format_call_stats())


def _recommend(
    tickers: list[str],
    amount: float,
    risk: str,
    target: str,
    auto: bool,
    universe: str,
    top_n: int,
    exclude: list[str] | None,
    allocator: str | None,
    profile_name: str | None,
    rebalance: str | None,
    band_abs_pct: float | None,
    band_rel_pct: float | None,
    band_buy_multiplier: float | None,
    dd_buy_pause_pct: float | None,
    cash_yield: float | None,
    max_single_pct: float | None,
    cash_floor_pct: float | None,
    universe_extras: list[str] | None,
) -> None:
    from agentic_investor.llm.client import format_call_stats, reset_call_stats
    from agentic_investor.orchestrator.graph import run_orchestrator
    from agentic_investor.orchestrator.state import OrchestratorRequest
    from agentic_investor.orchestrator.store import save_recommendation
    from agentic_investor.orchestrator.strategy import (
        apply_overrides,
        get_preset,
        load_profile,
    )

    if auto and tickers:
        print("Cannot combine --auto with explicit tickers.")
        return
    if not auto and not tickers:
        print("Provide tickers, or use --auto to let the AI pick from a universe.")
        return

    reset_call_stats()

    # Resolve StrategyProfile: --profile OVERRIDES --risk if set; then CLI
    # per-dimension flags override profile fields.
    if profile_name is not None:
        profile = load_profile(profile_name)
    else:
        profile = get_preset(risk)  # type: ignore[arg-type]
    profile = apply_overrides(
        profile,
        allocator=allocator,
        rebalance=rebalance,
        band_abs_pct=band_abs_pct,
        band_rel_pct=band_rel_pct,
        band_buy_multiplier=band_buy_multiplier,
        dd_buy_pause_pct=dd_buy_pause_pct,
        cash_yield_annual=cash_yield,
        max_single_pct=max_single_pct,
        cash_floor_pct=cash_floor_pct,
        universe_extras=[t.upper() for t in universe_extras] if universe_extras else None,
    )

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
        tickers = [p.ticker for p in picks]
        # Profile-driven always-include diversifiers (e.g. TLT+GLD for conservative).
        added_extras = [e for e in profile.universe_extras if e not in tickers]
        if added_extras:
            print(f"\n  + profile extras (always include): {', '.join(added_extras)}")
            tickers.extend(added_extras)
        print()

    req = OrchestratorRequest(
        tickers=[t.upper() for t in tickers],
        amount=amount,
        risk=risk,  # type: ignore[arg-type]  argparse choices restrict values
        target=target,
    )
    rec = run_orchestrator(req, profile=profile)
    rec_id = save_recommendation(rec)

    print(
        f"Recommendation #{rec_id}  "
        f"(profile: {profile.name}, allocator: {profile.allocator}, "
        f"${req.amount:,.2f}, target: {req.target})\n"
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
    print("\n" + format_call_stats())


def _eval_report(
    rec_id: int | None,
    start: str | None,
    end: str | None,
    rebalance: str | None,
    cash_yield: float | None,
    tcost_bps: float | None,
    slippage_bps: float | None,
    n_samples: int,
    judge: bool,
    judge_model: str | None,
    k: int,
    out_dir: str,
    skip: list[str] | None,
    compare_to: str | None,
    profile_name: str | None,
) -> int:
    import sys

    from agentic_investor.eval.report import (
        compare_scorecards,
        load_baseline,
        run_full_eval,
        write_report,
    )
    from agentic_investor.llm.client import format_call_stats, reset_call_stats
    from agentic_investor.orchestrator.strategy import (
        StrategyProfile,
        apply_overrides,
        load_profile,
    )

    reset_call_stats()

    # Profile-driven L3 backtest defaults. Same pattern as _backtest.
    if profile_name is not None:
        profile = load_profile(profile_name)
    else:
        profile = StrategyProfile(name="cli-defaults")
    profile = apply_overrides(
        profile,
        rebalance=rebalance,
        cash_yield_annual=cash_yield,
        tcost_bps=tcost_bps,
        slippage_bps=slippage_bps,
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
        rebalance=profile.rebalance,
        cash_yield_annual=profile.cash_yield_annual,
        tcost_bps=profile.tcost_bps,
        slippage_bps=profile.slippage_bps,
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
    print()
    print(format_call_stats())

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
    from agentic_investor.llm.client import format_call_stats, reset_call_stats

    reset_call_stats()

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
    print()
    print(format_call_stats())


def _backtest(
    rec_id: int,
    start: str | None,
    end: str | None,
    benchmark: str,
    plot: str | None,
    rebalance: str | None,
    band_abs_pct: float | None,
    band_rel_pct: float | None,
    band_buy_multiplier: float | None,
    dd_buy_pause_pct: float | None,
    cash_yield: float | None,
    tcost_bps: float | None,
    slippage_bps: float | None,
    profile_name: str | None,
) -> None:
    from agentic_investor.eval.backtest import backtest_recommendation
    from agentic_investor.orchestrator.store import load_recommendation
    from agentic_investor.orchestrator.strategy import (
        StrategyProfile,
        apply_overrides,
        load_profile,
    )

    rec = load_recommendation(rec_id)
    if rec is None:
        print(f"No recommendation with id {rec_id}. Run `agentic-investor recommend ...` first.")
        return

    # Profile-driven backtest defaults. If no --profile, use a bare profile
    # with the historical hardcoded defaults so behavior stays unchanged for
    # commands that don't opt in.
    if profile_name is not None:
        profile = load_profile(profile_name)
    else:
        profile = StrategyProfile(name="cli-defaults")
    profile = apply_overrides(
        profile,
        rebalance=rebalance,
        band_abs_pct=band_abs_pct,
        band_rel_pct=band_rel_pct,
        band_buy_multiplier=band_buy_multiplier,
        dd_buy_pause_pct=dd_buy_pause_pct,
        cash_yield_annual=cash_yield,
        tcost_bps=tcost_bps,
        slippage_bps=slippage_bps,
    )

    result = backtest_recommendation(
        rec,
        start=start,
        end=end,
        benchmark=benchmark,
        rebalance=profile.rebalance,
        band_abs_pct=profile.band_abs_pct,
        band_rel_pct=profile.band_rel_pct,
        band_buy_multiplier=profile.band_buy_multiplier,
        dd_buy_pause_pct=profile.dd_buy_pause_pct,
        cash_yield_annual=profile.cash_yield_annual,
        tcost_bps=profile.tcost_bps,
        slippage_bps=profile.slippage_bps,
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
            rebalance=profile.rebalance,
            band_abs_pct=profile.band_abs_pct,
            band_rel_pct=profile.band_rel_pct,
            band_buy_multiplier=profile.band_buy_multiplier,
            dd_buy_pause_pct=profile.dd_buy_pause_pct,
            cash_yield_annual=profile.cash_yield_annual,
            tcost_bps=profile.tcost_bps,
            slippage_bps=profile.slippage_bps,
        )
        print(f"\n  Equity curve saved: {out}")


def _render_comparison_table(comparison) -> None:
    """Print a fixed-width comparison table to stdout."""
    print(
        f"Window: {comparison.start} to {comparison.end} "
        f"({comparison.n_days} bars). Init cash: ${comparison.init_cash:,.0f}.\n"
    )
    hdr = (
        f"{'Strategy':<44}{'Return%':>10}{'CAGR%':>8}{'Sharpe':>8}"
        f"{'MaxDD%':>9}{'Alpha%':>9}{'Beta':>7}{'Trades':>8}{'Cost$':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for e in comparison.entries:
        p = e.portfolio
        print(
            f"{e.label:<44}{p.total_return_pct:>10.1f}{p.cagr_pct:>8.1f}"
            f"{p.sharpe:>8.2f}{p.max_drawdown_pct:>9.1f}"
            f"{e.alpha_annual_pct:>+9.2f}{e.beta:>7.2f}"
            f"{e.n_trades:>8d}{e.total_costs:>10.2f}"
        )
    b = comparison.benchmark_metrics
    bench_label = f"{comparison.benchmark} (benchmark)"
    print(
        f"{bench_label:<44}{b.total_return_pct:>10.1f}{b.cagr_pct:>8.1f}"
        f"{b.sharpe:>8.2f}{b.max_drawdown_pct:>9.1f}"
        f"{'-':>9}{'1.00':>7}{'-':>8}{'-':>10}"
    )


def _compare_strategies(
    rec_id: int,
    start: str | None,
    end: str | None,
    benchmark: str,
    out: str | None,
) -> None:
    from agentic_investor.eval.backtest import (
        compare_strategies,
        render_comparison_markdown,
    )
    from agentic_investor.orchestrator.store import load_recommendation

    rec = load_recommendation(rec_id)
    if rec is None:
        print(f"No recommendation with id {rec_id}.")
        return

    comparison = compare_strategies(
        rec, start=start, end=end, benchmark=benchmark, rec_id=rec_id
    )
    print(f"\nStrategy comparison for Recommendation #{rec_id}")
    print("(same allocation, varies rebalance + friction only)\n")
    _render_comparison_table(comparison)

    if out is not None:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(render_comparison_markdown(comparison), encoding="utf-8")
        print(f"\nSaved markdown: {out}")


def _compare_allocators(
    tickers: list[str],
    amount: float,
    target: str,
    start: str | None,
    end: str | None,
    benchmark: str,
    out: str | None,
    no_baseline: bool,
    auto: bool,
    universe: str,
    top_n: int,
    exclude: list[str] | None,
    as_of: str | None,
) -> None:
    from agentic_investor.eval.backtest import (
        compare_allocators,
        render_comparison_markdown,
    )
    from agentic_investor.llm.client import format_call_stats, reset_call_stats
    from agentic_investor.orchestrator.state import OrchestratorRequest

    if auto and tickers:
        print("Cannot combine --auto with explicit --tickers.")
        return
    if not auto and not tickers:
        print("Provide --tickers or use --auto with --universe.")
        return

    reset_call_stats()

    if auto:
        from agentic_investor.orchestrator.picker import pick_top_n
        from agentic_investor.universes import get_universe

        # Point-in-time: if the caller specified --start but not --as-of,
        # default as_of=start so the picker can't see the future window.
        effective_as_of = as_of or start
        pool = get_universe(universe)
        excludes = {t.upper() for t in (exclude or [])}
        picks = pick_top_n(
            pool, top_n=top_n, exclude=excludes, as_of=effective_as_of
        )
        if not picks:
            print(f"No tickers passed the picker from universe '{universe}'.")
            return
        as_of_tag = f" as-of {effective_as_of}" if effective_as_of else ""
        print(
            f"\nAuto picker{as_of_tag}: {len(picks)} of "
            f"{len(pool) - len(excludes & set(pool))} scanned tickers "
            f"(universe '{universe}')\n"
        )
        print(f"  {'Ticker':<8}{'Score':>7}   Reasons")
        print("  " + "-" * 80)
        for p in picks:
            reasons = ", ".join(p.reasons[:4]) or "(no notable signals)"
            print(f"  {p.ticker:<8}{p.score:>7.2f}   {reasons}")
        tickers = [p.ticker for p in picks]
        print()

    request = OrchestratorRequest(
        tickers=tickers, amount=amount, risk="moderate", target=target
    )
    comparison = compare_allocators(
        request,
        start=start,
        end=end,
        benchmark=benchmark,
        include_baseline=not no_baseline,
    )
    print("\nFull-strategy comparison (each preset regenerates a fresh rec)\n")
    _render_comparison_table(comparison)

    if out is not None:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(render_comparison_markdown(comparison), encoding="utf-8")
        print(f"\nSaved markdown: {out}")

    print()
    print(format_call_stats())


# Paper trading (M7)


def _paper_status() -> None:
    from agentic_investor.tools.paper_broker import get_broker
    from agentic_investor.tools.paper_store import record_snapshot

    broker = get_broker()
    acct = broker.get_account()
    positions = broker.get_positions()
    record_snapshot(acct, positions)

    print(f"\nAlpaca paper account #{acct.account_number}")
    print(f"  Equity           ${acct.equity:>12,.2f}")
    print(f"  Cash             ${acct.cash:>12,.2f}")
    print(f"  Buying power     ${acct.buying_power:>12,.2f}")
    print(f"  Portfolio value  ${acct.portfolio_value:>12,.2f}\n")
    if not positions:
        print("  (no open positions)")
        return
    print(f"  {'Ticker':<8}{'Qty':>10}{'Avg entry':>12}{'Mkt value':>14}"
          f"{'Unrealized':>14}{'P&L %':>9}")
    print("  " + "-" * 65)
    for p in positions:
        print(f"  {p.ticker:<8}{p.qty:>10.2f}{p.avg_entry_price:>12.2f}"
              f"${p.market_value:>13,.2f}${p.unrealized_pl:>13,.2f}"
              f"{p.unrealized_pl_pct:>8.2f}%")


def _paper_orders(limit: int, status: str) -> None:
    from agentic_investor.tools.paper_broker import get_broker

    broker = get_broker()
    orders = broker.list_orders(limit=limit, status=status)
    if not orders:
        print(f"(no {status} orders)")
        return
    print(f"\n{len(orders)} recent orders ({status}):")
    print(f"  {'Ticker':<8}{'Side':>6}{'Qty':>10}{'Type':>10}"
          f"{'Status':>16}{'Filled@':>10}   Submitted")
    print("  " + "-" * 90)
    for o in orders:
        fill = f"{o.filled_avg_price:.2f}" if o.filled_avg_price else "-"
        print(f"  {o.ticker:<8}{o.side:>6}{o.qty:>10.2f}{o.order_type:>10}"
              f"{o.status:>16}{fill:>10}   {o.submitted_at}")


def _paper_submit(
    ticker: str,
    side: str,
    qty: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> None:
    from agentic_investor.tools.paper_broker import get_broker
    from agentic_investor.tools.paper_store import record_order

    broker = get_broker()
    order = broker.submit_market_order(
        ticker, side, qty,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
    )
    record_order(order, source="manual")
    print(f"\nSubmitted: {order.side} {order.qty} {order.ticker} ({order.order_type})")
    print(f"  Broker id       {order.id}")
    print(f"  Client order id {order.client_order_id}")
    print(f"  Status          {order.status}")


def _paper_rebalance(
    rec_id: int,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    min_trade_dollars: float,
    dry_run: bool,
) -> None:
    from agentic_investor.orchestrator.rebalancer import (
        compute_trade_plan,
        execute_trade_plan,
    )
    from agentic_investor.orchestrator.store import load_recommendation
    from agentic_investor.tools.market import fetch_ohlcv
    from agentic_investor.tools.paper_broker import get_broker
    from agentic_investor.tools.paper_store import record_order

    rec = load_recommendation(rec_id)
    if rec is None:
        print(f"No recommendation with id {rec_id}.")
        return

    broker = get_broker()
    acct = broker.get_account()
    positions = broker.get_positions()
    current_dollars = {p.ticker.upper(): p.market_value for p in positions}

    tickers = {p.ticker.upper() for p in rec.allocation.positions} | set(current_dollars)
    prices: dict[str, float] = {}
    for t in tickers:
        try:
            prices[t] = float(fetch_ohlcv(t, period="1y")["Close"].iloc[-1])
        except Exception as e:  # noqa: BLE001
            print(f"  warn: no price for {t}: {e}")

    plans = compute_trade_plan(
        rec, current_dollars, acct.equity,
        prices=prices, min_trade_dollars=min_trade_dollars,
    )

    print(f"\nRebalance plan for recommendation #{rec_id} "
          f"(equity ${acct.equity:,.2f}, min-trade ${min_trade_dollars:.0f})")
    if not plans:
        print("  Already on target - no trades needed.")
        return
    print(f"  {'Ticker':<8}{'Side':>6}{'Qty':>10}{'$Delta':>12}"
          f"{'Current%':>10}{'Target%':>10}   Reason")
    print("  " + "-" * 80)
    for p in plans:
        print(f"  {p.ticker:<8}{p.side:>6}{p.qty:>10.2f}${p.dollars:>11,.2f}"
              f"{p.current_pct:>10.2f}{p.target_pct:>10.2f}   {p.reason}")

    if dry_run:
        print("\n[dry-run] no orders submitted. Re-run without --dry-run to execute.")
        return

    orders = execute_trade_plan(
        plans, broker, rec_id=rec_id,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
    )
    for o in orders:
        record_order(o, source="rebalance", rec_id=rec_id)
    print(f"\nSubmitted {len(orders)} orders.")


def _parse_interval(text: str) -> int:
    """Turn "30m", "1h", "45" into seconds. Bare numbers are seconds."""
    t = text.strip().lower()
    if t.endswith("h"):
        return int(float(t[:-1]) * 3600)
    if t.endswith("m"):
        return int(float(t[:-1]) * 60)
    if t.endswith("s"):
        return int(float(t[:-1]))
    return int(float(t))


def _paper_test_event(
    tickers: list[str],
    headlines: list[str],
    profile_name: str,
    amount: float,
    dry_run: bool,
) -> None:
    """One synthetic decision moment - useful to verify event-driven wiring
    when market is closed or news is quiet. Builds a batch from CLI headlines,
    threads it through the allocator prompt, prints everything.
    """
    import logging
    from datetime import UTC, datetime

    from agentic_investor.llm.client import format_call_stats, reset_call_stats
    from agentic_investor.ops.session import SessionRecorder
    from agentic_investor.orchestrator.decision_engine import (
        DecisionBatch,
        TaggedNews,
        render_batch_context,
    )
    from agentic_investor.orchestrator.loop import LoopConfig, LoopState, run_tick
    from agentic_investor.tools.news_stream import NewsEvent
    from agentic_investor.tools.paper_broker import get_broker

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", force=True
    )
    session = SessionRecorder.start()
    print(f"\nSession artifacts -> {session.out_dir}\n")
    reset_call_stats()

    now = datetime.now(UTC)
    batch = DecisionBatch(fire_at=now.isoformat())
    # Distribute headlines across tickers (round-robin) so the batch sees
    # multi-ticker news like a realistic burst.
    for i, h in enumerate(headlines):
        ticker = tickers[i % len(tickers)]
        event = NewsEvent(
            ticker=ticker, headline=h, summary="",
            published_at=now.isoformat(), received_at=now.isoformat(),
        )
        batch.hot.append(TaggedNews(event=event, age_seconds=30.0, tag="HOT"))

    ctx = render_batch_context(batch)
    print("Synthetic news batch:")
    print(ctx)
    print()
    session.log("decision_moment", {"reason": "synthetic-test", **batch.summary()})

    cfg = LoopConfig(
        profile_name=profile_name, amount=amount,
        tickers=[t.upper() for t in tickers], auto=False,
        band_abs_pct=5.0, min_trade_dollars=1.0, dry_run=dry_run,
    )
    state = LoopState(pending_news_context=ctx)
    broker = get_broker()
    result = run_tick(cfg, state, broker, session=session, now=now)

    print(f"\nTick result: rec_id={result.rec_id} equity=${result.equity:,.2f} "
          f"plans={result.plan_count} submitted={len(result.submitted)}")
    session.finalize()
    print(f"\nSession summary: {session.summary_path}")
    print(format_call_stats())


def _paper_loop(
    profile_name: str,
    amount: float,
    tickers: list[str],
    auto: bool,
    universe: str,
    top_n: int,
    interval: str,
    band_abs_pct: float,
    min_trade_dollars: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    dry_run: bool,
    once: bool,
    log_file: str | None,
    regen_mode: str,
    force_open: bool,
) -> None:
    import logging

    from agentic_investor.llm.client import format_call_stats, reset_call_stats
    from agentic_investor.ops.session import SessionRecorder
    from agentic_investor.orchestrator.loop import (
        LoopConfig,
        format_session_summary,
        run_event_loop,
        run_loop,
    )
    from agentic_investor.tools.paper_broker import get_broker

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    session = SessionRecorder.start()
    print(f"\nSession artifacts -> {session.out_dir}\n")
    reset_call_stats()
    cfg = LoopConfig(
        profile_name=profile_name,
        amount=amount,
        tickers=tickers,
        auto=auto,
        universe=universe,
        top_n=top_n,
        interval_seconds=_parse_interval(interval),
        band_abs_pct=band_abs_pct,
        min_trade_dollars=min_trade_dollars,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        dry_run=dry_run,
        once=once,
        force_open=force_open,
    )
    broker = get_broker()
    try:
        if regen_mode == "event":
            state = run_event_loop(cfg, broker, session=session)
        else:
            state = run_loop(cfg, broker, session=session)
    finally:
        session.finalize()
    print(format_session_summary(state))
    print(format_call_stats())
    print(f"\nSession summary written: {session.summary_path}")


def _paper_tick(
    profile_name: str,
    amount: float,
    tickers: list[str],
    auto: bool,
    universe: str,
    top_n: int,
    band_abs_pct: float,
    min_trade_dollars: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    dry_run: bool,
) -> None:
    """Single-shot tick - useful for cron / manual runs / testing."""
    import logging

    from agentic_investor.llm.client import format_call_stats, reset_call_stats
    from agentic_investor.orchestrator.loop import LoopConfig, LoopState, run_tick
    from agentic_investor.tools.paper_broker import get_broker

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    reset_call_stats()

    cfg = LoopConfig(
        profile_name=profile_name, amount=amount, tickers=tickers,
        auto=auto, universe=universe, top_n=top_n,
        band_abs_pct=band_abs_pct, min_trade_dollars=min_trade_dollars,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        dry_run=dry_run, once=True,
    )
    broker = get_broker()
    state = LoopState()
    result = run_tick(cfg, state, broker)
    tag = " (fresh rec)" if result.regenerated_rec else ""
    print(f"\nTick{tag}: rec_id={result.rec_id} equity=${result.equity:,.2f} "
          f"plans={result.plan_count} submitted={len(result.submitted)}")
    print(format_call_stats())


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
    rec.add_argument("--allocator",
                     choices=["llm", "equal_weight", "inverse_vol"],
                     default=None,
                     help="override profile's allocator (default: preset picks it)")
    rec.add_argument("--profile", default=None,
                     help="preset name (conservative/moderate/aggressive) "
                          "or path to a TOML profile file")
    rec.add_argument("--rebalance", default=None,
                     choices=["never", "monthly", "quarterly", "bands", "on_signal"])
    rec.add_argument("--band-abs-pct", type=float, default=None)
    rec.add_argument("--band-rel-pct", type=float, default=None)
    rec.add_argument("--band-buy-multiplier", type=float, default=None,
                     help="asymmetric widen on the buy side (>1 = stingy)")
    rec.add_argument("--dd-buy-pause-pct", type=float, default=None,
                     help="pause BUY trades when portfolio DD deeper than this %%")
    rec.add_argument("--cash-yield", type=float, default=None,
                     help="annual risk-free yield on cash (e.g. 0.045)")
    rec.add_argument("--max-single-pct", type=float, default=None)
    rec.add_argument("--cash-floor-pct", type=float, default=None)
    rec.add_argument("--universe-extras", nargs="*", default=None,
                     help="always-include tickers appended after --auto pick "
                          "(e.g. --universe-extras TLT GLD)")

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
    rp.add_argument("--profile", default=None,
                    help="preset name or TOML profile path; supplies backtest defaults")
    rp.add_argument("--rebalance",
                    choices=["never", "monthly", "quarterly", "bands"], default=None)
    rp.add_argument("--cash-yield", type=float, default=None)
    rp.add_argument("--tcost-bps", type=float, default=None)
    rp.add_argument("--slippage-bps", type=float, default=None)
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
    bt.add_argument("--profile", default=None,
                    help="preset name (conservative/moderate/aggressive) OR path to "
                         "TOML profile file; supplies backtest defaults, CLI flags "
                         "override individual fields")
    bt.add_argument(
        "--rebalance",
        choices=["never", "monthly", "quarterly", "bands"],
        default=None,
        help="rebalance mode (default: profile or 'never')",
    )
    bt.add_argument("--band-abs-pct", type=float, default=None,
                    help="drift threshold in percentage points for --rebalance bands")
    bt.add_argument("--band-rel-pct", type=float, default=None,
                    help="relative drift threshold %% of target for bands mode")
    bt.add_argument("--band-buy-multiplier", type=float, default=None,
                    help="widen underweight-drift thresholds (>1 = stingy buy-back)")
    bt.add_argument("--dd-buy-pause-pct", type=float, default=None,
                    help="skip BUY trades when portfolio DD deeper than this %%")
    bt.add_argument("--cash-yield", type=float, default=None,
                    help="annual risk-free yield on cash, e.g. 0.045 for 4.5%%")
    bt.add_argument("--tcost-bps", type=float, default=None,
                    help="commission in basis points per trade")
    bt.add_argument("--slippage-bps", type=float, default=None,
                    help="slippage in basis points per fill")

    cs = sub.add_parser(
        "compare-strategies",
        help="backtest one recommendation under baseline + 3 presets "
             "(same allocation, varies rebalance/friction)",
    )
    cs.add_argument("rec_id", type=int)
    cs.add_argument("--start", default=None, help="YYYY-MM-DD")
    cs.add_argument("--end", default=None, help="YYYY-MM-DD")
    cs.add_argument("--benchmark", default="SPY")
    cs.add_argument("--out", default=None,
                    help="also save the comparison table as markdown to this path")

    ca = sub.add_parser(
        "compare-allocators",
        help="regenerate a fresh recommendation under each preset "
             "(varies allocator too), then backtest each; costs 3x LLM calls",
    )
    ca.add_argument("--tickers", default=None,
                    help="comma-separated tickers (e.g. NVDA,TSLA,AAPL); "
                         "omit and use --auto to let the picker choose")
    ca.add_argument("--amount", type=float, required=True)
    ca.add_argument("--target", default="12-month growth")
    ca.add_argument("--start", default=None, help="YYYY-MM-DD")
    ca.add_argument("--end", default=None, help="YYYY-MM-DD")
    ca.add_argument("--benchmark", default="SPY")
    ca.add_argument("--out", default=None,
                    help="also save the comparison table as markdown to this path")
    ca.add_argument("--no-baseline", action="store_true",
                    help="skip the default-preset baseline row (3 runs instead of 4)")
    ca.add_argument("--auto", action="store_true",
                    help="let the M5 picker choose tickers from --universe "
                         "(same list passed to every preset)")
    ca.add_argument("--universe", default="mega_tech",
                    help="universe key for --auto (dow30 / mega_tech / sectors)")
    ca.add_argument("--top-n", type=int, default=8,
                    help="how many tickers to pick when --auto is set")
    ca.add_argument("--exclude", nargs="*", default=None,
                    help="tickers to skip during --auto scan")
    ca.add_argument("--as-of", default=None,
                    help="YYYY-MM-DD; picker uses prices only up to this date "
                         "(defaults to --start if omitted, eliminating look-ahead)")

    # Paper trading (M7)
    sub.add_parser("paper-status",
                   help="show Alpaca paper account balance + open positions")

    po = sub.add_parser("paper-orders", help="list recent paper orders from Alpaca")
    po.add_argument("--limit", type=int, default=20)
    po.add_argument("--status", default="all",
                    choices=["all", "open", "closed"])

    ps = sub.add_parser("paper-submit",
                        help="submit a manual paper market order (sanity check)")
    ps.add_argument("ticker")
    ps.add_argument("side", choices=["buy", "sell"])
    ps.add_argument("qty", type=float)
    ps.add_argument("--stop-loss-pct", type=float, default=None,
                    help="attach a stop-loss leg at this %% below entry "
                         "(bracket order; buys only)")
    ps.add_argument("--take-profit-pct", type=float, default=None,
                    help="attach a take-profit leg at this %% above entry "
                         "(bracket order; buys only)")

    pr = sub.add_parser("paper-rebalance",
                        help="turn a saved recommendation into paper orders "
                             "(computes diff vs current positions)")
    pr.add_argument("rec_id", type=int)
    pr.add_argument("--stop-loss-pct", type=float, default=None)
    pr.add_argument("--take-profit-pct", type=float, default=None)
    pr.add_argument("--min-trade-dollars", type=float, default=25.0,
                    help="skip trades smaller than this to avoid churn")
    pr.add_argument("--dry-run", action="store_true",
                    help="print the plan but submit nothing")

    pt = sub.add_parser("paper-tick",
                        help="single loop iteration (regenerate rec if new day, "
                             "rebalance if drift > band); useful for cron/manual")
    pt.add_argument("--profile", default="moderate")
    pt.add_argument("--amount", type=float, default=10_000.0)
    pt.add_argument("--tickers", default=None,
                    help="comma-separated tickers; omit if using --auto")
    pt.add_argument("--auto", action="store_true")
    pt.add_argument("--universe", default="mega_tech")
    pt.add_argument("--top-n", type=int, default=8)
    pt.add_argument("--band-abs-pct", type=float, default=5.0)
    pt.add_argument("--min-trade-dollars", type=float, default=50.0)
    pt.add_argument("--stop-loss-pct", type=float, default=None)
    pt.add_argument("--take-profit-pct", type=float, default=None)
    pt.add_argument("--dry-run", action="store_true")

    pl = sub.add_parser("paper-loop",
                        help="continuous paper-trading loop during market hours; "
                             "regenerates rec once per day, ticks at --interval")
    pl.add_argument("--profile", default="moderate")
    pl.add_argument("--amount", type=float, default=10_000.0)
    pl.add_argument("--tickers", default=None,
                    help="comma-separated tickers; omit if using --auto")
    pl.add_argument("--auto", action="store_true")
    pl.add_argument("--universe", default="mega_tech")
    pl.add_argument("--top-n", type=int, default=8)
    pl.add_argument("--interval", default="30m",
                    help="tick cadence; accepts 30m / 1h / 45s (default 30m)")
    pl.add_argument("--band-abs-pct", type=float, default=5.0)
    pl.add_argument("--min-trade-dollars", type=float, default=50.0)
    pl.add_argument("--stop-loss-pct", type=float, default=None)
    pl.add_argument("--take-profit-pct", type=float, default=None)
    pl.add_argument("--dry-run", action="store_true",
                    help="every tick prints its plan but submits nothing")
    pl.add_argument("--once", action="store_true",
                    help="one tick then exit (great for testing)")
    pl.add_argument("--log-file", default="out/paper_loop.log",
                    help="also stream logs to this file (default out/paper_loop.log)")
    pl.add_argument("--regen-mode", default="daily",
                    choices=["daily", "event"],
                    help="daily: regen rec once at open. event: subscribe to "
                         "alpaca news, fire on decision moments (micro-batched)")
    pl.add_argument("--force-open", action="store_true",
                    help="skip market-hours check (dev/testing only - orders "
                         "will queue at Alpaca for next open)")

    pt2 = sub.add_parser("paper-test-event",
                         help="synthetic decision moment: inject fake news, "
                              "run one full tick, verify event-driven wiring")
    pt2.add_argument("--tickers", required=True,
                     help="comma-separated tickers to build the batch for")
    pt2.add_argument("--headlines", nargs="+", required=True,
                     help="one or more fake headlines (distributed across tickers)")
    pt2.add_argument("--profile", default="moderate")
    pt2.add_argument("--amount", type=float, default=10_000.0)
    pt2.add_argument("--dry-run", action="store_true",
                     help="print the plan but submit nothing")

    args = parser.parse_args()
    if args.cmd == "analyze":
        _analyze(args.tickers, args.model)
    elif args.cmd == "recommend":
        _recommend(
            args.tickers, args.amount, args.risk, args.target,
            args.auto, args.universe, args.top_n, args.exclude,
            args.allocator, args.profile,
            args.rebalance, args.band_abs_pct, args.band_rel_pct,
            args.band_buy_multiplier, args.dd_buy_pause_pct,
            args.cash_yield, args.max_single_pct, args.cash_floor_pct,
            args.universe_extras,
        )
    elif args.cmd == "eval-agents":
        _eval_agents(args.n_samples, args.model, args.cases, args.judge, args.judge_model)
    elif args.cmd == "eval-retrieval":
        _eval_retrieval(args.k, args.fixtures, args.cases)
    elif args.cmd == "eval-report":
        _eval_report(
            args.rec_id, args.start, args.end, args.rebalance,
            args.cash_yield, args.tcost_bps, args.slippage_bps,
            args.n_samples, args.judge, args.judge_model, args.k,
            args.out_dir, args.skip, args.compare_to, args.profile,
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
            args.profile,
        )
    elif args.cmd == "compare-strategies":
        _compare_strategies(
            args.rec_id, args.start, args.end, args.benchmark, args.out
        )
    elif args.cmd == "compare-allocators":
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers else []
        )
        _compare_allocators(
            tickers, args.amount, args.target,
            args.start, args.end, args.benchmark, args.out, args.no_baseline,
            args.auto, args.universe, args.top_n, args.exclude, args.as_of,
        )
    elif args.cmd == "paper-status":
        _paper_status()
    elif args.cmd == "paper-orders":
        _paper_orders(args.limit, args.status)
    elif args.cmd == "paper-submit":
        _paper_submit(
            args.ticker.upper(), args.side, args.qty,
            args.stop_loss_pct, args.take_profit_pct,
        )
    elif args.cmd == "paper-rebalance":
        _paper_rebalance(
            args.rec_id, args.stop_loss_pct, args.take_profit_pct,
            args.min_trade_dollars, args.dry_run,
        )
    elif args.cmd == "paper-tick":
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers else []
        )
        _paper_tick(
            args.profile, args.amount, tickers, args.auto, args.universe,
            args.top_n, args.band_abs_pct, args.min_trade_dollars,
            args.stop_loss_pct, args.take_profit_pct, args.dry_run,
        )
    elif args.cmd == "paper-test-event":
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        _paper_test_event(
            tickers, args.headlines, args.profile, args.amount, args.dry_run,
        )
    elif args.cmd == "paper-loop":
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers else []
        )
        _paper_loop(
            args.profile, args.amount, tickers, args.auto, args.universe,
            args.top_n, args.interval, args.band_abs_pct, args.min_trade_dollars,
            args.stop_loss_pct, args.take_profit_pct, args.dry_run, args.once,
            args.log_file, args.regen_mode, args.force_open,
        )
    else:
        _print_config()


if __name__ == "__main__":
    main()
