"""M4 report aggregator: run L1 + L2 + L3, write REPORT.md + scorecard.json.

Turns the three sub-evals into one artifact humans can read and CI can gate on.
Verdicts per layer use conservative thresholds (retrieval NDCG >= 0.7 = PASS,
agent stance+schema >= gates PASS, backtest is REPORTED but never gated - the
eval spec's honesty guardrail says market-dependent metrics shouldn't be
pass/fail).
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agentic_investor.eval.agents import AgentEvalReport, run_agent_eval
from agentic_investor.eval.backtest import BacktestResult, backtest_recommendation
from agentic_investor.eval.retrieval import RetrievalEvalReport, run_retrieval_eval
from agentic_investor.orchestrator.store import load_recommendation

Status = Literal["PASSING", "PARTIAL", "FAILING", "REPORTED", "SKIPPED"]


class LayerVerdict(BaseModel):
    layer: str
    status: Status
    summary: str


class BacktestConfig(BaseModel):
    rec_id: int
    start: str | None = None
    end: str | None = None
    rebalance: str = "never"
    cash_yield_annual: float = 0.0
    tcost_bps: float = 0.0
    slippage_bps: float = 0.0


class EvalReport(BaseModel):
    generated_at: str
    verdicts: list[LayerVerdict] = Field(default_factory=list)
    retrieval: RetrievalEvalReport | None = None
    agents: AgentEvalReport | None = None
    backtest: BacktestResult | None = None
    backtest_config: BacktestConfig | None = None


# Verdict logic

def _verdict_retrieval(r: RetrievalEvalReport) -> LayerVerdict:
    ndcg = r.aggregate.ndcg_at_k
    if ndcg >= 0.7:
        status: Status = "PASSING"
    elif ndcg >= 0.5:
        status = "PARTIAL"
    else:
        status = "FAILING"
    return LayerVerdict(
        layer="L1 Retrieval",
        status=status,
        summary=(
            f"NDCG@{r.k}={ndcg:.3f}, Hit@k={r.aggregate.hit_at_k:.3f}, "
            f"MRR={r.aggregate.mrr:.3f} across {r.n_cases} cases"
        ),
    )


def _verdict_agent(r: AgentEvalReport) -> LayerVerdict:
    stance = r.stance_pass_rate
    schema = r.schema_validity
    rat = r.avg_rationale_overall
    if stance >= 0.8 and schema >= 0.95 and (rat is None or rat >= 4.0):
        status: Status = "PASSING"
    elif stance >= 0.6 and schema >= 0.85:
        status = "PARTIAL"
    else:
        status = "FAILING"
    summary = (
        f"Stance {stance:.0%}, schema {schema:.0%}, "
        f"fully-passing {r.fully_passing_cases}/{r.n_cases}"
    )
    if rat is not None:
        summary += f", rationale {rat:.2f}/5"
    return LayerVerdict(layer="L2 Agent", status=status, summary=summary)


def _verdict_backtest(r: BacktestResult) -> LayerVerdict:
    # Not gated - market conditions swing metrics; report and interpret elsewhere.
    return LayerVerdict(
        layer="L3 Backtest",
        status="REPORTED",
        summary=(
            f"Port {r.portfolio.total_return_pct:+.1f}% (Sharpe {r.portfolio.sharpe:.2f}, "
            f"DD {r.portfolio.max_drawdown_pct:.1f}%) vs benchmark "
            f"{r.benchmark.total_return_pct:+.1f}% (Sharpe {r.benchmark.sharpe:.2f}), "
            f"alpha {r.alpha_annual_pct:+.2f}%, beta {r.beta:.2f}"
        ),
    )


def assemble_report(
    *,
    retrieval: RetrievalEvalReport | None = None,
    agents: AgentEvalReport | None = None,
    backtest: BacktestResult | None = None,
    backtest_config: BacktestConfig | None = None,
) -> EvalReport:
    """Build an EvalReport from already-computed sub-reports (plus verdicts)."""
    verdicts: list[LayerVerdict] = []
    if retrieval is not None:
        verdicts.append(_verdict_retrieval(retrieval))
    else:
        verdicts.append(LayerVerdict(layer="L1 Retrieval", status="SKIPPED", summary="not run"))
    if agents is not None:
        verdicts.append(_verdict_agent(agents))
    else:
        verdicts.append(LayerVerdict(layer="L2 Agent", status="SKIPPED", summary="not run"))
    if backtest is not None:
        verdicts.append(_verdict_backtest(backtest))
    else:
        verdicts.append(LayerVerdict(layer="L3 Backtest", status="SKIPPED", summary="not run"))

    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(),
        verdicts=verdicts,
        retrieval=retrieval,
        agents=agents,
        backtest=backtest,
        backtest_config=backtest_config,
    )


def run_full_eval(
    *,
    rec_id: int | None = None,
    start: str | None = None,
    end: str | None = None,
    rebalance: str = "never",
    cash_yield_annual: float = 0.0,
    tcost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    n_samples: int = 3,
    judge_model: str | None = None,
    k: int = 5,
    skip: set[str] | None = None,
) -> EvalReport:
    """Run L1 + L2 + L3 (L3 skipped if no rec_id) and assemble a single report."""
    skip = skip or set()

    retrieval = None if "L1" in skip else run_retrieval_eval(k=k)

    agents = None if "L2" in skip else run_agent_eval(
        n_samples=n_samples, judge_model=judge_model
    )

    backtest = None
    backtest_config = None
    if "L3" not in skip and rec_id is not None:
        rec = load_recommendation(rec_id)
        if rec is not None:
            backtest = backtest_recommendation(
                rec,
                start=start,
                end=end,
                rebalance=rebalance,
                cash_yield_annual=cash_yield_annual,
                tcost_bps=tcost_bps,
                slippage_bps=slippage_bps,
            )
            backtest_config = BacktestConfig(
                rec_id=rec_id,
                start=start,
                end=end,
                rebalance=rebalance,
                cash_yield_annual=cash_yield_annual,
                tcost_bps=tcost_bps,
                slippage_bps=slippage_bps,
            )

    return assemble_report(
        retrieval=retrieval,
        agents=agents,
        backtest=backtest,
        backtest_config=backtest_config,
    )


# Writers

def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("# Agentic Investor - Eval Report")
    lines.append("")
    lines.append(f"_Generated: {report.generated_at}_")
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    lines.append("| Layer | Status | Summary |")
    lines.append("|---|---|---|")
    for v in report.verdicts:
        lines.append(f"| {v.layer} | **{v.status}** | {v.summary} |")
    lines.append("")

    if report.retrieval is not None:
        r = report.retrieval
        lines.append("## L1 - RAG Retrieval")
        lines.append("")
        lines.append(f"- Cases: {r.n_cases}, fixtures: {r.n_fixtures}, k: {r.k}")
        lines.append(f"- Hit@k: {r.aggregate.hit_at_k:.3f}")
        lines.append(f"- Recall@k: {r.aggregate.recall_at_k:.3f}")
        lines.append(f"- MRR: {r.aggregate.mrr:.3f}")
        lines.append(f"- NDCG@k: {r.aggregate.ndcg_at_k:.3f}")
        lines.append("")

    if report.agents is not None:
        a = report.agents
        lines.append("## L2 - Agent Output")
        lines.append("")
        lines.append(f"- Cases: {a.n_cases}, samples per case: {a.n_samples_per_case}")
        lines.append(f"- Schema validity: {a.schema_validity:.2%}")
        lines.append(f"- Stance pass rate: {a.stance_pass_rate:.2%}")
        lines.append(f"- Confidence-in-range: {a.confidence_in_range_rate:.2%}")
        lines.append(f"- Fully passing cases: {a.fully_passing_cases}/{a.n_cases}")
        if a.avg_rationale_overall is not None:
            lines.append(
                f"- Rationale (LLM-judge): {a.avg_rationale_overall:.2f}/5"
                f" (judge model: `{a.judge_model}`)"
            )
        lines.append("")

    if report.backtest is not None and report.backtest_config is not None:
        b = report.backtest
        c = report.backtest_config
        lines.append("## L3 - Backtest")
        lines.append("")
        lines.append(
            f"- Recommendation #{c.rec_id}, window {b.start} -> {b.end} ({b.n_days} bars)"
        )
        lines.append(
            f"- Rebalance: `{c.rebalance}`, cash yield: {c.cash_yield_annual:.2%}, "
            f"txn cost: {c.tcost_bps} bps, slippage: {c.slippage_bps} bps"
        )
        lines.append("")
        lines.append("| Metric | Portfolio | Benchmark |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Total return | {b.portfolio.total_return_pct:+.2f}% | "
            f"{b.benchmark.total_return_pct:+.2f}% |"
        )
        lines.append(f"| CAGR | {b.portfolio.cagr_pct:+.2f}% | {b.benchmark.cagr_pct:+.2f}% |")
        lines.append(f"| Sharpe | {b.portfolio.sharpe:.2f} | {b.benchmark.sharpe:.2f} |")
        lines.append(
            f"| Max drawdown | {b.portfolio.max_drawdown_pct:.2f}% | "
            f"{b.benchmark.max_drawdown_pct:.2f}% |"
        )
        lines.append(
            f"| Volatility | {b.portfolio.volatility_annual_pct:.2f}% | "
            f"{b.benchmark.volatility_annual_pct:.2f}% |"
        )
        lines.append(f"| Alpha (annualized) | {b.alpha_annual_pct:+.2f}% | - |")
        lines.append(f"| Beta | {b.beta:.3f} | 1.000 |")
        lines.append(f"| Trades | {b.n_trades} | - |")
        lines.append(f"| Total friction | ${b.total_costs:,.2f} | - |")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- LLM-judge running the same model as the agent under test carries "
                 "self-preference bias; prefer a stronger `--judge-model` when possible.")
    lines.append("- L1 golden set is well-separated by topic; add near-duplicate "
                 "distractors to stress-test M15 rerank improvements.")
    lines.append("- L3 backtest is a single window; run `multi_window_backtest` for "
                 "robustness across regimes.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("uv run agentic-investor eval-report --rec-id 1 --judge \\")
    lines.append("  --start 2024-01-01 --end 2026-08-01 --rebalance bands \\")
    lines.append("  --cash-yield 0.045 --tcost-bps 10 --slippage-bps 5")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def write_report(report: EvalReport, out_dir: str | Path = "out/eval") -> tuple[Path, Path]:
    """Write REPORT.md and scorecard.json under out_dir. Returns (md_path, json_path)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "REPORT.md"
    json_path = out / "scorecard.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path
