"""Tests for the M4 aggregate report (verdicts + markdown/JSON writers)."""

import json

from agentic_investor.eval.agents import AgentEvalReport, CaseResult
from agentic_investor.eval.backtest import BacktestMetrics, BacktestResult
from agentic_investor.eval.report import (
    BacktestConfig,
    _verdict_agent,
    _verdict_backtest,
    _verdict_retrieval,
    assemble_report,
    render_markdown,
    write_report,
)
from agentic_investor.eval.retrieval import RetrievalEvalReport, RetrievalMetrics


def _retrieval_report(ndcg: float = 0.95) -> RetrievalEvalReport:
    return RetrievalEvalReport(
        k=5,
        n_cases=8,
        n_fixtures=15,
        aggregate=RetrievalMetrics(hit_at_k=1.0, recall_at_k=0.9, mrr=0.9, ndcg_at_k=ndcg),
        per_case=[],
    )


def _agent_report(
    stance: float = 0.85, schema: float = 1.0, rat: float | None = 4.5
) -> AgentEvalReport:
    return AgentEvalReport(
        n_cases=5,
        n_samples_per_case=3,
        schema_validity=schema,
        stance_pass_rate=stance,
        confidence_in_range_rate=1.0,
        fully_passing_cases=4,
        per_case=[
            CaseResult(
                case_id="c1", n_samples=3, schema_validity=1.0,
                stance_pass_rate=1.0, avg_confidence=0.7, confidence_in_range=True,
            )
        ],
        avg_rationale_overall=rat,
        judge_model="gpt-4o-mini" if rat is not None else None,
    )


def _backtest_result() -> BacktestResult:
    metrics = BacktestMetrics(
        total_return_pct=100.0, cagr_pct=30.0, sharpe=1.0,
        max_drawdown_pct=-25.0, volatility_annual_pct=25.0,
    )
    bench = BacktestMetrics(
        total_return_pct=60.0, cagr_pct=20.0, sharpe=1.2,
        max_drawdown_pct=-18.0, volatility_annual_pct=15.0,
    )
    return BacktestResult(
        start="2024-01-01", end="2026-08-01", n_days=647, init_cash=10000.0,
        portfolio=metrics, benchmark=bench,
        alpha_annual_pct=2.0, beta=1.3,
        portfolio_final_value=20000.0, benchmark_final_value=16000.0,
        n_trades=30, total_costs=25.0,
    )


# Verdict logic

def test_retrieval_verdict_thresholds():
    assert _verdict_retrieval(_retrieval_report(ndcg=0.9)).status == "PASSING"
    assert _verdict_retrieval(_retrieval_report(ndcg=0.6)).status == "PARTIAL"
    assert _verdict_retrieval(_retrieval_report(ndcg=0.3)).status == "FAILING"


def test_agent_verdict_gates_stance_and_schema_and_rationale():
    # All high -> PASSING
    assert _verdict_agent(_agent_report(0.9, 1.0, 4.5)).status == "PASSING"
    # Stance 0.7 -> PARTIAL
    assert _verdict_agent(_agent_report(0.7, 1.0, 4.5)).status == "PARTIAL"
    # Stance 0.4 -> FAILING
    assert _verdict_agent(_agent_report(0.4, 1.0, 4.5)).status == "FAILING"
    # High stance but rationale weak -> PARTIAL (fails the >=4.0 rationale gate)
    assert _verdict_agent(_agent_report(0.9, 1.0, 3.5)).status == "PARTIAL"
    # No judge run -> rationale gate skipped, uses stance/schema only
    assert _verdict_agent(_agent_report(0.9, 1.0, None)).status == "PASSING"


def test_backtest_verdict_is_never_gated():
    v = _verdict_backtest(_backtest_result())
    assert v.status == "REPORTED"
    assert "Sharpe" in v.summary


# Assembly + skipped layers

def test_assemble_marks_missing_layers_as_skipped():
    report = assemble_report()
    statuses = {v.layer: v.status for v in report.verdicts}
    assert statuses == {
        "L1 Retrieval": "SKIPPED",
        "L2 Agent": "SKIPPED",
        "L3 Backtest": "SKIPPED",
    }


def test_assemble_full_report_populates_verdicts_and_data():
    report = assemble_report(
        retrieval=_retrieval_report(),
        agents=_agent_report(),
        backtest=_backtest_result(),
        backtest_config=BacktestConfig(rec_id=1, start="2024-01-01", end="2026-08-01"),
    )
    assert len(report.verdicts) == 3
    assert report.retrieval is not None
    assert report.agents is not None
    assert report.backtest is not None
    assert report.generated_at  # ISO timestamp populated


# Markdown + JSON output

def test_render_markdown_contains_all_sections():
    report = assemble_report(
        retrieval=_retrieval_report(),
        agents=_agent_report(),
        backtest=_backtest_result(),
        backtest_config=BacktestConfig(rec_id=1),
    )
    md = render_markdown(report)
    assert "# Agentic Investor" in md
    assert "L1 - RAG Retrieval" in md
    assert "L2 - Agent Output" in md
    assert "L3 - Backtest" in md
    assert "Caveats" in md
    assert "Reproduce" in md


def test_write_report_creates_both_files(tmp_path):
    report = assemble_report(retrieval=_retrieval_report(), agents=_agent_report())
    md_path, json_path = write_report(report, out_dir=tmp_path / "out")

    assert md_path.exists() and md_path.name == "REPORT.md"
    assert json_path.exists() and json_path.name == "scorecard.json"
    parsed = json.loads(json_path.read_text())
    assert parsed["retrieval"]["n_cases"] == 8
    assert parsed["agents"]["stance_pass_rate"] == 0.85
    assert len(parsed["verdicts"]) == 3
