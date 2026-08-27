"""L2 agent-output evals: run each agent against a golden dataset.

Each case in eval/datasets/agent_cases.jsonl declares a MarketSnapshot with a
defensible expected stance and a confidence range. We call the technical agent
N times per case (N-sample consistency), then measure:
- schema validity: fraction of calls that returned a valid TechnicalSignal
- stance pass rate: fraction of returned stances in the acceptable set
- confidence in range: avg confidence within case's [min, max] bounds
Rationale-quality grading (LLM-judge) lives in eval/judge.py (later).
"""

from collections.abc import Callable
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from agentic_investor.agents.technical import (
    Stance,
    TechnicalSignal,
    analyze_technical,
)
from agentic_investor.tools.market import MarketSnapshot

DEFAULT_CASES = Path(__file__).parent / "datasets" / "agent_cases.jsonl"


class AgentCase(BaseModel):
    id: str
    description: str
    snapshot: MarketSnapshot
    acceptable_stances: list[Stance]
    min_confidence: float = 0.0
    max_confidence: float = 1.0


class CaseResult(BaseModel):
    case_id: str
    n_samples: int
    schema_validity: float
    stance_pass_rate: float
    avg_confidence: float
    confidence_in_range: bool
    stances_seen: list[Stance] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AgentEvalReport(BaseModel):
    n_cases: int
    n_samples_per_case: int
    schema_validity: float  # aggregate across all calls
    stance_pass_rate: float  # aggregate
    confidence_in_range_rate: float
    fully_passing_cases: int  # cases with 100% schema + 100% stance + confidence in range
    per_case: list[CaseResult]


def load_cases(path: str | Path = DEFAULT_CASES) -> list[AgentCase]:
    raw = Path(path).read_text().splitlines()
    return [AgentCase.model_validate_json(line) for line in raw if line.strip()]


def _run_one_case(
    case: AgentCase,
    n_samples: int,
    analyze: Callable[[MarketSnapshot], TechnicalSignal],
) -> CaseResult:
    stances: list[Stance] = []
    confidences: list[float] = []
    errors: list[str] = []
    valid = 0

    for _ in range(n_samples):
        try:
            signal = analyze(case.snapshot)
        except Exception as e:  # noqa: BLE001 - eval must not crash on one bad call
            errors.append(str(e)[:200])
            continue
        valid += 1
        stances.append(signal.stance)
        confidences.append(signal.confidence)

    acceptable = set(case.acceptable_stances)
    stance_hits = sum(1 for s in stances if s in acceptable)
    avg_conf = mean(confidences) if confidences else 0.0
    conf_ok = case.min_confidence <= avg_conf <= case.max_confidence if confidences else False

    return CaseResult(
        case_id=case.id,
        n_samples=n_samples,
        schema_validity=valid / n_samples if n_samples else 0.0,
        stance_pass_rate=stance_hits / max(len(stances), 1),
        avg_confidence=round(avg_conf, 3),
        confidence_in_range=conf_ok,
        stances_seen=stances,
        errors=errors,
    )


def run_agent_eval(
    *,
    cases_path: str | Path = DEFAULT_CASES,
    n_samples: int = 3,
    model: str | None = None,
    analyze: Callable[[MarketSnapshot], TechnicalSignal] | None = None,
) -> AgentEvalReport:
    """Run every case N times through the technical agent; return a scorecard.

    `analyze` defaults to the real agent (`analyze_technical` bound to model);
    tests can inject a deterministic fake to keep the suite offline.
    """
    cases = load_cases(cases_path)
    if analyze is None:
        def analyze(snap: MarketSnapshot) -> TechnicalSignal:
            return analyze_technical(snap, model=model)

    per_case = [_run_one_case(c, n_samples, analyze) for c in cases]

    total_calls = sum(r.n_samples for r in per_case)
    total_valid = sum(r.schema_validity * r.n_samples for r in per_case)
    total_valid_calls = int(round(total_valid))
    total_stance_hits = sum(r.stance_pass_rate * r.n_samples for r in per_case)
    conf_ok_cases = sum(1 for r in per_case if r.confidence_in_range)
    fully_passing = sum(
        1
        for r in per_case
        if r.schema_validity == 1.0 and r.stance_pass_rate == 1.0 and r.confidence_in_range
    )

    return AgentEvalReport(
        n_cases=len(cases),
        n_samples_per_case=n_samples,
        schema_validity=round(total_valid_calls / total_calls, 3) if total_calls else 0.0,
        stance_pass_rate=round(total_stance_hits / total_calls, 3) if total_calls else 0.0,
        confidence_in_range_rate=round(conf_ok_cases / len(per_case), 3) if per_case else 0.0,
        fully_passing_cases=fully_passing,
        per_case=per_case,
    )
