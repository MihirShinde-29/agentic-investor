"""LLM-as-judge: rubric-based rationale quality grading.

A strong LLM grades another agent's rationale on multiple criteria (1-5 each).
Returns a Pydantic Verdict via instructor. Temperature is pinned to 0 for
consistency across runs. Interview note: using the SAME model as judge that
generated the output introduces self-preference bias; prefer a stronger or
different model when the budget allows (configurable per call).
"""

from pydantic import BaseModel, Field

from agentic_investor.agents.technical import TechnicalSignal
from agentic_investor.llm.client import structured_complete
from agentic_investor.tools.market import MarketSnapshot


class RationaleVerdict(BaseModel):
    """Multi-criteria rubric grade from an LLM judge. Each dimension is 1-5."""

    groundedness: int = Field(ge=1, le=5, description="cites specific input values")
    coherence: int = Field(ge=1, le=5, description="reasoning is logically consistent")
    completeness: int = Field(ge=1, le=5, description="covered the major dimensions")
    accuracy: int = Field(ge=1, le=5, description="cited values match the snapshot")
    overall: int = Field(ge=1, le=5, description="overall rationale quality")
    reasoning: str = Field(description="1-2 sentence justification")

    @property
    def passed(self) -> bool:
        return self.overall >= 3


JUDGE_SYSTEM = """\
You are a strict evaluator of technical-analysis reasoning quality. Given a
precomputed indicator snapshot and an agent's stance + reasoning, score the
agent's rationale on five criteria, each 1-5:

- groundedness: does the reasoning cite specific indicator values or named
  signals from the input? 5 = every claim tied to a value; 1 = handwaves.
- coherence: does the reasoning follow logically? 5 = tight causal chain;
  1 = contradictory or non-sequiturs.
- completeness: did it consider the major dimensions (trend, momentum,
  volatility, volume, price structure)? 5 = weighed all relevant dims;
  1 = only looked at one thing.
- accuracy: do the cited values actually match the snapshot? 5 = perfect
  match; 1 = fabricated values or misquoted numbers.
- overall: overall rationale quality. 5 = excellent, 1 = poor.

Be strict. Reserve 5 for genuinely excellent rationales. Give 1 for
hallucinated values or missing reasoning. Keep 'reasoning' to 1-2 sentences.
"""


def _judge_messages(snapshot: MarketSnapshot, signal: TechnicalSignal) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Snapshot (input):\n{snapshot.model_dump_json(indent=2)}\n\n"
                f"Agent output:\n"
                f"- stance: {signal.stance}\n"
                f"- confidence: {signal.confidence:.2f}\n"
                f"- key_drivers: {', '.join(signal.key_drivers) or '(none)'}\n"
                f"- reasoning: {signal.reasoning}\n\n"
                "Grade the rationale on the rubric."
            ),
        },
    ]


def grade_technical_rationale(
    snapshot: MarketSnapshot,
    signal: TechnicalSignal,
    *,
    model: str | None = None,
) -> RationaleVerdict:
    """Grade a technical agent's rationale via LLM judge (temperature=0)."""
    return structured_complete(
        RationaleVerdict, _judge_messages(snapshot, signal), model=model, temperature=0.0
    )
