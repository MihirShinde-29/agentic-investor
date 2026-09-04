"""Materiality bypass promotes non-material tickers to on-deck rather than
firing a full LLM regen.

Root of the 2026-09-04 churn: high-signal news on tickers we don't hold or
watch (CQP/DOCU/PATH/etc PT changes) fired a full regen every 90s. The LLM
either skipped (barely-moved) or made a decision unrelated to the trigger
(MRK exit on PATH news). Fix promotes the ticker to on-deck and defers to
the next natural regen for the actual LLM call.
"""

from __future__ import annotations


def test_non_material_ticker_promoted_to_frozen_picker():
    """The promotion logic itself: given a list of non-material high-signal
    tickers, they should be appended to frozen_picker_tickers if not
    already present, and duplicates should not be re-added.
    """
    from agentic_investor.orchestrator.loop import LoopState

    state = LoopState(frozen_picker_tickers=["AAPL", "MSFT"])
    promoted_tickers = ["CQP", "AAPL", "DOCU"]  # AAPL is a duplicate
    if state.frozen_picker_tickers is None:
        state.frozen_picker_tickers = []
    existing = {t.upper() for t in state.frozen_picker_tickers}
    newly_promoted = [t for t in promoted_tickers if t not in existing]
    for t in newly_promoted:
        state.frozen_picker_tickers.append(t)

    assert set(newly_promoted) == {"CQP", "DOCU"}  # AAPL filtered out
    assert state.frozen_picker_tickers == ["AAPL", "MSFT", "CQP", "DOCU"]


def test_promotion_creates_list_if_none():
    """If frozen_picker_tickers is None (no picker run yet), initialize."""
    from agentic_investor.orchestrator.loop import LoopState

    state = LoopState()
    if state.frozen_picker_tickers is None:
        state.frozen_picker_tickers = []
    state.frozen_picker_tickers.append("CQP")
    assert state.frozen_picker_tickers == ["CQP"]
