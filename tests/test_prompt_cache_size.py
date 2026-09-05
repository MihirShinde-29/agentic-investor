"""Regression guard for the M17 prompt cache optimization.

If somebody trims _profile_ruleset_block or moves stable content out of
slow_prefix, the block can drop below Anthropic's 1024-token min cache
block and the cache_control marker silently becomes a no-op.
"""

from __future__ import annotations

from agentic_investor.orchestrator.graph import (
    USER_PREAMBLE,
    _messages,
    _profile_ruleset_block,
)
from agentic_investor.orchestrator.state import OrchestratorRequest
from agentic_investor.orchestrator.strategy import load_profile

_ANTHROPIC_MIN_CACHE_TOKENS = 1024
_ROUGH_TOKENS_PER_CHAR = 4


def _est_tokens(text: str) -> int:
    return len(text) // _ROUGH_TOKENS_PER_CHAR


def test_ruleset_block_is_substantial():
    for profile_name in ("conservative", "moderate", "aggressive"):
        profile = load_profile(profile_name)
        block = _profile_ruleset_block(profile)
        tokens = _est_tokens(block)
        assert tokens >= 900, (
            f"{profile_name} ruleset block is {tokens} tokens; needs to stay "
            "large enough that slow_prefix clears Anthropic's 1024-token "
            "min cache block"
        )


def test_slow_prefix_exceeds_anthropic_min_cache_block():
    """Minimal-state prompt (no macro block, no correlation): slow prefix
    must still be > 1024 tokens or the cache_control marker is dead code."""
    profile = load_profile("moderate")
    ruleset = _profile_ruleset_block(profile)
    minimal_slow_prefix = (
        USER_PREAMBLE
        + "\n\n## 1. Request\n  amount:  $100,000.00\n"
        "  risk:    moderate\n  target:  12-month growth"
        + "\n\n## 3. Profile guardrails\n" + ruleset
    )
    tokens = _est_tokens(minimal_slow_prefix)
    assert tokens >= _ANTHROPIC_MIN_CACHE_TOKENS, (
        f"slow_prefix minimal estimate is {tokens} tokens; drops below "
        "Anthropic's 1024-token cache floor. Add stable content to "
        "_profile_ruleset_block or restore whatever was trimmed."
    )


def test_slow_prefix_contains_no_volatile_content():
    """Regen-scoped state (tickers, holdings, batch_ctx, corr universe)
    must live in fast_tail; putting any of it in slow_prefix busts the
    cache on every regen."""
    # Pick tickers not mentioned in the ruleset's worked example
    # (which uses NVDA/AMD/MSFT/AAPL) so a leak from state -> slow is
    # attributable to the plumbing, not the illustrative text.
    state: dict = {
        "request": OrchestratorRequest(
            tickers=["TSLA", "GOOGL", "SNOW"],
            amount=100_000.0, risk="moderate",
        ),
        "technical_signals": [],
        "news_signals": [],
        "market_snapshots": {},
        "news_batch_context": "TSLA Q4 delivery numbers stronger than consensus",
    }
    msgs = _messages(state)
    slow_prefix = msgs[1]["content"][0]["text"]
    fast_tail = msgs[1]["content"][1]["text"]
    for volatile_token in ("TSLA", "GOOGL", "SNOW", "Q4 delivery"):
        assert volatile_token not in slow_prefix, (
            f"{volatile_token!r} leaked into slow_prefix - moves per regen "
            "and would tank cache hit rate"
        )
        assert volatile_token in fast_tail, (
            f"{volatile_token!r} missing from fast_tail entirely - "
            "the split lost data the LLM needs"
        )
