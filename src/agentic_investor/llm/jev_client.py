"""TypeSafe AI Jev wrapper — typed decisions for the two spots in our
stack that are classification, not text generation.

Two public functions, each feature-flagged and each with a
deterministic fallback:

- `verdict_for_trade(trade, trajectory)` returns one of
  WORKING / WRONG / UNCLEAR / TOO-EARLY with calibrated probability.
  Used by the allocator prompt's fast_tail to feed the LLM its own
  recent decision quality (arm B in the current A/B).

- `materiality_check(headlines, portfolio_tickers)` returns a
  Noul (bool) + confidence for "is this news batch material to the
  current portfolio". Used by the finBERT prefilter path as a
  swap-in replacement (arm C in the current A/B).

Design notes:

- SDK is `typesafe-sdk` under an optional pyproject extra (`jev` group).
  A fresh clone without the group installs cleanly; both functions
  fall back to the deterministic path when the import fails.
- Auth is `TYPESAFE_API_KEY` from the environment, read by the SDK
  automatically.
- Each function is gated by its own env flag (AGENTIC_JEV_VERDICT_ENABLED
  / AGENTIC_JEV_MATERIALITY_ENABLED). When off, the fallback path runs
  directly — no Jev call, no network hop.
- Failures always fall back rather than raise. Loop resilience matters
  more than Jev availability; a Jev outage should degrade to today's
  behavior, not wedge the arm.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDK availability — optional import; module works either way.
# ---------------------------------------------------------------------------

try:
    from typesafe_sdk import Choice, Noul, TypeSafeClient  # type: ignore[import-not-found]
    _JEV_IMPORTABLE = True
except ImportError:  # pragma: no cover — depends on optional dep group
    Choice = None  # type: ignore[assignment]
    Noul = None  # type: ignore[assignment]
    TypeSafeClient = None  # type: ignore[assignment]
    _JEV_IMPORTABLE = False


# Cached client — TypeSafeClient() is cheap to construct but there's
# no reason to build one per call. Thread-safe first-touch pattern
# mirrors what we already do for the news_store connection.
_client_lock = threading.Lock()
_client: Any = None
_client_fatal_error: Exception | None = None


def _get_client():
    """Return a cached TypeSafeClient or None when unavailable.

    None triggers the deterministic fallback path. A once-per-process
    warning fires on the first failure so operators know the flag is
    on but Jev isn't reachable; subsequent calls stay silent so a
    persistent outage doesn't flood the log.
    """
    global _client, _client_fatal_error
    if not _JEV_IMPORTABLE:
        return None
    if _client_fatal_error is not None:
        return None
    with _client_lock:
        if _client is not None:
            return _client
        if not os.environ.get("TYPESAFE_API_KEY"):
            logger.warning(
                "jev_client: TYPESAFE_API_KEY unset; falling back to "
                "deterministic path for this process",
            )
            _client_fatal_error = RuntimeError("TYPESAFE_API_KEY unset")
            return None
        try:
            _client = TypeSafeClient()
        except Exception as e:  # noqa: BLE001 — never let init break the loop
            logger.warning(
                "jev_client: TypeSafeClient() init failed: %s; falling "
                "back to deterministic path for this process", e,
            )
            _client_fatal_error = e
            _client = None
    return _client


# ---------------------------------------------------------------------------
# 1. Verdict labels for recent trades (arm B).
# ---------------------------------------------------------------------------

VERDICT_LABELS = ("WORKING", "WRONG", "UNCLEAR", "TOO_EARLY")


@dataclass(frozen=True)
class VerdictResult:
    label: str          # one of VERDICT_LABELS
    confidence: float   # [0, 1]; 1.0 for deterministic fallback labels
    from_jev: bool      # True when Jev returned it, False when fallback


def _deterministic_verdict(trade: dict, trajectory: list[dict]) -> VerdictResult:
    """Fallback labeller when Jev is off or unavailable.

    Thresholds pinned intentionally simple so the fallback is
    predictable + auditable. Tuned against the 5-day A/B (2026-09-14
    through 2026-09-18) - most fills either settled within +/-0.5%
    of entry after 30-60 min or bounced enough to justify UNCLEAR.
    """
    age_min = float(trade.get("age_min") or 0)
    side = (trade.get("side") or "").lower()
    if age_min < 30:
        return VerdictResult("TOO_EARLY", 1.0, from_jev=False)
    # trajectory: list of {"minutes": int, "pnl_pct": float}, sorted by minutes
    pnl_60m = None
    for row in trajectory:
        if int(row.get("minutes") or 0) >= 60:
            pnl_60m = float(row.get("pnl_pct") or 0.0)
            break
    if pnl_60m is None and trajectory:
        pnl_60m = float(trajectory[-1].get("pnl_pct") or 0.0)
    if pnl_60m is None:
        return VerdictResult("UNCLEAR", 1.0, from_jev=False)
    # Sign convention: for a BUY, positive pnl = WORKING. For a SELL
    # (position exit / trim), positive underlying-price move after the
    # exit = we sold too early = WRONG. Symmetric for the negative
    # direction. Simplest form that preserves side-awareness.
    signed = pnl_60m if side == "buy" else -pnl_60m
    if signed >= 0.5:
        return VerdictResult("WORKING", 1.0, from_jev=False)
    if signed <= -0.5:
        return VerdictResult("WRONG", 1.0, from_jev=False)
    return VerdictResult("UNCLEAR", 1.0, from_jev=False)


def verdict_for_trade(
    trade: dict, trajectory: list[dict],
) -> VerdictResult:
    """Return a typed verdict on one recent trade.

    `trade`: {"ticker", "side" ("buy"|"sell"), "qty", "price",
              "filled_at", "news_ids", "age_min"}.
    `trajectory`: sorted list of {"minutes": int, "pnl_pct": float}
                  samples of the position's P&L since fill.

    Gated on AGENTIC_JEV_VERDICT_ENABLED=1. When off, or when the
    SDK is unavailable, or on any Jev API failure, falls back to the
    deterministic threshold labeller.
    """
    from agentic_investor.flags import flags
    if not flags.JEV_VERDICT_ENABLED:
        return _deterministic_verdict(trade, trajectory)
    client = _get_client()
    if client is None:
        return _deterministic_verdict(trade, trajectory)
    try:
        state = _render_trade_state(trade, trajectory)
        resp = client.system_one(
            state=state,
            questions={
                "verdict": Choice(
                    instructions="Classify the recent trade's outcome",
                    criteria={
                        "WORKING": "The trade's thesis is validated by "
                                   "the price trajectory after entry",
                        "WRONG": "The trade's thesis is contradicted by "
                                 "the price trajectory after entry",
                        "UNCLEAR": "The price hasn't moved meaningfully "
                                   "in either direction yet",
                        "TOO_EARLY": "Less than 30 minutes have passed "
                                     "since entry; too early to judge",
                    },
                ),
            },
        )
        answer = resp.answers["verdict"]
        label = str(answer.choice)
        if label not in VERDICT_LABELS:
            logger.warning(
                "jev_client: verdict returned unknown label %r; "
                "falling back", label,
            )
            return _deterministic_verdict(trade, trajectory)
        conf = _extract_confidence(answer)
        return VerdictResult(label=label, confidence=conf, from_jev=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "jev_client.verdict_for_trade failed (%s); falling back", e,
        )
        return _deterministic_verdict(trade, trajectory)


def _render_trade_state(trade: dict, trajectory: list[dict]) -> str:
    """Compact human-readable state string for Jev.

    Kept plain-text so a future model swap doesn't force a schema
    rewrite. Jev consumes the string and evaluates typed questions
    against it in a single parallel pass.
    """
    lines = [
        f"trade: {trade.get('side', '?').upper()} "
        f"{trade.get('qty', '?')} {trade.get('ticker', '?')} "
        f"at ${trade.get('price', 0):.2f} "
        f"{trade.get('age_min', 0):.0f} minutes ago",
    ]
    news_ids = trade.get("news_ids") or []
    if news_ids:
        lines.append(f"cited news_ids: {', '.join(str(n) for n in news_ids)}")
    if trajectory:
        pts = ", ".join(
            f"{int(t.get('minutes', 0))}m: {float(t.get('pnl_pct', 0)):+.2f}%"
            for t in trajectory
        )
        lines.append(f"trajectory: {pts}")
    else:
        lines.append("trajectory: (no samples yet)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Materiality check for news batches (arm C).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialityResult:
    material: bool      # True = worth firing a regen on
    confidence: float   # [0, 1]
    from_jev: bool


def _deterministic_materiality(
    headlines: list[str], portfolio_tickers: set[str],
) -> MaterialityResult:
    """Fallback: any headline mentioning any held ticker is material.
    Coarse but honest — matches the pre-finBERT baseline behaviour.
    """
    up_port = {t.upper() for t in portfolio_tickers}
    for h in headlines:
        u = (h or "").upper()
        if any(t in u for t in up_port):
            return MaterialityResult(material=True, confidence=1.0, from_jev=False)
    return MaterialityResult(material=False, confidence=1.0, from_jev=False)


def materiality_check(
    headlines: list[str], portfolio_tickers: set[str],
) -> MaterialityResult:
    """Return a typed material/not decision on a news batch.

    Gated on AGENTIC_JEV_MATERIALITY_ENABLED=1. When off, or on any
    Jev failure, falls back to the deterministic ticker-mention
    check above.
    """
    from agentic_investor.flags import flags
    if not flags.JEV_MATERIALITY_ENABLED:
        return _deterministic_materiality(headlines, portfolio_tickers)
    client = _get_client()
    if client is None:
        return _deterministic_materiality(headlines, portfolio_tickers)
    if not headlines:
        return MaterialityResult(material=False, confidence=1.0, from_jev=False)
    try:
        state = _render_materiality_state(headlines, portfolio_tickers)
        resp = client.system_one(
            state=state,
            questions={
                "material": Noul(
                    instructions=(
                        "The news batch contains at least one item that "
                        "would change the risk or return outlook for "
                        "any ticker in the current portfolio, at a "
                        "magnitude worth re-evaluating positions for"
                    ),
                ),
            },
        )
        answer = resp.answers["material"]
        # NoulAnswer.noul is a *calibrated probability* that the
        # assertion is true, not a bool. Bucket at 0.5 to get the
        # yes/no; keep the probability as the confidence field so
        # downstream logging can grade calibration over time. The
        # SDK docstring and earlier docs implied bool - this bit us
        # in the live smoke test where every batch came back True
        # regardless of content because bool(0.02) == True.
        noul_prob = float(getattr(answer, "noul", 0.5) or 0.0)
        is_material = noul_prob >= 0.5
        # Distance-from-ambiguity as confidence: 0.5 -> 0, 0.02 or
        # 0.98 -> ~1.0. Matches how a Bernoulli-like output should
        # be scored.
        conf = min(1.0, abs(noul_prob - 0.5) * 2.0)
        return MaterialityResult(
            material=is_material, confidence=conf, from_jev=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "jev_client.materiality_check failed (%s); falling back", e,
        )
        return _deterministic_materiality(headlines, portfolio_tickers)


def _render_materiality_state(
    headlines: list[str], portfolio_tickers: set[str],
) -> str:
    port = ", ".join(sorted(t.upper() for t in portfolio_tickers)) or "(empty)"
    lines = [f"portfolio: {port}", "news batch:"]
    for h in headlines[:20]:  # bound the token count on huge batches
        lines.append(f"- {h.strip()[:200]}")
    if len(headlines) > 20:
        lines.append(f"... and {len(headlines) - 20} more headlines")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_confidence(answer: Any) -> float:
    """Best-effort read of the calibrated probability off a Jev answer.
    SDK versions have moved this field around (probability, confidence,
    scores dict). Return 0.5 as a safe neutral when we can't find it.
    """
    for attr in ("probability", "confidence"):
        v = getattr(answer, attr, None)
        if isinstance(v, int | float):
            return max(0.0, min(1.0, float(v)))
    scores = getattr(answer, "scores", None) or getattr(answer, "probabilities", None)
    if isinstance(scores, dict) and scores:
        try:
            return max(0.0, min(1.0, float(max(scores.values()))))
        except (TypeError, ValueError):
            pass
    return 0.5


def _reset_client_for_tests() -> None:
    """Test helper: drop the cached client so a monkeypatched SDK
    is picked up on next call. Not used at runtime."""
    global _client, _client_fatal_error
    with _client_lock:
        _client = None
        _client_fatal_error = None
