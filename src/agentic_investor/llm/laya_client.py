"""Laya (ConvAI Innovations) per-headline materiality gate.

Mirror of jev_client.materiality_check but backed by the open-source
Laya System-One model instead of the hosted Jev API. Same primitive
shape - typed noul (yes/no with calibrated probability) - so the
loop.py wiring is a parallel branch with its own flag and its own
log event, not a swap on the same call site.

Design notes:
- Model loads on first call: ~5s cold start on GPU, ~15s on CPU.
  Cache the Agent instance module-level so subsequent calls reuse it.
- Laya's calibration is materially different from Jev's (this
  weekend's 1256-headline benchmark: 66% bucket agreement, -0.06
  probability correlation). Laya is systematically more
  conservative. The Mon A/B tests whether that helps or hurts P&L
  vs Jev at the same insertion point.
- Deterministic fallback (ticker-mention) matches jev_client so an
  arm using Laya degrades to the same safe behavior on model or
  hardware failure.

Flag gate: AGENTIC_LAYA_MATERIALITY_ENABLED=1 (default off).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayaMaterialityResult:
    """Same structural shape as jev_client.MaterialityResult but
    tagged laya-specific so loop.py can log the right event names
    and the audit script can attribute per-gate cost + latency.
    """
    material: bool
    confidence: float
    from_laya: bool             # True when Laya returned it, False on fallback
    per_headline: tuple = ()    # tuple[tuple[str, float, bool], ...]
    laya_ms_total: float = 0.0  # sum of per-headline forward passes
    laya_ms_max: float = 0.0    # slowest single call (tail)


# Module-level cache: laya.Agent init downloads weights on first
# instantiation, so we amortize across the process. `None` sentinel
# lets us distinguish "never tried" from "tried and failed".
_AGENT_CACHE: object | None = None
_AGENT_LOAD_FAILED: bool = False


def _get_agent() -> object | None:
    """Return a cached laya.Agent or None if the SDK isn't available
    or model loading failed. Never raises - callers should treat None
    as "fall back to deterministic".
    """
    global _AGENT_CACHE, _AGENT_LOAD_FAILED
    if _AGENT_CACHE is not None:
        return _AGENT_CACHE
    if _AGENT_LOAD_FAILED:
        return None
    try:
        import laya  # type: ignore
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("laya_client: loading Agent on device=%s", device)
        _AGENT_CACHE = laya.Agent("convaiinnovations/laya", device=device)
        return _AGENT_CACHE
    except Exception as e:  # noqa: BLE001
        logger.warning("laya_client: Agent load failed (%s); "
                       "falling back to deterministic materiality", e)
        _AGENT_LOAD_FAILED = True
        return None


def preload() -> bool:
    """Explicit warm-up hook. Call from the supervisor at arm-start
    so the first pre-market regen doesn't eat the 5-15s cold-start
    inside the regen tick. Returns True when the Agent is cached.
    """
    return _get_agent() is not None


def _deterministic_materiality(
    headlines: list[str], portfolio_tickers: set[str],
) -> LayaMaterialityResult:
    """Fallback: any headline mentioning any held ticker is material.
    Same coarse baseline jev_client falls back to on any SDK / model
    error, so Laya-arm and Jev-arm degrade identically under failure.
    """
    up_port = {t.upper() for t in portfolio_tickers}
    for h in headlines:
        u = (h or "").upper()
        if any(t in u for t in up_port):
            return LayaMaterialityResult(
                material=True, confidence=1.0, from_laya=False,
            )
    return LayaMaterialityResult(
        material=False, confidence=1.0, from_laya=False,
    )


_INSTRUCTIONS = (
    "This single headline would change the risk or return outlook "
    "for at least one ticker in the current portfolio at a "
    "magnitude worth re-evaluating positions for"
)


def _laya_noul_per_headline(
    agent: object, headline: str, portfolio_tickers: set[str],
) -> tuple[float, bool, float] | None:
    """One Laya forward pass for one headline. Returns
    (probability, material_bool, elapsed_ms) or None on any error
    (caller falls back for this specific headline only).
    """
    try:
        port = ", ".join(sorted(t.upper() for t in portfolio_tickers)) or "(empty)"
        state = (
            f"portfolio: {port}\n"
            f"headline: {(headline or '').strip()[:400]}"
        )
        t0 = time.perf_counter()
        resp = agent.system_one(  # type: ignore[attr-defined]
            state=state,
            questions={
                "material": {"type": "noul", "instructions": _INSTRUCTIONS},
            },
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Laya's noul lives at resp['answers']['material']['noul']
        # (float in [0,1]). Bucket at 0.5 same as jev.
        prob = float(resp["answers"]["material"].get("noul", 0.5) or 0.0)
        return prob, prob >= 0.5, elapsed_ms
    except Exception as e:  # noqa: BLE001
        logger.debug("laya per-headline noul failed: %s", e)
        return None


def materiality_check(
    headlines: list[str], portfolio_tickers: set[str],
) -> LayaMaterialityResult:
    """Per-headline typed materiality decision, Laya edition.

    Loops each headline through Laya individually (same per-headline
    mode Jev uses since 2026-09-22). Batch is material if ANY headline
    is material. Confidence is max per-headline probability when
    material=True, distance-from-ambiguity otherwise (matches Jev's
    formulation so the audit script can compare like-for-like).

    Gated on AGENTIC_LAYA_MATERIALITY_ENABLED=1. Off -> deterministic
    ticker-mention fallback. Model load failure or empty batch also
    routes to deterministic.
    """
    from agentic_investor.flags import flags
    if not flags.LAYA_MATERIALITY_ENABLED:
        return _deterministic_materiality(headlines, portfolio_tickers)
    agent = _get_agent()
    if agent is None:
        return _deterministic_materiality(headlines, portfolio_tickers)
    if not headlines:
        return LayaMaterialityResult(
            material=False, confidence=1.0, from_laya=False,
        )
    try:
        per: list[tuple[str, float, bool]] = []
        any_material = False
        max_prob = 0.0
        ms_total = 0.0
        ms_max = 0.0
        for h in headlines:
            result = _laya_noul_per_headline(agent, h, portfolio_tickers)
            if result is None:
                # Per-headline fallback: keep the loop going.
                up_port = {t.upper() for t in portfolio_tickers}
                u = (h or "").upper()
                mentioned = any(t in u for t in up_port)
                prob = 1.0 if mentioned else 0.0
                mat = mentioned
            else:
                prob, mat, ms = result
                ms_total += ms
                if ms > ms_max:
                    ms_max = ms
            per.append((h[:200], prob, mat))
            if mat:
                any_material = True
            if prob > max_prob:
                max_prob = prob
        if any_material:
            conf = max_prob
        else:
            conf = min(1.0, abs(max_prob - 0.5) * 2.0)
        return LayaMaterialityResult(
            material=any_material,
            confidence=conf,
            from_laya=True,
            per_headline=tuple(per),
            laya_ms_total=ms_total,
            laya_ms_max=ms_max,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "laya_client.materiality_check unexpected failure (%s); "
            "falling back to deterministic", e,
        )
        return _deterministic_materiality(headlines, portfolio_tickers)
