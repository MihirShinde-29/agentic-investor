"""A/B significance calculator for per-ticker per-day P&L across arms.

Reads every `out/postmortems/YYYY-MM-DD_close.txt` file, parses the
`top P&L winners:` + `top P&L losers:` blocks per arm to get a
per-arm per-ticker per-day P&L in USD, then runs paired-sample
tests on each arm-pair's delta series over the shared (ticker, day)
observations.

Roadmap item #12: hardens interview claims. Transforms "A won by
$291 on Mon" into a confidence-interval-anchored statement about
the arm-choice effect vs the shared-news baseline.

Two tests reported per arm pair:
- Wilcoxon signed-rank paired test on (ticker,day) delta series.
  Non-parametric, robust to the fat P&L tails of individual trades.
- Bootstrap 95% CI on the mean delta (10000 resamples).

Usage:
    python scripts/ab_significance.py                # all available days
    python scripts/ab_significance.py --since 2026-09-22
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

POSTMORTEM_DIR = Path("out/postmortems")

# Match rows like: `PLTR   $+192.00  (start_mv=...)`. Same shape
# fri_close.py's TICKER_ROW handles.
TICKER_ROW = re.compile(
    r"^\s*(?P<tk>[A-Z][A-Z0-9.\-]+)\s+\$"
    r"(?P<sign>[+-])(?P<amt>[0-9]+\.[0-9]+)\s+\(",
)
FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_close\.txt$")


def _parse_one_day(text: str) -> dict[str, dict[str, float]]:
    """Return {arm: {ticker: pnl_usd}} for one close_postmortem file."""
    out: dict[str, dict[str, float]] = {}
    current_arm: str | None = None
    in_pnl = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("== arm ") and s.endswith("=="):
            current_arm = s.replace("== arm ", "").replace(" ==", "").strip()
            out.setdefault(current_arm, {})
            in_pnl = False
            continue
        if s.startswith("top P&L winners:") or s.startswith("top P&L losers:"):
            in_pnl = True
            continue
        if not in_pnl or current_arm is None:
            continue
        m = TICKER_ROW.match(line)
        if m:
            amt = float(m.group("amt"))
            if m.group("sign") == "-":
                amt = -amt
            out[current_arm][m.group("tk")] = amt
        elif s and not line.startswith(" "):
            in_pnl = False
    return out


def _load_days(since: str | None) -> dict[str, dict[str, dict[str, float]]]:
    """{date: {arm: {ticker: pnl}}} for every close postmortem
    on or after `since` (or all if None)."""
    days: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(POSTMORTEM_DIR.iterdir()):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        d = m.group(1)
        if since and d < since:
            continue
        days[d] = _parse_one_day(path.read_text(encoding="utf-8"))
    return days


def _pair_deltas(days: dict, arm_hi: str, arm_lo: str) -> np.ndarray:
    """Series of (arm_hi - arm_lo) per (date, ticker) where BOTH arms
    have a P&L entry for that ticker on that date."""
    deltas: list[float] = []
    for _d, arms in days.items():
        a_hi = arms.get(arm_hi, {})
        a_lo = arms.get(arm_lo, {})
        for tk in set(a_hi) & set(a_lo):
            deltas.append(a_hi[tk] - a_lo[tk])
    return np.array(deltas, dtype=float)


def _bootstrap_ci(deltas: np.ndarray, n_resamples: int = 10000,
                  ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    if len(deltas) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    resample_means = rng.choice(
        deltas, size=(n_resamples, len(deltas)), replace=True,
    ).mean(axis=1)
    lo = np.quantile(resample_means, (1 - ci) / 2)
    hi = np.quantile(resample_means, 1 - (1 - ci) / 2)
    return (float(lo), float(hi))


def _wilcoxon_p(deltas: np.ndarray) -> float:
    if len(deltas) < 6:
        # Wilcoxon needs a few paired observations to be meaningful;
        # scipy raises below n=6 with default zero_method.
        return float("nan")
    try:
        _, p = stats.wilcoxon(deltas, alternative="two-sided",
                              zero_method="wilcox")
        return float(p)
    except Exception:  # noqa: BLE001
        return float("nan")


def _sample_size_note(n: int) -> str:
    if n < 6:
        return "n<6 -> Wilcoxon N/A, bootstrap unreliable, treat as anecdote"
    if n < 20:
        return f"n={n} -> noisy CI, directional only"
    if n < 50:
        return f"n={n} -> meaningful direction; magnitude has ~30% wiggle"
    return f"n={n} -> firm claim"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="earliest date to include (YYYY-MM-DD)")
    args = ap.parse_args()

    days = _load_days(args.since)
    if not days:
        print("no postmortem files matched", file=sys.stderr)
        return 1

    print(f"loaded {len(days)} days: {min(days)} .. {max(days)}")
    arms = sorted({a for d in days.values() for a in d})
    print(f"arms present: {arms}\n")

    pairs = [(hi, lo) for i, hi in enumerate(arms) for lo in arms[i+1:]]
    for hi, lo in pairs:
        deltas = _pair_deltas(days, hi, lo)
        n = len(deltas)
        if n == 0:
            print(f"{hi} vs {lo}: no shared (ticker,day) observations")
            continue
        mean = float(deltas.mean())
        med = float(np.median(deltas))
        ci_lo, ci_hi = _bootstrap_ci(deltas)
        p = _wilcoxon_p(deltas)
        print(f"=== {hi} vs {lo} ===")
        print(f"  n paired (ticker,day): {n}")
        print(f"  mean delta:            ${mean:+.2f}  "
              f"(median ${med:+.2f})")
        print(f"  bootstrap 95% CI:      "
              f"[${ci_lo:+.2f}, ${ci_hi:+.2f}]")
        print(f"  Wilcoxon p-value:      "
              f"{p:.4f}" if p == p else "  Wilcoxon p-value:      N/A")
        # `p == p` filters NaN. `!= "N/A"` handled by the ternary above.
        sig_flag = ""
        if p == p and p < 0.05:
            sig_flag = "  *** statistically significant at p<0.05 ***"
        print(f"  {_sample_size_note(n)}{sig_flag}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
