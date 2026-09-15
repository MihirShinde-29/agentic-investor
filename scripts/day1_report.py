"""Day-1 performance snapshot for the reasoning-quality M14.5 A/B.

Reads each arm's session log + local DB + live Alpaca account state
so the picture is consistent whether the arms are up or down.
"""

import os
import pathlib
import re
import sqlite3
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

BASE = pathlib.Path("out/experiments/reasoning-quality")
FIX_TIME = "2026-09-14 12:04"

TICK_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[tick_cost\]"
    r".*llm_calls=(\d+).*prompt_tokens=(\d+).*cached_tokens=(\d+)"
    r".*cost_usd=\$([\d.]+)"
)


@dataclass
class Arm:
    arm_id: str
    account_key: str
    account_secret: str

    def parse_ticks(self) -> list[tuple]:
        p = BASE / f"{self.arm_id}.log"
        rows = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = TICK_RE.search(line)
            if m:
                rows.append((
                    m.group(1),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    float(m.group(5)),
                ))
        return rows

    def trade_stats(self) -> dict:
        c = sqlite3.connect(str(BASE / f"{self.arm_id}.db"))
        filled = c.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE status='filled'"
        ).fetchone()[0]
        buys = c.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE side='buy' AND status='filled'"
        ).fetchone()[0]
        sells = filled - buys
        tickers = c.execute(
            "SELECT COUNT(DISTINCT ticker) FROM paper_orders WHERE status='filled'"
        ).fetchone()[0]
        notional = c.execute(
            "SELECT COALESCE(SUM(qty*filled_avg_price), 0) FROM paper_orders "
            "WHERE status='filled'"
        ).fetchone()[0]
        top = c.execute(
            "SELECT ticker, COUNT(*) AS n FROM paper_orders WHERE status='filled' "
            "GROUP BY ticker ORDER BY n DESC LIMIT 5"
        ).fetchall()
        held = c.execute(
            "SELECT COUNT(DISTINCT ticker) FROM paper_positions"
        ).fetchone()[0] if _has_table(c, "paper_positions") else None
        return {
            "filled": filled,
            "buys": buys,
            "sells": sells,
            "unique_tickers": tickers,
            "notional": notional,
            "top": top,
            "held_local": held,
        }

    def live_account(self) -> dict:
        if not self.account_key:
            return {}
        from alpaca.trading.client import TradingClient
        cli = TradingClient(self.account_key, self.account_secret, paper=True)
        acct = cli.get_account()
        pos = cli.get_all_positions()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "positions_count": len(pos),
            "unrealized_pl": sum(float(p.unrealized_pl) for p in pos),
            "portfolio_value": float(acct.portfolio_value),
            "top_positions": [
                (p.symbol, float(p.market_value or 0), float(p.unrealized_plpc or 0))
                for p in sorted(
                    pos, key=lambda p: float(p.market_value or 0), reverse=True
                )[:5]
            ],
        }


def _has_table(c, name: str) -> bool:
    return bool(
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
    )


ARMS = [
    Arm("A", os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")),
    Arm("B", os.getenv("ALPACA_API_KEY_B"), os.getenv("ALPACA_API_SECRET_B")),
    Arm("C", os.getenv("ALPACA_API_KEY_C"), os.getenv("ALPACA_API_SECRET_C")),
]

print("=" * 72)
print("M14.5 A/B day-1 report  (arms: A=cooldown, B=cite-to-trade, C=ensemble)")
print("=" * 72)

for arm in ARMS:
    ticks = arm.parse_ticks()
    trades = arm.trade_stats()
    live = arm.live_account()

    total_cost = sum(x[4] for x in ticks)
    total_calls = sum(x[1] for x in ticks)
    total_prompt_tok = sum(x[2] for x in ticks)
    total_cached_tok = sum(x[3] for x in ticks)

    print(f"\n--- ARM {arm.arm_id} ---")
    print(f"  ticks:            {len(ticks)}")
    print(f"  LLM calls total:  {total_calls:,}")
    print(f"  tokens in/cached: {total_prompt_tok:,} / {total_cached_tok:,} "
          f"({total_cached_tok/total_prompt_tok*100:.1f}% cache hit)")
    print(f"  total cost:       ${total_cost:.4f}")
    print(f"  avg/tick:         ${total_cost/max(1,len(ticks)):.4f}")
    print()
    print(f"  trades filled:    {trades['filled']}  "
          f"(buys={trades['buys']}, sells={trades['sells']})")
    print(f"  unique tickers:   {trades['unique_tickers']}")
    print(f"  gross notional:   ${trades['notional']:,.0f}")
    print(f"  most-traded:      {', '.join(f'{t}x{n}' for t, n in trades['top'])}")
    print()
    if live:
        pnl_pct = (live["equity"] - 100000) / 100000 * 100
        print(f"  live equity:      ${live['equity']:,.2f}  (delta${live['equity']-100000:+,.2f}, {pnl_pct:+.4f}%)")
        print(f"  cash:             ${live['cash']:,.2f} ({live['cash']/live['equity']*100:.1f}%)")
        print(f"  open positions:   {live['positions_count']}")
        print(f"  unrealized PnL:   ${live['unrealized_pl']:+.2f}")
        if live["top_positions"]:
            print("  top holdings:")
            for t, mv, upl_pct in live["top_positions"]:
                print(f"    {t:6}  ${mv:>8,.0f}  ({upl_pct:+.2%})")

# Pre/post caching-fix comparison on C
c_ticks = ARMS[2].parse_ticks()
pre = [x for x in c_ticks if x[0] < FIX_TIME]
post = [x for x in c_ticks if x[0] >= FIX_TIME]
if pre and post:
    print("\n--- C caching-fix impact ---")
    pre_avg = sum(x[4] for x in pre) / len(pre)
    post_avg = sum(x[4] for x in post) / len(post)
    print(f"  pre-fix  ({len(pre):>2} ticks): avg ${pre_avg:.4f}/tick")
    print(f"  post-fix ({len(post):>2} ticks): avg ${post_avg:.4f}/tick")
    print(f"  delta:   {(post_avg-pre_avg)/pre_avg*100:+.1f}%  (${post_avg-pre_avg:+.4f}/tick)")
