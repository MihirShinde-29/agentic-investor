import pathlib
import sqlite3

BASE = pathlib.Path("out/experiments/reasoning-quality")


def stats(arm: str) -> None:
    p = BASE / f"{arm}.db"
    if not p.exists():
        return
    c = sqlite3.connect(str(p))
    total = c.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
    filled = c.execute("SELECT COUNT(*) FROM paper_orders WHERE status='filled'").fetchone()[0]
    pending = c.execute(
        "SELECT COUNT(*) FROM paper_orders WHERE status NOT IN ('filled','canceled','rejected','expired')"
    ).fetchone()[0]
    buys = c.execute("SELECT COUNT(*) FROM paper_orders WHERE side='buy' AND status='filled'").fetchone()[0]
    sells = c.execute(
        "SELECT COUNT(*) FROM paper_orders WHERE side='sell' AND status='filled'"
    ).fetchone()[0]
    tickers = c.execute(
        "SELECT COUNT(DISTINCT ticker) FROM paper_orders WHERE status='filled'"
    ).fetchone()[0]
    top = c.execute(
        "SELECT ticker, COUNT(*) as n FROM paper_orders WHERE status='filled' GROUP BY ticker ORDER BY n DESC LIMIT 5"
    ).fetchall()
    dollars = c.execute(
        "SELECT COALESCE(SUM(qty*filled_avg_price),0) FROM paper_orders WHERE status='filled'"
    ).fetchone()[0]

    print(f"=== {arm} ===")
    print(
        f"  orders: total={total}  filled={filled}  pending={pending}  "
        f"buys={buys}  sells={sells}"
    )
    print(f"  unique_tickers_traded={tickers}   gross_notional=${dollars:,.0f}")
    print(f"  most-traded: {', '.join(f'{t}x{n}' for t, n in top)}")


for arm in ("A", "B", "C"):
    stats(arm)
    print()
