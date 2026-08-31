"""FastAPI + WebSocket server for the live paper-loop dashboard.

Runs alongside the loop on a background thread so a single `paper-loop
--serve-dashboard` invocation gives the operator a live view without a
second process.

REST endpoints hydrate the initial page state; the WebSocket streams every
subsequent session event so the UI updates in real time. Static files (the
built Vite bundle) are served from `dashboard/dist/` when present, giving
you a single production URL to hit.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agentic_investor.dashboard.events import get_bus

logger = logging.getLogger(__name__)

# On Windows, Python's mimetypes.guess_type reads from the registry and
# sometimes returns 'text/plain' for .js files. Browsers refuse to execute
# scripts with that MIME type. Force the correct types before StaticFiles
# resolves them.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

_DIST = Path(__file__).parent.parent.parent.parent / "dashboard" / "dist"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Bind the running loop to the bus so cross-thread publishes work.
    get_bus().bind_loop(asyncio.get_running_loop())
    logger.info("dashboard event bus bound to loop")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Investor dashboard",
        version="0.1.0",
        lifespan=_lifespan,
    )
    # Dev-mode CORS so `npm run dev` on :5173 can hit the API on :8000.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/portfolio")
    def portfolio() -> dict:
        from agentic_investor.tools.paper_broker import get_broker

        try:
            broker = get_broker()
            acct = broker.get_account()
            return {
                "equity": acct.equity,
                "cash": acct.cash,
                "buying_power": acct.buying_power,
                "portfolio_value": acct.portfolio_value,
                "account_number": acct.account_number,
            }
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    @app.get("/api/positions")
    def positions() -> list[dict]:
        from agentic_investor.tools.paper_broker import get_broker

        try:
            broker = get_broker()
            return [
                {
                    "ticker": p.ticker,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "market_value": p.market_value,
                    "unrealized_pl": p.unrealized_pl,
                    "unrealized_pl_pct": p.unrealized_pl_pct,
                }
                for p in broker.get_positions()
            ]
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/trades")
    def trades(limit: int = 50) -> list[dict]:
        from agentic_investor.tools.paper_store import list_orders

        return list_orders(limit=limit)

    @app.get("/api/events")
    def events(limit: int = 200) -> list[dict]:
        """Recent buffered events (used by the frontend on initial load)."""
        return get_bus().recent(limit=limit)

    @app.get("/api/sessions")
    def sessions() -> list[dict]:
        """List past session directories for the session-picker."""
        root = Path("out/sessions")
        if not root.exists():
            return []
        out = []
        for p in sorted(root.iterdir(), reverse=True):
            if p.is_dir() and (p / "session.jsonl").exists():
                out.append({
                    "id": p.name,
                    "started_at": p.stat().st_mtime,
                })
        return out

    @app.get("/api/snapshots")
    def snapshots(
        limit: int = 500,
        session: str | None = None,
        period: str | None = None,
    ) -> list[dict]:
        """Portfolio equity snapshots for the equity-curve chart.

        When ?session=<id> is passed, restrict to snapshots captured within
        that session's start..end window (parsed from session.jsonl).

        When ?period=1d|1mo|3mo|1y is passed (and no session), restrict to
        snapshots within that lookback window from now.
        """
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        from agentic_investor.tools.paper_store import list_snapshots

        raw = list_snapshots(limit=max(limit, 5000))
        window: tuple[_dt, _dt] | None = None
        if session:
            jsonl = Path("out/sessions") / session / "session.jsonl"
            if jsonl.exists():
                import json as _json

                first: str | None = None
                last: str | None = None
                for line in jsonl.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    ts = obj.get("ts")
                    if ts:
                        if first is None:
                            first = ts
                        last = ts
                if first and last:
                    window = (
                        _dt.fromisoformat(first.replace("Z", "+00:00")),
                        _dt.fromisoformat(last.replace("Z", "+00:00")),
                    )
        elif period:
            p = period.lower()
            now = _dt.now(_UTC)
            if p == "1d":
                # "Today's session" = since today's market open (9:30 ET).
                # If we're before today's open, roll back to yesterday's open
                # so the chart still shows something.
                import zoneinfo

                et = zoneinfo.ZoneInfo("America/New_York")
                now_et = now.astimezone(et)
                open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                if now_et < open_et:
                    open_et = open_et - _td(days=1)
                window = (open_et.astimezone(_UTC), now)
            else:
                days_map = {"1mo": 31, "3mo": 93, "1y": 366}
                n = days_map.get(p)
                if n:
                    window = (now - _td(days=n), now)

        def _in_window(iso_ts: str) -> bool:
            if window is None:
                return True
            try:
                ts = _dt.fromisoformat(iso_ts.replace("Z", "+00:00"))
            except ValueError:
                return True
            return window[0] <= ts <= window[1]

        filtered = [s for s in raw if _in_window(s["captured_at"])]
        return [
            {
                "ts": s["captured_at"],
                "equity": float(s["account"]["equity"]),
                "cash": float(s["account"]["cash"]),
                "portfolio_value": float(s["account"]["portfolio_value"]),
            }
            for s in reversed(filtered)  # oldest first for time series
        ][:limit]

    @app.get("/api/bars/{ticker}")
    def bars(
        ticker: str,
        period: str = "1d",
        interval: str = "5m",
        session: str | None = None,
    ) -> dict:
        """OHLCV bars for a ticker; used by per-ticker charts + SPY overlay.

        When ?session=<id> is passed, ignore `period` and instead fetch bars
        for the calendar day(s) that the session ran (parsed from JSONL).
        This lets the SPY overlay match the equity curve for past-session
        replays instead of always showing today's bars.
        """
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        import yfinance as _yf

        try:
            df = None
            if session:
                jsonl = Path("out/sessions") / session / "session.jsonl"
                if jsonl.exists():
                    import json as _json

                    first_ts: str | None = None
                    last_ts: str | None = None
                    for line in jsonl.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        ts = obj.get("ts")
                        if ts:
                            if first_ts is None:
                                first_ts = ts
                            last_ts = ts
                    if first_ts and last_ts:
                        first_dt = _dt.fromisoformat(first_ts.replace("Z", "+00:00"))
                        last_dt = _dt.fromisoformat(last_ts.replace("Z", "+00:00"))
                        # yfinance 'end' is exclusive; pad by a day to include
                        # the last bar. 1m bars only work for last 7 days; use
                        # 5m for anything older to stay within the 60-day cap.
                        age_days = (_dt.now(first_dt.tzinfo) - first_dt).days
                        chosen_interval = (
                            "1m" if age_days <= 6 else "5m"
                            if age_days <= 55 else "1d"
                        )
                        df = _yf.download(
                            ticker.upper(),
                            start=first_dt.date().isoformat(),
                            end=(last_dt.date() + _td(days=1)).isoformat(),
                            interval=chosen_interval,
                            progress=False,
                            auto_adjust=False,
                        )
                        if df is not None and not df.empty and isinstance(
                            df.columns, __import__("pandas").MultiIndex
                        ):
                            df.columns = df.columns.get_level_values(0)

            if df is None:
                from agentic_investor.tools.market import fetch_ohlcv

                df = fetch_ohlcv(ticker.upper(), period=period, interval=interval)

            if df is None or df.empty:
                return {"ticker": ticker.upper(), "bars": []}
            close = df["Close"].astype(float)
            sma20 = close.rolling(20).mean()
            bars_out = []
            for ts, row in df.iterrows():
                bars_out.append({
                    "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "o": float(row["Open"]),
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "c": float(row["Close"]),
                    "v": float(row["Volume"]),
                    "sma20": (
                        float(sma20.loc[ts])
                        if not (sma20.loc[ts] != sma20.loc[ts])
                        else None
                    ),
                })
            return {"ticker": ticker.upper(), "bars": bars_out}
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                {"ticker": ticker.upper(), "error": str(e)}, status_code=500
            )

    @app.get("/api/latest/{ticker}")
    def latest_price(ticker: str) -> dict:
        """Current last-trade price for one ticker; Alpaca first, yfinance
        fallback. Feeds the per-ticker chart's live-price overlay so the
        chart doesn't look frozen between 5-minute bar rolls."""
        from datetime import UTC as _UTC
        from datetime import datetime as _dt

        from agentic_investor.tools.paper_broker import get_latest_price

        try:
            price = get_latest_price(ticker.upper())
            return {
                "ticker": ticker.upper(),
                "price": float(price) if price is not None else None,
                "ts": _dt.now(_UTC).isoformat(),
            }
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                {"ticker": ticker.upper(), "price": None, "error": str(e)},
                status_code=500,
            )

    @app.get("/api/rec/{rec_id}")
    def rec(rec_id: int) -> dict:
        """Recommendation details for the trade drill-down: allocation +
        per-ticker technical/news signals + violations. All the reasoning
        the LLM used, so the frontend can show 'why did we trade?'"""
        from agentic_investor.orchestrator.store import load_recommendation

        r = load_recommendation(rec_id)
        if r is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {
            "rec_id": rec_id,
            "amount": r.request.amount,
            "risk": r.request.risk,
            "tickers": r.request.tickers,
            "target": r.request.target,
            "positions": [
                {
                    "ticker": p.ticker,
                    "weight_pct": p.weight_pct,
                    "dollars": p.dollars,
                    "confidence": p.confidence,
                    "rationale": p.rationale,
                }
                for p in r.allocation.positions
            ],
            "cash_pct": r.allocation.cash_pct,
            "cash_dollars": r.allocation.cash_dollars,
            "portfolio_rationale": r.allocation.portfolio_rationale,
            "technical_signals": [
                {
                    "ticker": s.ticker,
                    "stance": s.stance,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                    "key_drivers": list(getattr(s, "key_drivers", []) or []),
                }
                for s in (r.technical_signals or [])
            ],
            "news_signals": [
                {
                    "ticker": s.ticker,
                    "stance": s.stance,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                }
                for s in (r.news_signals or [])
            ],
            "violations": list(r.violations or []),
        }

    @app.get("/api/filter-skips")
    def filter_skips(limit: int = 50) -> list[dict]:
        """Recent opinion-drift-filter skips for the attribution counter."""
        from agentic_investor.tools.paper_store import list_filter_skips

        return list_filter_skips(limit=limit)

    @app.get("/api/calibration")
    def calibration(horizon_minutes: int = 60, n_buckets: int = 5) -> dict:
        """Bucketed confidence-vs-win-rate for the calibration mini-widget."""
        from agentic_investor.ops.calibration import (
            bucket_outcomes,
            compute_trade_outcomes,
        )

        try:
            outcomes = compute_trade_outcomes(horizon_minutes=horizon_minutes)
            buckets = bucket_outcomes(outcomes, n_buckets=n_buckets)
            return {
                "horizon_minutes": horizon_minutes,
                "n_trades": len(outcomes),
                "overall_win_rate": (
                    sum(o.win for o in outcomes) / len(outcomes)
                    if outcomes else 0.0
                ),
                "buckets": [
                    {
                        "lo": b.lo,
                        "hi": b.hi,
                        "n_trades": b.n_trades,
                        "mean_confidence": b.mean_confidence,
                        "win_rate": b.win_rate,
                    }
                    for b in buckets
                ],
            }
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                {"error": str(e), "buckets": [], "n_trades": 0},
                status_code=500,
            )

    @app.get("/api/broker/status")
    def broker_status() -> dict:
        """Live health of Alpaca connection (clock + market state)."""
        from agentic_investor.tools.paper_broker import get_broker

        try:
            broker = get_broker()
            clock = broker.get_clock()
            return {
                "connected": True,
                "market_open": bool(getattr(clock, "is_open", False)),
                "next_open": str(getattr(clock, "next_open", None)),
                "next_close": str(getattr(clock, "next_close", None)),
                "server_time": str(getattr(clock, "timestamp", None)),
            }
        except Exception as e:  # noqa: BLE001
            return {"connected": False, "error": str(e)}

    @app.get("/api/correlation")
    def correlation(
        tickers: str | None = None, window_days: int = 60
    ) -> dict:
        """Pairwise 60-day return correlation matrix for the heatmap widget.

        When ?tickers=... isn't given, uses current open positions from the
        broker so the widget "just works" without config.
        """
        from agentic_investor.orchestrator.correlation import (
            compute_correlation_matrix,
        )

        if tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            try:
                from agentic_investor.tools.paper_broker import get_broker

                symbols = [p.ticker for p in get_broker().get_positions()]
            except Exception:  # noqa: BLE001
                symbols = []
        if len(symbols) < 2:
            return {"tickers": symbols, "matrix": [], "window_days": window_days}
        try:
            df = compute_correlation_matrix(symbols, window_days=window_days)
            if df is None:
                return {
                    "tickers": symbols, "matrix": [], "window_days": window_days,
                }
            ordered = list(df.columns)
            matrix = [
                [float(df.loc[a, b]) for b in ordered] for a in ordered
            ]
            return {
                "tickers": ordered,
                "matrix": matrix,
                "window_days": window_days,
            }
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/session/{session_id}/events")
    def session_events(session_id: str, limit: int = 5000) -> list[dict]:
        """Replay events from a past session's JSONL for the session picker."""
        import json as _json

        jsonl = Path("out/sessions") / session_id / "session.jsonl"
        if not jsonl.exists():
            return JSONResponse({"error": "session not found"}, status_code=404)
        out: list[dict] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    @app.websocket("/ws/live")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        q = get_bus().subscribe()
        try:
            while True:
                event = await q.get()
                await ws.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            get_bus().unsubscribe(q)

    # Static frontend (Vite build output). Only mounts if built.
    if _DIST.exists():
        app.mount("/", StaticFiles(directory=_DIST, html=True), name="dashboard")

    return app


def serve_in_thread(port: int = 8000) -> threading.Thread:
    """Start uvicorn in a daemon thread. Returns the thread handle."""
    import uvicorn

    app = create_app()
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    # Prevent uvicorn from installing signal handlers off the main thread.
    server.install_signal_handlers = lambda: None  # type: ignore[assignment]

    def _run() -> None:
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, name="dashboard-server", daemon=True)
    t.start()
    logger.info("dashboard listening on http://localhost:%d", port)
    return t
