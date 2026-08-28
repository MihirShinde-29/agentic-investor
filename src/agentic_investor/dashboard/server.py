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
