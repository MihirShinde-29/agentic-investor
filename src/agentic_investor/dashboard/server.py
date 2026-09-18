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
import base64
import json
import logging
import mimetypes
import os
import re
import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from agentic_investor.dashboard.arm_context import ExperimentContext
from agentic_investor.dashboard.events import get_bus

logger = logging.getLogger(__name__)


_TICK_COST_RE = re.compile(
    r"\[tick_cost\].*prompt_tokens=(\d+).*cached_tokens=(\d+)"
    r".*cost_usd=\$([\d.]+)"
)


def _parse_tick_cost_stats(exp_name: str, arm_id: str) -> dict | None:
    """Read the arm's session log and aggregate its tick_cost rows.

    Runs on every compare-summary request; on 5 days of ticks the log is
    still small enough to whole-file scan under the SWR poll interval.
    If perf becomes an issue we can memo by mtime.
    """
    path = Path("out/experiments") / exp_name / f"{arm_id}.log"
    if not path.exists():
        return None
    costs: list[float] = []
    tot_prompt = 0
    tot_cached = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _TICK_COST_RE.search(line)
                if m:
                    tot_prompt += int(m.group(1))
                    tot_cached += int(m.group(2))
                    costs.append(float(m.group(3)))
    except OSError:
        return None
    if not costs:
        return {}
    last5 = costs[-5:]
    return {
        "ticks": len(costs),
        "regen_cost_total": round(sum(costs), 4),
        "regen_cost_avg": round(sum(costs) / len(costs), 4),
        "regen_cost_last5_avg": round(sum(last5) / len(last5), 4),
        "cache_hit_pct": round(
            (tot_cached / tot_prompt) * 100, 1
        ) if tot_prompt > 0 else 0.0,
    }

# On Windows, Python's mimetypes.guess_type reads from the registry and
# sometimes returns 'text/plain' for .js files. Browsers refuse to execute
# scripts with that MIME type. Force the correct types before StaticFiles
# resolves them.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

_DIST = Path(__file__).parent.parent.parent.parent / "dashboard" / "dist"

# arm_id -> (session_dir, cached_at_epoch). Refreshed lazily every
# _ARM_SESSION_CACHE_TTL seconds since arms can (in principle) rotate
# session dirs mid-run (e.g. streamer restart).
_ARM_SESSION_DIRS: dict[str, tuple[Path, float]] = {}
_ARM_SESSION_CACHE_TTL = 60.0


def _resolve_arm_session_dir(arm_id: str) -> Path | None:
    """Find the most-recent session dir whose events are tagged arm_id."""
    cached = _ARM_SESSION_DIRS.get(arm_id)
    if cached and (time.time() - cached[1]) < _ARM_SESSION_CACHE_TTL:
        if cached[0].exists():
            return cached[0]
    root = Path("out/sessions")
    if not root.exists():
        return None
    # Newest first — first hit wins.
    for d in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        jl = d / "session.jsonl"
        if not jl.exists():
            continue
        try:
            with jl.open("r", encoding="utf-8", errors="replace") as f:
                # Peek up to 5 lines; session_start may not have arm_id
                # but subsequent events will (arm_id is stamped in the
                # SessionRecorder.log path, so anything after startup is
                # tagged).
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("arm_id") == arm_id:
                        _ARM_SESSION_DIRS[arm_id] = (d, time.time())
                        return d
        except OSError:
            continue
    return None


def _tail_arm_session_events(arm_id: str, limit: int) -> list[dict]:
    """Return the last `limit` events from arm's session.jsonl on disk."""
    d = _resolve_arm_session_dir(arm_id)
    if d is None:
        return []
    jl = d / "session.jsonl"
    if not jl.exists():
        return []
    # deque(maxlen=limit) keeps only the tail without loading everything.
    buf: deque[dict] = deque(maxlen=limit)
    try:
        with jl.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    buf.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return list(buf)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Bind the running loop to the bus so cross-thread publishes work.
    get_bus().bind_loop(asyncio.get_running_loop())
    logger.info("dashboard event bus bound to loop")
    yield


def _check_basic_creds(authorization_header: str) -> bool:
    """Match the Basic-auth header against DASHBOARD_USER / DASHBOARD_PASS.

    Returns True when creds match, False otherwise. Returns True when the
    env vars aren't set (dev mode — no auth). Uses constant-time compare
    so it doesn't leak timing info about which of user/pass was wrong.
    """
    user = os.getenv("DASHBOARD_USER")
    pw = os.getenv("DASHBOARD_PASS")
    if not user or not pw:
        return True
    if not authorization_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(authorization_header[6:]).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return False
    got_user, _, got_pw = decoded.partition(":")
    return secrets.compare_digest(got_user, user) and secrets.compare_digest(
        got_pw, pw
    )


def _basic_auth_guard(request: Request) -> Response | None:
    """HTTP middleware helper: 401 on bad creds, None to let request through.
    Health endpoint stays open so uptime pings don't need creds.
    """
    if request.url.path == "/api/health":
        return None
    if _check_basic_creds(request.headers.get("authorization", "")):
        return None
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="agentic-investor"'},
    )


def create_app(
    experiment: ExperimentContext | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Agentic Investor dashboard",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.experiment = experiment  # None => legacy single-arm mode
    # Dev-mode CORS so `npm run dev` on :5173 can hit the API on :8000.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        blocked = _basic_auth_guard(request)
        if blocked is not None:
            return blocked
        return await call_next(request)

    @app.middleware("http")
    async def _arm_ctx(request: Request, call_next):
        exp = app.state.experiment
        if exp is None:
            return await call_next(request)
        arm_id = request.query_params.get("arm")
        arm = exp.arm(arm_id) if arm_id else exp.default_arm()
        if arm is None:
            return await call_next(request)
        from agentic_investor.runtime_context import (
            reset_arm_context,
            set_arm_context,
        )
        tokens = set_arm_context(arm.db_url, arm.alpaca_account)
        try:
            return await call_next(request)
        finally:
            reset_arm_context(tokens)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/experiment/meta")
    def experiment_meta() -> dict:
        exp = app.state.experiment
        if exp is None:
            return {"mode": "single"}
        return {
            "mode": "experiment",
            "name": exp.name,
            "arms": [
                {"id": a.arm_id, "account": a.alpaca_account}
                for a in exp.arms
            ],
            "default_arm": exp.default_arm().arm_id,
        }

    @app.get("/api/experiment/compare/summary")
    def experiment_compare_summary() -> dict:
        exp = app.state.experiment
        if exp is None:
            return JSONResponse(
                {"error": "not in experiment mode"}, status_code=400,
            )
        from agentic_investor.runtime_context import (
            reset_arm_context,
            set_arm_context,
        )
        from agentic_investor.tools.paper_broker import get_broker
        from agentic_investor.tools.paper_store import (
            list_orders,
            list_snapshots,
        )

        rows = []
        for arm in exp.arms:
            tokens = set_arm_context(arm.db_url, arm.alpaca_account)
            try:
                summary: dict = {
                    "arm_id": arm.arm_id,
                    "account": arm.alpaca_account,
                }
                try:
                    broker = get_broker()
                    acct = broker.get_account()
                    summary["equity"] = float(acct.equity)
                    summary["cash"] = float(acct.cash)
                    summary["portfolio_value"] = float(acct.portfolio_value)
                    positions = broker.get_positions()
                    summary["positions_count"] = len(positions)
                    summary["positions"] = [
                        {
                            "ticker": p.ticker,
                            "qty": p.qty,
                            "market_value": p.market_value,
                            "unrealized_pl_pct": p.unrealized_pl_pct,
                        }
                        for p in positions
                    ]
                    summary["cash_pct"] = round(
                        (float(acct.cash) / float(acct.equity)) * 100, 2
                    ) if float(acct.equity) > 0 else None
                except Exception as e:  # noqa: BLE001
                    summary["broker_error"] = str(e)
                try:
                    orders = list_orders(limit=1000)
                    summary["n_orders"] = len(orders)
                    filled = [
                        o for o in orders
                        if o.get("status") in ("filled", "partially_filled")
                    ]
                    buys = sum(
                        float(o.get("qty") or 0)
                        * float(o.get("filled_avg_price") or 0)
                        for o in filled if o.get("side") == "buy"
                    )
                    sells = sum(
                        float(o.get("qty") or 0)
                        * float(o.get("filled_avg_price") or 0)
                        for o in filled if o.get("side") == "sell"
                    )
                    summary["buys_notional"] = round(buys, 2)
                    summary["sells_notional"] = round(sells, 2)
                    summary["turnover"] = round(buys + sells, 2)
                except Exception as e:  # noqa: BLE001
                    summary["orders_error"] = str(e)
                try:
                    snaps = list_snapshots(limit=5000)
                    if snaps:
                        summary["last_snapshot_at"] = snaps[0]["captured_at"]
                        # Snapshots are captured chronologically; the oldest
                        # (last in the list, since list_snapshots orders
                        # newest-first) is the arm's session-open baseline.
                        # Equity lives inside the parsed account_json blob,
                        # not at the row's top level.
                        open_equity = float(snaps[-1]["account"]["equity"])
                        summary["opening_equity"] = round(open_equity, 2)
                        if "equity" in summary and open_equity > 0:
                            delta = summary["equity"] - open_equity
                            summary["delta_dollars"] = round(delta, 2)
                            summary["delta_pct"] = round(
                                (delta / open_equity) * 100, 4
                            )
                except Exception:  # noqa: BLE001
                    pass
                # Regen cost + cache hit: parsed from the arm's session log.
                # Written by paper-loop tick_cost events, aggregated live so
                # A/B compare shows the ensemble-vs-single cost gap directly.
                stats = _parse_tick_cost_stats(exp.name, arm.arm_id)
                if stats:
                    summary.update(stats)
            finally:
                reset_arm_context(tokens)
            rows.append(summary)
        return {"experiment": exp.name, "arms": rows}

    @app.get("/api/experiment/compare/news-reactions")
    def experiment_compare_news_reactions(
        limit: int = 30,
        scan_limit: int = 500,
        only_matched: bool = True,
    ) -> dict:
        """For each recent news event, show which arms fired a regen
        within a short window after it.

        All arms share the same news bus so news_received events are
        identical across arm logs. We pick arm A's log as the canonical
        news list, then correlate each news_ts against per-arm regen_done
        events in a fixed lookahead window. Not a causal claim - a regen
        firing within 5 min of a news event might be triggered by other
        news / price / interval - but a useful "reactivity" signal for
        A/B comparison at a glance.
        """
        exp = app.state.experiment
        if exp is None:
            return JSONResponse(
                {"error": "not in experiment mode"}, status_code=400,
            )
        from datetime import datetime as _dt

        WINDOW_SEC = 300

        def _parse_ts(s: str) -> _dt | None:
            try:
                return _dt.fromisoformat(s.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                return None

        def _load_events(arm_id: str, wanted: set[str]) -> list[dict]:
            d = _resolve_arm_session_dir(arm_id)
            out: list[dict] = []
            if d is None or not (d / "session.jsonl").exists():
                return out
            try:
                with (d / "session.jsonl").open(
                    "r", encoding="utf-8", errors="replace",
                ) as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row.get("event") in wanted:
                            out.append(row)
            except OSError:
                return []
            return out

        first_arm = exp.arms[0].arm_id
        news = _load_events(first_arm, {"news_received"})
        if not news:
            return {"experiment": exp.name, "news_reactions": []}

        # Per-arm indexes: regens (for rec metadata lookup) + attribution
        # rows (the actual trade-burst anchors) + skips + order
        # submissions + hot signals. Two paths fire orders in the loop:
        #
        # 1. Fresh regen: regen_start -> regen_done -> trade_plan ->
        #    regen_attribution -> order_submitted x N.
        # 2. Drift-band on stale rec: regen_start -> opinion_drift_skip
        #    -> trade_plan -> regen_attribution -> order_submitted x N
        #    (no regen_done - the LLM said "barely moved", but prices
        #    drifted enough that the existing rec's targets warrant
        #    trades).
        #
        # Both paths emit regen_attribution ~150ms before the orders.
        # So we anchor trade buckets on regen_attribution, not
        # regen_done. Pre-fix we bucketed on regen_done + a 5s grace
        # window and lost all drift-band trades (their originating rec's
        # regen_done was minutes/hours earlier, way outside the grace).
        per_arm_regens: dict[str, list[tuple[_dt, dict]]] = {}
        per_arm_skips: dict[str, list[tuple[_dt, dict]]] = {}
        per_arm_orders: dict[str, list[tuple[_dt, dict]]] = {}
        per_arm_hot_signals: dict[str, list[tuple[_dt, dict]]] = {}
        per_arm_attributions: dict[str, list[tuple[_dt, dict]]] = {}
        for arm in exp.arms:
            evs = _load_events(
                arm.arm_id,
                {
                    "regen_done",
                    "materiality_skip",
                    "materiality_bypass_promoted",
                    "order_submitted",
                    "finbert_hot_signal",
                    "regen_attribution",
                },
            )
            regens: list[tuple[_dt, dict]] = []
            skips: list[tuple[_dt, dict]] = []
            orders: list[tuple[_dt, dict]] = []
            hots: list[tuple[_dt, dict]] = []
            attrs: list[tuple[_dt, dict]] = []
            for ev in evs:
                ts = _parse_ts(ev.get("ts", ""))
                if ts is None:
                    continue
                if ev["event"] == "regen_done":
                    regens.append((ts, ev))
                elif ev["event"] == "order_submitted":
                    orders.append((ts, ev))
                elif ev["event"] == "finbert_hot_signal":
                    hots.append((ts, ev))
                elif ev["event"] == "regen_attribution":
                    attrs.append((ts, ev))
                else:
                    skips.append((ts, ev))
            per_arm_regens[arm.arm_id] = regens
            per_arm_skips[arm.arm_id] = skips
            per_arm_orders[arm.arm_id] = orders
            per_arm_hot_signals[arm.arm_id] = hots
            per_arm_attributions[arm.arm_id] = sorted(attrs)

        # rec_id -> most-recent regen_done metadata, for looking up
        # targets_count + cash_pct + trigger on drift-band bursts that
        # trade against a stale rec.
        per_arm_rec_meta: dict[str, dict[int, dict]] = {
            a.arm_id: {} for a in exp.arms
        }
        for arm in exp.arms:
            for _r_ts, r in per_arm_regens.get(arm.arm_id, []):
                rid = r.get("rec_id")
                if rid is None:
                    continue
                per_arm_rec_meta[arm.arm_id][int(rid)] = r

        # A regen counts as "reacting to news" only if its trigger came
        # from the news pipeline. Startup regens (no-unprocessed-news),
        # price-move triggers, etc. would dump their orders under
        # whatever news happened to arrive nearby - noise, not signal.
        from agentic_investor.orchestrator.decision_engine import (
            NEWS_DRIVEN_TRIGGERS as news_driven_triggers,
        )
        # Broad ETFs: macro news often tags SPY/QQQ but the LLM might
        # act on any held name (Fed rate change -> trim tech). Allow
        # attribution when the news is one of these even if the traded
        # tickers don't match by name.
        macro_news_tickers = {
            "SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "VGK", "EEM",
        }
        # Bursts are the attribution anchor now (not regen_done), so a
        # single rec_id can produce multiple bursts (fresh regen + N
        # drift-band trades). Track claimed by (rec_id, attribution_ts)
        # so the "first news wins" rule still holds per-burst.
        claimed_bursts: dict[str, set[tuple[int, float]]] = {
            a.arm_id: set() for a in exp.arms
        }

        # For each attribution, collect orders in the window
        # [attribution_ts, next_attribution_ts) capped at ORDER_GRACE.
        # ORDER_GRACE is per-burst, not per-rec, so the drift-band
        # trades that fire 17 min after their rec's regen_done still
        # get bucketed correctly - their attribution row landed ~150ms
        # before the orders regardless of how stale the rec is.
        ORDER_GRACE = 5.0

        # Per-arm bursts: list of dicts anchored on regen_attribution.
        # Each entry has attribution ts, rec_id, trigger (from the
        # attribution row itself), trigger_tickers (news tickers the
        # loop attributed to this burst), and the orders bucket.
        per_arm_bursts: dict[str, list[dict]] = {a.arm_id: [] for a in exp.arms}
        for arm in exp.arms:
            attribs = per_arm_attributions.get(arm.arm_id, [])
            orders_all = sorted(per_arm_orders.get(arm.arm_id, []))
            for i, (a_ts, a) in enumerate(attribs):
                rec_id = a.get("rec_id")
                if rec_id is None:
                    continue
                cutoff = attribs[i + 1][0] if i + 1 < len(attribs) else None
                bucket: list[dict] = []
                for o_ts, o in orders_all:
                    delta = (o_ts - a_ts).total_seconds()
                    if delta < 0 or delta > ORDER_GRACE:
                        continue
                    if cutoff is not None and o_ts >= cutoff:
                        continue
                    try:
                        qty = float(o.get("qty") or 0)
                    except (TypeError, ValueError):
                        qty = 0.0
                    bucket.append({
                        "ticker": (o.get("ticker") or "?").upper(),
                        "side": (o.get("side") or "?").lower(),
                        "qty": round(qty, 4),
                    })
                per_arm_bursts[arm.arm_id].append({
                    "attribution_ts": a_ts,
                    "rec_id": int(rec_id),
                    "trigger": a.get("trigger") or "",
                    "trigger_tickers": {
                        t.upper() for t in (a.get("trigger_tickers") or [])
                    },
                    "orders": bucket,
                })

        # Scan a wider window of news than we return so we don't miss
        # older-but-matched events (news arrives faster than reactions
        # log, so a fresh limit=20 window can push out a matched news
        # from 3 min ago that has an arm reaction).
        news = news[-scan_limit:]
        reactions: list[dict] = []
        for n in news:
            n_ts = _parse_ts(n.get("ts", ""))
            if n_ts is None:
                continue
            news_ticker = (n.get("ticker") or "").upper() or None
            is_macro = news_ticker in macro_news_tickers if news_ticker else False
            per_arm: dict[str, dict] = {}
            for arm in exp.arms:
                # A burst B attributes to news N if either N.ticker is
                # in B's orders (direct trade), OR an arm-local
                # finbert_hot_signal for N.ticker fired within ~90s
                # before B (causal trigger, even if B rotated to a peer
                # instead), OR N is a broad-market macro ticker, OR
                # the loop's own regen_attribution.trigger_tickers on
                # B lists N.ticker. On match, all orders in B are
                # shown so rotations count as reactions, not just the
                # ticker-matching leg.
                HOT_LOOKBACK_SEC = 90.0
                matching_orders: list[dict] = []
                matched_regen: dict | None = None
                arm_hots = per_arm_hot_signals.get(arm.arm_id, [])
                for burst in per_arm_bursts.get(arm.arm_id, []):
                    b_ts = burst["attribution_ts"]
                    delta = (b_ts - n_ts).total_seconds()
                    if delta < 0 or delta > WINDOW_SEC:
                        continue
                    trigger = burst["trigger"]
                    if trigger not in news_driven_triggers:
                        continue
                    rec_id = burst["rec_id"]
                    candidate_orders = burst["orders"]
                    order_tickers = {
                        (o.get("ticker") or "").upper() for o in candidate_orders
                    }
                    # Rule (a): direct order-ticker match.
                    matches = bool(news_ticker and news_ticker in order_tickers)
                    # Rule (b): news ticker triggered a hot-signal within
                    # the ~90s before the burst.
                    if not matches and news_ticker:
                        for h_ts, h in arm_hots:
                            gap = (b_ts - h_ts).total_seconds()
                            if gap < 0 or gap > HOT_LOOKBACK_SEC:
                                continue
                            if (h.get("ticker") or "").upper() == news_ticker:
                                matches = True
                                break
                    # Rule (c): broad-market macro news.
                    if not matches and is_macro:
                        matches = True
                    # Rule (d): loop's own attribution. If the burst's
                    # regen_attribution.trigger_tickers lists this news
                    # ticker, count it as attributed even if the LLM
                    # rotated away to a different name.
                    if not matches and news_ticker:
                        if news_ticker in burst["trigger_tickers"]:
                            matches = True
                    if not matches:
                        continue
                    burst_key = (rec_id, b_ts.timestamp())
                    if burst_key in claimed_bursts[arm.arm_id]:
                        # This burst has already been attributed to an
                        # earlier news event - don't duplicate it under
                        # every nearby SPY/QQQ macro row. First
                        # matching news wins.
                        continue
                    matching_orders = candidate_orders
                    rec_meta = per_arm_rec_meta[arm.arm_id].get(rec_id) or {}
                    matched_regen = {
                        "seconds": round(delta, 1),
                        "rec_id": rec_id,
                        "targets_count": len(rec_meta.get("targets") or {}),
                        "cash_pct": rec_meta.get("cash_pct"),
                        # Prefer burst's own trigger (may differ from
                        # the rec's original regen_done trigger when
                        # this is a drift-band re-attribution).
                        "trigger": trigger,
                    }
                    claimed_bursts[arm.arm_id].add(burst_key)
                    break
                regen_info = matched_regen
                orders_list = matching_orders
                # Skip only counts when there's no matched regen.
                skip_info: dict | None = None
                if regen_info is None:
                    for s_ts, s in per_arm_skips.get(arm.arm_id, []):
                        delta = (s_ts - n_ts).total_seconds()
                        if 0 <= delta <= 30:
                            skip_info = {
                                "seconds": round(delta, 1),
                                "kind": s.get("event"),
                            }
                            break
                if regen_info:
                    per_arm[arm.arm_id] = {
                        "reacted": "regen",
                        "regen": regen_info,
                        "orders": orders_list,
                    }
                elif skip_info:
                    per_arm[arm.arm_id] = {"reacted": "skip", **skip_info}
                else:
                    per_arm[arm.arm_id] = {"reacted": "none"}
            reactions.append({
                "ts": n.get("ts"),
                "ticker": n.get("ticker"),
                "headline": (n.get("headline") or "")[:140],
                "per_arm": per_arm,
            })
        if only_matched:
            reactions = [
                r for r in reactions
                if any(
                    a.get("reacted") in ("regen", "orders")
                    for a in r["per_arm"].values()
                )
            ]
        reactions = reactions[-limit:]
        return {"experiment": exp.name, "news_reactions": list(reversed(reactions))}

    @app.get("/api/experiment/compare/equity")
    def experiment_compare_equity(period: str = "1d") -> dict:
        exp = app.state.experiment
        if exp is None:
            return JSONResponse(
                {"error": "not in experiment mode"}, status_code=400,
            )
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        from agentic_investor.runtime_context import (
            reset_arm_context,
            set_arm_context,
        )
        from agentic_investor.tools.paper_store import list_snapshots

        days_map = {
            "1d": 1, "3d": 3, "1w": 7, "1mo": 31, "3mo": 93, "1y": 366,
        }
        n = days_map.get(period.lower(), 1)
        cutoff = _dt.now(_UTC) - _td(days=n)

        def _fresh(iso: str) -> bool:
            try:
                return _dt.fromisoformat(iso.replace("Z", "+00:00")) >= cutoff
            except ValueError:
                return True

        series = []
        for arm in exp.arms:
            tokens = set_arm_context(arm.db_url, arm.alpaca_account)
            try:
                raw = list_snapshots(limit=5000)
                pts = [
                    {"ts": s["captured_at"],
                     "equity": float(s["account"]["equity"])}
                    for s in raw if _fresh(s["captured_at"])
                ]
                # oldest first for time-series rendering
                pts.reverse()
            except Exception:  # noqa: BLE001
                pts = []
            finally:
                reset_arm_context(tokens)
            series.append({"arm_id": arm.arm_id, "points": pts})
        return {"experiment": exp.name, "period": period, "arms": series}

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
    def events(limit: int = 200, arm: str | None = None) -> list[dict]:
        """Recent buffered events (used by the frontend on initial load).

        In experiment mode, `?arm=X` tails the corresponding arm's
        session.jsonl from disk (each arm subprocess writes its own file,
        tagged with `arm_id` via AGENTIC_ARM_ID). The dashboard subprocess
        can't see the arms' in-process event bus, so disk-tail is the only
        way to feed the live event panel per-arm.
        """
        exp = app.state.experiment
        if exp is not None and arm is not None:
            return _tail_arm_session_events(arm, limit)
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
                days_map = {"3d": 3, "1w": 7, "1mo": 31, "3mo": 93, "1y": 366}
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
                            timeout=15,
                        )
                        if df is not None and not df.empty and isinstance(
                            df.columns, __import__("pandas").MultiIndex
                        ):
                            df.columns = df.columns.get_level_values(0)

            if df is None:
                from agentic_investor.tools.market import fetch_ohlcv

                # yfinance doesn't accept "3d" / "1w" directly - the Alpaca
                # path handles them via _period_to_days, but the yfinance
                # fallback needs the nearest keyword it does recognise.
                yf_period_alias = {"3d": "5d", "1w": "5d"}.get(period, period)
                df = fetch_ohlcv(
                    ticker.upper(), period=yf_period_alias, interval=interval,
                )

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

    @app.get("/api/watchlist")
    def watchlist() -> dict:
        """Shadow book: held / recent_exits / on_deck.

        held        - current Alpaca positions
        recent_exits - tickers we sold to zero in the last 24h
        on_deck     - tickers in the latest rec's target that aren't held
        """
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        from agentic_investor.orchestrator.store import (
            list_recommendations,
            load_recommendation,
        )
        from agentic_investor.tools.paper_broker import (
            get_broker,
            get_latest_price,
        )
        from agentic_investor.tools.paper_store import (
            recent_sold_tickers,
        )

        broker = get_broker()
        try:
            positions = broker.get_positions()
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)
        held_set = {p.ticker.upper() for p in positions}
        held = [
            {
                "ticker": p.ticker,
                "qty": p.qty,
                "market_value": p.market_value,
                "unrealized_pl_pct": p.unrealized_pl_pct,
            }
            for p in positions
        ]

        since = str(_dt.now(_UTC) - _td(hours=2))
        exit_tickers = [
            t for t in recent_sold_tickers(since_iso=since) if t not in held_set
        ]
        recent_exits = []
        for t in exit_tickers[:20]:
            price = None
            try:
                price = float(get_latest_price(t))
            except Exception:  # noqa: BLE001
                pass
            recent_exits.append({"ticker": t, "current_price": price})

        # On-deck = picker's frozen candidate set minus what we already
        # hold. Was previously "latest rec's positions minus held", but the
        # loop executes trades immediately so that gap was almost always
        # empty. The picker's top-N is the real "considered but not
        # deployed" set - the true watchlist meaning.
        on_deck = []
        latest = None
        recs = list_recommendations(limit=1)
        if recs:
            latest = load_recommendation(recs[0][0])
        target_by_ticker = {}
        if latest is not None:
            target_by_ticker = {
                p.ticker.upper(): p for p in latest.allocation.positions
            }
        try:
            from agentic_investor.tools.paper_store import (
                load_loop_state as _load_state,
            )
            loop_state = _load_state() or {}
        except Exception:  # noqa: BLE001
            loop_state = {}
        picker_tickers = [
            t.upper() for t in (loop_state.get("frozen_picker_tickers") or [])
        ]
        # Same-session dedup: if a ticker shows in recent_exits it means we
        # tried it and dropped it; don't also list it on-deck. Matches the
        # loop-side picker_exit_suppress behavior (task #109).
        exit_set = {t.upper() for t in exit_tickers}
        # Promotion metadata for enrichment: same signals the LLM sees.
        promoted_at = loop_state.get("promoted_at") or {}
        last_news_price = loop_state.get("last_news_price") or {}
        now_utc = _dt.now(_UTC)
        for tk in picker_tickers:
            if tk in held_set or tk in exit_set:
                continue
            price = None
            try:
                price = float(get_latest_price(tk))
            except Exception:  # noqa: BLE001
                pass
            p = target_by_ticker.get(tk)
            entry: dict = {
                "ticker": tk,
                "target_weight_pct": p.weight_pct if p else 0.0,
                "target_dollars": p.dollars if p else 0.0,
                "confidence": p.confidence if p else None,
                "current_price": price,
            }
            promoted_iso = promoted_at.get(tk)
            if promoted_iso:
                try:
                    p_dt = _dt.fromisoformat(promoted_iso)
                    if p_dt.tzinfo is None:
                        p_dt = p_dt.replace(tzinfo=_UTC)
                    entry["promoted_min_ago"] = int(
                        (now_utc - p_dt).total_seconds() / 60
                    )
                except Exception:  # noqa: BLE001
                    pass
            news_px = last_news_price.get(tk)
            if news_px and price:
                entry["news_price"] = round(float(news_px), 2)
                entry["price_change_pct"] = round(
                    (price / float(news_px) - 1) * 100, 2
                )
            on_deck.append(entry)

        return {"held": held, "recent_exits": recent_exits, "on_deck": on_deck}

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

    @app.get("/api/rec/{rec_id}/precedents")
    def rec_precedents(rec_id: int, k: int = 4) -> dict:
        """Top-k past recs the LLM would consult when regenerating rec_id.

        Arm scoping via ?arm=X flows through runtime_context, so the
        A/B isolation invariant is enforced in the UI path too.
        """
        from agentic_investor.orchestrator.store import load_recommendation

        r = load_recommendation(rec_id)
        if r is None:
            return JSONResponse({"error": "not found"}, status_code=404)

        query_parts: list[str] = []
        portfolio_rationale = (r.allocation.portfolio_rationale or "").strip()
        if portfolio_rationale:
            query_parts.append(portfolio_rationale)
        held = sorted({p.ticker.upper() for p in r.allocation.positions})
        if held:
            query_parts.append(f"Currently holding: {', '.join(held)}")
        query_parts.append(
            f"Risk profile: {r.request.risk}, target {r.request.target}"
        )
        query_text = "\n".join(query_parts).strip()

        exp = app.state.experiment
        arm_id = "solo"
        if exp is not None:
            from agentic_investor.runtime_context import (
                get_active_alpaca_account,
            )
            for a in exp.arms:
                if a.alpaca_account == get_active_alpaca_account():
                    arm_id = a.arm_id
                    break

        try:
            from agentic_investor.memory.retrieval import retrieve_similar

            results = retrieve_similar(query_text, arm_id=arm_id, k=k)
        except Exception as e:  # noqa: BLE001
            return {"rec_id": rec_id, "arm_id": arm_id, "precedents": [],
                    "error": str(e)}
        return {
            "rec_id": rec_id,
            "arm_id": arm_id,
            "query": query_text[:400],
            "precedents": [
                {
                    "rec_id": p.rec_id,
                    "source": p.source,
                    "created_at": p.created_at,
                    "tickers": p.tickers,
                    "similarity": p.similarity,
                    "text": p.text,
                    "outcome_pl_pct_15m": p.outcome_pl_pct_15m,
                    "outcome_pl_pct_60m": p.outcome_pl_pct_60m,
                    "outcome_pl_pct_1d": p.outcome_pl_pct_1d,
                    "outcome_pl_pct_1w": p.outcome_pl_pct_1w,
                    "prompt_line": p.to_prompt_line(),
                }
                for p in results
            ],
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

        When ?tickers=... isn't given, unions four sets so the heatmap shows
        the whole shadow book the allocator considers:
          held  ∪  picker on-deck (frozen top-N)  ∪  latest-rec targets
                ∪  recent exits (24h)
        Trades execute immediately so "latest-rec targets minus held" is
        usually empty; the picker's frozen top-N is what actually keeps
        the on-deck candidates visible on the heatmap.
        """
        from agentic_investor.orchestrator.correlation import (
            compute_correlation_matrix,
        )

        if tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            symbol_set: set[str] = set()
            try:
                from agentic_investor.tools.paper_broker import get_broker

                for p in get_broker().get_positions():
                    symbol_set.add(p.ticker.upper())
            except Exception:  # noqa: BLE001
                pass
            try:
                from agentic_investor.orchestrator.store import (
                    list_recommendations,
                    load_recommendation,
                )

                recs = list_recommendations(limit=1)
                if recs:
                    latest = load_recommendation(recs[0][0])
                    if latest:
                        for p in latest.allocation.positions:
                            symbol_set.add(p.ticker.upper())
            except Exception:  # noqa: BLE001
                pass
            try:
                from agentic_investor.tools.paper_store import (
                    load_loop_state as _load_state,
                )

                loop_state = _load_state() or {}
                for t in (loop_state.get("frozen_picker_tickers") or []):
                    symbol_set.add(t.upper())
            except Exception:  # noqa: BLE001
                pass
            try:
                from datetime import UTC as _UTC
                from datetime import datetime as _dt
                from datetime import timedelta as _td

                from agentic_investor.tools.paper_store import (
                    recent_sold_tickers,
                )

                since = str(_dt.now(_UTC) - _td(hours=2))
                for t in recent_sold_tickers(since_iso=since):
                    symbol_set.add(t)
            except Exception:  # noqa: BLE001
                pass
            symbols = sorted(symbol_set)
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
        # WebSocket upgrades bypass the HTTP middleware; re-check the same
        # Basic creds the browser sends on the upgrade request.
        if not _check_basic_creds(ws.headers.get("authorization", "")):
            await ws.close(code=1008)
            return
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

    @app.websocket("/ws/session/{arm_id}/events")
    async def session_events_ws(ws: WebSocket, arm_id: str) -> None:
        """Live-tail an arm's session.jsonl on disk (task #156).

        Complements /ws/live (which serves the in-process EventBus, only
        useful when publisher + dashboard share a process). This one
        works across subprocess boundaries: dashboard is its own
        subprocess in paper-experiment mode and doesn't share the arm's
        bus, so disk-tail is the only cross-process live feed.

        Sends an initial hydration of the last ~200 events, then polls
        the file every 500 ms and pushes new lines as they land. No
        server-side buffer of clients (each WS keeps its own file
        cursor), so N connected dashboards is O(N) file reads.
        """
        if not _check_basic_creds(ws.headers.get("authorization", "")):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            d = _resolve_arm_session_dir(arm_id)
            if d is None:
                await ws.send_json({
                    "event": "_error",
                    "message": f"no session dir found for arm {arm_id!r}",
                })
                return
            jl = d / "session.jsonl"
            if not jl.exists():
                await ws.send_json({
                    "event": "_error",
                    "message": f"session.jsonl missing under {d}",
                })
                return
            # Initial hydration: last 200 rows.
            initial = _tail_arm_session_events(arm_id, limit=200)
            for row in initial:
                await ws.send_json(row)
            # Now tail. Seek to end so subsequent reads only see new lines.
            f = jl.open("r", encoding="utf-8", errors="replace")
            try:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.5)
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # Partial write at file tail; wait for the writer
                        # to finish the line and try again next tick.
                        f.seek(-len(line), os.SEEK_CUR)
                        await asyncio.sleep(0.5)
                        continue
                    await ws.send_json(row)
            finally:
                f.close()
        except WebSocketDisconnect:
            pass

    # Static frontend (Vite build output). Only mounts if built.
    if _DIST.exists():
        app.mount("/", StaticFiles(directory=_DIST, html=True), name="dashboard")

    return app


def serve_in_thread(
    port: int = 8000,
    experiment: ExperimentContext | None = None,
) -> threading.Thread:
    """Start uvicorn in a daemon thread. Returns the thread handle."""
    import uvicorn

    app = create_app(experiment=experiment)
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


def serve_forever(
    port: int = 8000,
    experiment: ExperimentContext | None = None,
) -> int:
    """Blocking uvicorn.run for the standalone `paper-dashboard` subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    import uvicorn

    app = create_app(experiment=experiment)
    if experiment is not None:
        logger.info(
            "experiment dashboard: name=%s arms=%s -> http://localhost:%d",
            experiment.name,
            [a.arm_id for a in experiment.arms],
            port,
        )
    else:
        logger.info(
            "single-arm dashboard -> http://localhost:%d", port,
        )
    uvicorn.run(
        app, host="0.0.0.0", port=port, log_level="warning",
        access_log=False,
    )
    return 0
