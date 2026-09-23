# Agentic Investor

A multi-agent LLM investment research assistant that paper-trades on
Alpaca. Specialized agents analyze technicals, news, and macro
signals; a LangGraph orchestrator combines them into an allocation
that a paper-loop then executes against a real broker sandbox with
live news + price streams.

> **Educational project, not financial advice.** Live-money trading
> is off by default (`ALPACA_PAPER=true` is required to swap it).
> The interesting parts of the codebase are the orchestration,
> observability, and reliability engineering around the LLM loop
> more than any claim about market prediction.

## What it does

- **Continuous paper-loop**: an arm process wakes every tick, checks
  news + price drift + technical stance, decides whether to regen an
  allocation via the LLM, executes trades against Alpaca paper.
- **Multi-arm A/B experiments**: `paper-experiment` spawns N arms
  side-by-side on separate Alpaca paper accounts, each with a config
  diff (different model, different prompt gate, different retrieval
  policy). Shared news + price buses so all arms react to identical
  market state.
- **Deterministic replay**: an arm's session records every LLM call +
  clock + price tick + news event to JSONL. A later run against the
  same recording gets bit-exact behavior back — useful for A/B
  testing a prompt change against yesterday's market state without
  waiting for tomorrow's.
- **Live dashboard**: FastAPI + Vite/React front-end with per-arm
  session tail, order log, equity curve, correlation heatmap,
  precedent drill-down, and a live-tail WebSocket for structured
  session events.

## Architecture

```
Alpaca paper account(s)                yfinance / SEC EDGAR
        |                                       |
        v                                       v
+-------------- shared paper-experiment supervisor -----------------+
|  paper-news-bus     paper-price-bus     paper-ml-service         |
|  (single WS,        (single WS, N       (single finBERT +        |
|   fan out to arms)   arm-fanout)         sentence-transformer)   |
+---+------+-------+--------+---------+---------+------------------+
    |      |       |        |         |         |
    v      v       v        v         v         v
+-- arm A ------+ +-- arm B ------+ +-- arm C ------+
| news + prices | | news + prices | | news + prices |
| decision      | | decision      | | decision      |
|   engine      | |   engine      | |   engine      |
|     |         | |     |         | |     |         |
|     v         | |     v         | |     v         |
| LangGraph     | | LangGraph     | | LangGraph     |
|   gather ->   | |   gather ->   | |   gather ->   |
|   allocate -> | |   allocate -> | |   allocate -> |
|   repair ->   | |   repair ->   | |   repair ->   |
|   validate    | |   validate    | |   validate    |
|     |         | |     |         | |     |         |
|     v         | |     v         | |     v         |
| broker.submit | | broker.submit | | broker.submit |
+----+----------+ +---+-----------+ +---+-----------+
     |               |                 |
     +-------+-------+---+-------------+
             v           v
   sqlite (arm state)   session.jsonl (structured events)
                             |
                             v
                    FastAPI + Vite dashboard
                             |
                             v
                       Langfuse traces
```

## Tech stack

| Layer | Pick |
|---|---|
| Agent orchestration | LangGraph |
| LLM access | LiteLLM (OpenAI / Anthropic / Gemini / local) |
| Structured output | instructor (Pydantic schemas) |
| Prices + indicators | yfinance + pandas-ta, Alpaca IEX for live bars |
| News | Alpaca News (Benzinga feed) |
| Broker | Alpaca (paper by default, live gated behind env var) |
| RAG vector store | **sqlite-vec** (rec index + news collection, migrated off chromadb) |
| Embeddings | sentence-transformers (local) via shared paper-ml-service |
| Sentiment | finBERT (ProsusAI/finbert) via shared paper-ml-service |
| Typed decisions (opt-in) | Jev (TypeSafe AI) — non-autoregressive Bool/Choice/Score primitives; drives arm C's per-headline materiality gate + arm B's post-trade verdict feedback |
| Backtesting | vectorbt |
| Live dashboard | FastAPI + WebSockets + Vite/React |
| Observability | Langfuse (opt-in), structured JSONL session logs |
| Storage | SQLite (arm state), sqlite-vec (vector index) |
| Tooling | uv, ruff, pytest, Docker, GitHub Actions |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install deps into .venv from uv.lock
cp .env.example .env         # fill in Alpaca + LLM keys (paper-only by default)
uv run agentic-investor healthcheck
uv run pytest
uv run ruff check .

# Compare 3 presets against a saved recommendation (cheap, no LLM):
uv run agentic-investor compare-strategies 1 --start 2024-01-01 --end 2026-08-01

# Full-strategy comparison: regenerate a fresh rec per preset:
uv run agentic-investor compare-allocators --tickers NVDA,TSLA,AAPL --amount 10000 \
    --start 2024-01-01 --end 2026-08-01

# Let the picker choose tickers from a universe:
uv run agentic-investor compare-allocators --auto --universe sp500_top50 --top-n 8 \
    --amount 10000 --start 2024-01-01 --end 2026-08-01

# Run a live paper-loop (one arm, direct Alpaca):
uv run agentic-investor paper-loop --auto --top-n 8 --regen-mode event

# Run a multi-arm A/B experiment (supervisor spawns arms + buses + ml-service):
uv run agentic-investor paper-experiment reasoning-quality \
    --serve-dashboard --dashboard-port 8000 \
    --paper-loop-args --auto --top-n 8 --regen-mode event

# Deterministic replay of a recorded session (A/B a prompt change):
AGENTIC_RECORD_TO=out/sessions/day5_A/    uv run agentic-investor paper-loop --once ...  # record
AGENTIC_REPLAY_FROM=out/sessions/day5_A/  uv run agentic-investor paper-loop --once ...  # replay
```

## Feature flags

Runtime behavior is controlled by ~16 `AGENTIC_*` env vars — memory
watchdog, retrieval knobs, replay + record modes, TTLs, log rotation,
etc. Full auto-generated table in [FLAGS.md](FLAGS.md) or via:

```bash
uv run agentic-investor flags --format table
```

`FLAGS.md` is regenerated by `scripts/regen_flags_md.py` and enforced
via `scripts/precommit.sh` (install with `ln -sf
../../scripts/precommit.sh .git/hooks/pre-commit`).

## Layout

```
src/agentic_investor/
  config.py         typed settings (pydantic-settings)
  flags.py          central AGENTIC_* env-var registry
  cli.py            console entry point
  llm/              provider-agnostic LLM wrapper (LiteLLM + instructor)
  tools/            data sources: market, news, filings, paper broker,
                    ml_client (arm-side HTTP to paper-ml-service),
                    news_store (sqlite-vec news articles),
                    paper_store (SQLite orders/positions)
  agents/           technical, news, macro (LangGraph nodes)
  orchestrator/     LangGraph graph + allocation + decision engine +
                    memory watchdog + recorder (record/replay)
  memory/           M17 rec index (sqlite-vec) + retrieval + outcomes
  services/         shared subprocess services (paper-ml-service)
  experiments/      paper-experiment supervisor + shared buses
                    (news_bus, price_bus) + outcome-sweeper +
                    bus-purge + reconnect/backfill
  ops/              session recorder + preflight checks
  dashboard/        FastAPI server, WebSocket routes, React frontend
  eval/             backtest + retrieval + LLM-judge harnesses
tests/              ~560 tests
scripts/            operational scripts (leak hunts, flag regen, hooks)
                    + analytics (day_pnl_report, audit_jev_vs_finbert,
                    close_postmortem)
```

## Reliability + ops

Recent reliability work (see `docs/INTERVIEW_NOTES.md` B14 for the
long-form story):

- **Central flag registry**: every `AGENTIC_*` env var typed + defaulted +
  documented in one place, exposed via `agentic-investor flags`.
- **Memory watchdog**: background thread on each arm samples own
  PrivateUsage every 5 s; hard-exits at threshold so the supervisor
  respawns clean. Guards against arena leaks in third-party stacks.
- **sqlite-vec vector store**: both the M17 rec index and the news
  collection moved off chromadb after a corrupt HNSW segment
  reserved 44 GB of address space on load.
- **Shared paper-ml-service**: finBERT + sentence-transformer live in
  one subprocess; arms consume via localhost HTTP with graceful
  local-model fallback. Saves ~1 GB of duplicated model weights across
  three arms.
- **TTL + log rotation**: news_bus + price_bus + news_store all
  periodically purge old rows; per-arm logs use RotatingFileHandler.
- **WebSocket reconnect + REST backfill**: news + price bus writers
  supervise their Alpaca streams with exponential backoff and REST
  backfill for the disconnect window.
- **Structured JSONL session log**: every event dual-writes a
  human-readable console line and a JSONL row through a payload
  sanitizer (Pydantic / datetime / Path / Decimal).
  Programmatic reader: `ops.session.iter_events(session_dir, ...)`.
- **News-materiality gate (pluggable)**: default is finBERT sentiment
  + held-ticker-overlap with a high-signal-keyword bypass. Arm C
  runs a Jev per-headline calibrated probability instead (both
  paths flag-gated, each falls back to a deterministic ticker-
  mention check on failure). Per-arm cost + block-rate + cross-arm
  disagreement breakdown from `scripts/audit_jev_vs_finbert.py`.
- **Post-trade verdict feedback**: arm B appends a "recent decision
  verdicts" section to the allocator prompt scored by Jev
  (`verdict_for_trade`). Self-serving from the arm's own SQLite so
  no broker in scope at prompt-build time.
- **Close-of-day postmortem composer**: `scripts/close_postmortem.py`
  chains per-arm P&L + cross-arm gate audit + A-vs-B same-ticker
  divergence + whipsaw-guard fingerprint into one text report.
- **Deterministic replay**: `AGENTIC_RECORD_TO` + `AGENTIC_REPLAY_FROM`
  capture and reserve every LLM call (hash-keyed), market clock, price
  tick, news event. `AGENTIC_REPLAY_MISS=strict|live` picks
  bit-exact-or-raise vs A/B-friendly fallthrough. `datetime.now()` +
  `uuid4()` in the decision path route through the recorder for
  end-to-end determinism.

## Roadmap

Original agent-scope milestones:

- [x] M0: setup & scaffold
- [x] M1: data tools + Technical Agent
- [x] M2: News-Sentiment Agent + RAG
- [x] M3: orchestrator + allocation
- [x] M4: eval harness (backtest + agent/RAG evals + LLM-judge)
- [x] M5: autonomous stock picking (universe scan + selector)
- [x] M6: risk-driven strategy engine v2
- [x] M7: paper trading (Alpaca sandbox) + continuous AI loop
- [x] M8: live operating dashboard (FastAPI + WebSocket + Vite/React)
- [x] M9: Langfuse + Docker + CI + eval-gate
- [x] M10: Macro / regime agent + regime-aware allocation
- [ ] M11: Fundamentals + Catalyst agents
- [ ] M12: Social sentiment agent (Reddit + Google Trends)
- [x] M13: A/B testing on live paper (parallel-arm experiment framework)
- [x] M14: prompt & reasoning quality v2 (few-shot, CoT, self-consistency, cross-model ensemble)
- [ ] M15: advanced RAG (cross-encoder rerank, hybrid BM25+semantic, query rewrite / HyDE)
- [ ] M16: multi-agent debate + tool-use expansion
- [x] M17: agent memory: RAG over past decisions + outcome feedback

Reliability + observability + deterministic-replay work is captured
in `docs/INTERVIEW_NOTES.md` B14 addenda rather than as milestones —
the codebase evolves faster than a milestone-per-feature roadmap can
keep up with.

Later (nice-to-haves, not tracked as milestones):
- Personalization filters (ESG, industry blacklist, tax-aware)
- Weekly digest email / webhook alerts on signal flips
- Multi-user auth (Supabase or JWT), pgvector, React dashboard
- Options / derivatives, real-time streaming quotes
