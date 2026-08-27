# Agentic Investor

A multi-agent LLM investment research assistant. Specialized agents analyze
technicals, news, fundamentals, and catalysts; an orchestrator combines their
signals into a recommended paper-portfolio allocation for a given amount, risk
tolerance, and target, then backtests it against SPY.

> **Educational project, not financial advice.** It is paper/simulated only: it
> recommends and backtests, it does not place real trades. The point of value is
> the backtest against a benchmark, not a claim that it predicts prices.

## Architecture

```
User input (amount, risk, target, tickers/universe)
        |
        v
+--------------------- LangGraph orchestrator ----------------------+
|  +-----------+ +---------------+ +--------------+ +-----------+   |
|  | Technical | | News-Sentiment| | Fundamentals | | Catalyst  |   |
|  |  Agent    | |    Agent      | | /Macro Agent | |  Agent    |   |
|  +-----+-----+ +-------+-------+ +------+-------+ +-----+-----+   |
|        +---------------+--------+-------+---------------+          |
|                                v                                  |
|         Portfolio Orchestrator -> allocation + rationale          |
+---------------------------------+---------------------------------+
                                  v
          Paper portfolio -> vectorbt backtest vs SPY -> metrics
                                  v
     FastAPI -> Streamlit dashboard   |   Langfuse traces (cost/latency)
```

MVP ships the Technical and News-Sentiment agents. Fundamentals and Catalyst
are on the roadmap.

## Tech stack

| Layer | Pick |
|---|---|
| Agent orchestration | LangGraph |
| LLM access | LiteLLM (OpenAI / Anthropic / Gemini / local) |
| Prices + indicators | yfinance + pandas-ta |
| News | finnhub |
| Filings | SEC EDGAR |
| RAG / vector DB | ChromaDB, pgvector later |
| Embeddings | sentence-transformers (local), OpenAI later |
| Backtesting | vectorbt |
| Serving | FastAPI |
| Dashboard | Streamlit, React/Next later |
| Observability | Langfuse |
| Storage | SQLite, Postgres later |
| Tooling | uv, ruff, pytest, Docker, GitHub Actions |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install deps into .venv from uv.lock
cp .env.example .env         # fill in keys you have (optional; runs on mocks otherwise)
uv run agentic-investor      # verify the scaffold
uv run pytest
uv run ruff check .
```

## Layout

```
src/agentic_investor/
  config.py        typed settings (pydantic-settings)
  cli.py           console entry point
  llm/             provider-agnostic LLM wrapper (LiteLLM)
  tools/           data sources: market, news, filings
  agents/          technical, news, ... (LangGraph nodes)
  orchestrator/    LangGraph graph + allocation logic
  eval/            vectorbt backtest + LLM judge
  api/             FastAPI service
  dashboard/       Streamlit UI
tests/
```

## Roadmap

- [x] M0: setup & scaffold
- [x] M1: data tools + Technical Agent
- [x] M2: News-Sentiment Agent + RAG
- [x] M3: orchestrator + allocation
- [x] M4: eval harness (backtest + agent/RAG evals + LLM-judge; also calibration Brier/ECE, rubric judge, NDCG/MAP, adversarial + regression suites)
- [x] M5: autonomous stock picking (universe scan + selector for full-auto mode)
- [x] M6: risk-driven strategy engine v2 (presets per risk tier, per-dimension user overrides)
- [ ] M7: paper trading (Alpaca sandbox, live P&L, continuous AI loop scheduler)
- [ ] M8: **live operating dashboard** (capstone: config panel, per-stock live charts with AI buy/sell markers, live reasoning feed, decision timeline, "Start AI" button)
- [ ] M9: Langfuse + Docker + CI (+ eval-gated CI, ops metrics, prompt versioning + A/B, semantic cache, model routing, prompt-injection defense)
- [ ] M10: Macro / regime agent + regime-aware allocation
- [ ] M11: Fundamentals + Catalyst agents (completes 4-agent original spec)
- [ ] M12: Social sentiment agent (Reddit + Google Trends, composite score)
- [ ] M13: A/B testing on live paper (statistical strategy comparison, blocked by M7)
- [ ] M14: prompt & reasoning quality v2 (few-shot, CoT, self-consistency, cross-model ensembling)
- [ ] M15: advanced RAG (cross-encoder rerank, hybrid BM25+semantic, query rewrite / HyDE)
- [ ] M16: multi-agent debate + tool-use mid-reasoning (bull vs bear with judge)
- [ ] M17: agent memory + outcome feedback (per-ticker episodic memory, calibration from real hits)

Later (nice-to-haves, not tracked as milestones):
- Personalization filters (ESG, industry blacklist, tax-aware)
- Weekly digest email / webhook alerts on signal flips
- Multi-user auth (Supabase or JWT), pgvector, React dashboard
- Options / derivatives, real-time streaming quotes
