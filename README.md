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
- [ ] M3: orchestrator + allocation
- [ ] M4: backtesting + evals
- [ ] M5: FastAPI + Streamlit dashboard
- [ ] M6: Langfuse + Docker + CI
- [ ] Later: Fundamentals & Catalyst agents, pgvector, React dashboard, Alpaca paper API
