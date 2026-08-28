"""App configuration loaded from env vars (12-factor style).

Everything reads settings through get_settings() instead of touching
os.environ directly, so secrets stay out of code and the same build runs
in dev, CI, and prod with different env values.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LiteLLM model strings, e.g. "gpt-4o-mini" or "gemini/gemini-2.5-flash".
    llm_model: str = "gpt-4o-mini"
    orchestrator_model: str = "gpt-4o"
    llm_temperature: float = 0.2

    # Optional so the app boots without them; dev runs on mocks.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    finnhub_api_key: str | None = None

    # Local embeddings, no API key required.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # SEC EDGAR requires a descriptive User-Agent on every request.
    sec_edgar_user_agent: str = "agentic-investor (educational; you@example.com)"

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    database_url: str = "sqlite:///./agentic_investor.db"
    chroma_dir: str = "./.chroma"
    data_dir: str = "./data"

    # Alpaca paper trading. Sign up at alpaca.markets for free paper keys.
    # ALPACA_PAPER=false switches to live trading - do NOT flip this without
    # deliberate review; the client wrapper enforces paper=True by default.
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_paper: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
