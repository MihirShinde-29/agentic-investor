"""Smoke tests for config loading."""

from agentic_investor.config import Settings, get_settings


def test_defaults_load(monkeypatch):
    # pydantic-settings reads env vars even when _env_file=None. Unset any
    # provider keys that other test imports (llm/client.py's load_dotenv) may
    # have leaked into os.environ so defaults are deterministic.
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "FINNHUB_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    s = Settings(_env_file=None)
    assert s.llm_model
    assert s.embedding_model.startswith("sentence-transformers/")
    assert s.openai_api_key is None


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
