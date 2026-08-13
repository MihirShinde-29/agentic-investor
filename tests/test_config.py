"""Smoke tests for config loading."""

from agentic_investor.config import Settings, get_settings


def test_defaults_load():
    # _env_file=None ignores any local .env so defaults are deterministic in CI.
    s = Settings(_env_file=None)
    assert s.llm_model
    assert s.embedding_model.startswith("sentence-transformers/")
    assert s.openai_api_key is None


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
