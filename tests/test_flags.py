"""Tests for the central feature-flag registry.

Every AGENTIC_* env var used across the codebase is registered in
`agentic_investor.flags`. Tests verify:
- Defaults return the expected type and value
- Env overrides parse per-kind (int, float, bool_01, csv, str, str_opt)
- Bad values fall back to defaults with a warning
- Fresh read every access (monkeypatch.setenv works without cache
  invalidation)
- No `agentic_investor.*` imports leak in from flags.py (would break
  the "flags is stdlib-only" contract)
- Unknown flag names raise AttributeError cleanly
- Table rendering produces valid markdown
"""

from __future__ import annotations

from agentic_investor.flags import (
    _FLAGS,
    Flag,
    all_flags,
    flags,
    format_flags_table,
)


def test_registry_populated():
    """Non-empty registry with every flag prefixed AGENTIC_."""
    assert len(_FLAGS) > 0
    for name in _FLAGS:
        assert name.startswith("AGENTIC_"), name


def test_defaults_come_through_accessor(monkeypatch):
    # Clear anything an outer test env might have set.
    for name in _FLAGS:
        monkeypatch.delenv(name, raising=False)

    assert flags.MEMORY_RAG is True
    assert flags.MEM_RECYCLE_MB == 0
    assert flags.ARM_ID == "solo"
    assert flags.ENSEMBLE_MODELS == []
    assert flags.ML_SERVICE_URL is None
    assert flags.NEWS_BUS_TTL_HOURS == 4.0
    assert flags.PRICE_BUS_TTL_HOURS == 2.0
    assert flags.MAX_PROMPT_TOKENS == 100000
    assert flags.LOG_ROTATE_MB == 20
    assert flags.LOG_ROTATE_KEEP == 5
    assert flags.SELF_CONSISTENCY_N == 0


def test_bool_01_parses(monkeypatch):
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "0")
    assert flags.MEMORY_RAG is False
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "1")
    assert flags.MEMORY_RAG is True


def test_bool_01_bad_value_falls_back(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("AGENTIC_MEMORY_RAG", "yes")
    with caplog.at_level(logging.WARNING):
        assert flags.MEMORY_RAG is True  # default
    assert any("not 0/1" in r.getMessage() for r in caplog.records)


def test_int_parses_and_bad_value_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("AGENTIC_MEM_RECYCLE_MB", "4096")
    assert flags.MEM_RECYCLE_MB == 4096

    import logging
    monkeypatch.setenv("AGENTIC_MEM_RECYCLE_MB", "not-an-int")
    with caplog.at_level(logging.WARNING):
        assert flags.MEM_RECYCLE_MB == 0
    assert any("not int" in r.getMessage() for r in caplog.records)


def test_float_parses_and_bad_value_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("AGENTIC_NEWS_BUS_TTL_HOURS", "2.5")
    assert flags.NEWS_BUS_TTL_HOURS == 2.5

    import logging
    monkeypatch.setenv("AGENTIC_NEWS_BUS_TTL_HOURS", "float-me")
    with caplog.at_level(logging.WARNING):
        assert flags.NEWS_BUS_TTL_HOURS == 4.0
    assert any("not float" in r.getMessage() for r in caplog.records)


def test_csv_parses_and_strips_whitespace(monkeypatch):
    monkeypatch.setenv(
        "AGENTIC_ENSEMBLE_MODELS",
        "gpt-4o-mini, anthropic/claude-haiku-4-5 ,",
    )
    assert flags.ENSEMBLE_MODELS == [
        "gpt-4o-mini", "anthropic/claude-haiku-4-5",
    ]


def test_csv_empty_env_returns_default_empty_list(monkeypatch):
    monkeypatch.setenv("AGENTIC_ENSEMBLE_MODELS", "")
    assert flags.ENSEMBLE_MODELS == []
    monkeypatch.delenv("AGENTIC_ENSEMBLE_MODELS", raising=False)
    assert flags.ENSEMBLE_MODELS == []


def test_str_opt_empty_clears_to_none(monkeypatch):
    """`str_opt` treats empty string as explicit-unset -> None."""
    monkeypatch.setenv("AGENTIC_ML_SERVICE_URL", "")
    assert flags.ML_SERVICE_URL is None
    monkeypatch.setenv("AGENTIC_ML_SERVICE_URL", "http://x")
    assert flags.ML_SERVICE_URL == "http://x"


def test_str_default_holds_for_arm_id(monkeypatch):
    monkeypatch.delenv("AGENTIC_ARM_ID", raising=False)
    assert flags.ARM_ID == "solo"
    monkeypatch.setenv("AGENTIC_ARM_ID", "B")
    assert flags.ARM_ID == "B"


def test_reads_fresh_each_access(monkeypatch):
    """Critical contract: descriptor reads os.environ every call, so
    tests that setenv/delenv mid-run see the change immediately.
    """
    monkeypatch.delenv("AGENTIC_MEM_RECYCLE_MB", raising=False)
    assert flags.MEM_RECYCLE_MB == 0
    monkeypatch.setenv("AGENTIC_MEM_RECYCLE_MB", "1024")
    assert flags.MEM_RECYCLE_MB == 1024
    monkeypatch.setenv("AGENTIC_MEM_RECYCLE_MB", "2048")
    assert flags.MEM_RECYCLE_MB == 2048
    monkeypatch.delenv("AGENTIC_MEM_RECYCLE_MB", raising=False)
    assert flags.MEM_RECYCLE_MB == 0


def test_unknown_flag_raises_attributeerror():
    try:
        _ = flags.NOT_A_REAL_FLAG
    except AttributeError as e:
        assert "NOT_A_REAL_FLAG" in str(e)
        assert "registered flags" in str(e)
    else:
        raise AssertionError("expected AttributeError for unknown flag")


def test_all_flags_returns_registry_alphabetized():
    rows = all_flags()
    assert rows == sorted(rows, key=lambda f: f.name)
    assert all(isinstance(f, Flag) for f in rows)
    assert len(rows) == len(_FLAGS)


def test_format_flags_table_produces_markdown():
    md = format_flags_table()
    lines = md.splitlines()
    # First two lines are header + separator
    assert lines[0].startswith("| Env var")
    assert lines[1].startswith("| ---")
    # Every registered flag appears as a row
    for name in _FLAGS:
        assert any(f"`{name}`" in ln for ln in lines[2:]), name


def test_flags_module_has_no_agentic_dependencies():
    """The registry must stay stdlib-only so it's safe to import from
    anywhere without circular-import hazards. Any real import from
    `agentic_investor.*` inside flags.py breaks this contract; AST-parse
    the module and inspect Import/ImportFrom nodes so docstring examples
    don't false-positive.
    """
    import ast
    from pathlib import Path
    src = Path("src/agentic_investor/flags.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("agentic_investor"), (
                    f"flags.py must not import from agentic_investor: "
                    f"import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("agentic_investor"), (
                f"flags.py must not import from agentic_investor: "
                f"from {mod} import ..."
            )
