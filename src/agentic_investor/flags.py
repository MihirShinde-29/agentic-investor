"""Central feature-flag registry.

One source of truth for every `AGENTIC_*` env var. Consolidates the
scattered inline `os.environ.get("AGENTIC_...", "default")` reads that
grew organically across ~15 files. Each flag has a name, type, default,
one-line help, and optional choices/range validation.

Design:

- Descriptors read from `os.environ` at every access, so
  `monkeypatch.setenv` in tests still works without any test changes.
- Zero dependencies on `agentic_investor` modules - stdlib only, so
  this module can be imported from anywhere without circular-import
  hazards. Enforced by test.
- Env var names are unchanged from before the registry - external
  launchers (docker-compose, CI, shell) keep working.
- Introspection: `agentic_investor.flags.all_flags()` yields Flag
  objects; the CLI's `agentic-investor flags` subcommand renders them
  as a table for docs + on-console lookup.

Usage:

    from agentic_investor.flags import flags
    if flags.MEMORY_RAG:
        results = retrieve_similar(...)
    threshold = flags.MEM_RECYCLE_MB
    for m in flags.ENSEMBLE_MODELS:
        ...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Flag:
    """One env-var flag. Read via `read()` (or via the module-level
    `flags.<NAME>` accessor which does the same thing).
    """

    name: str          # full env var name, e.g. "AGENTIC_MEMORY_RAG"
    default: Any
    kind: str          # 'str' | 'str_opt' | 'int' | 'float' | 'bool_01' | 'csv'
    help: str
    choices: tuple[str, ...] | None = None

    def read(self) -> Any:
        raw = os.environ.get(self.name)
        if raw is None or (raw == "" and self.kind != "str_opt"):
            return self.default
        return _coerce(raw, self.kind, self.default, self.name, self.choices)


def _coerce(raw: str, kind: str, default: Any, name: str,
            choices: tuple[str, ...] | None) -> Any:
    if kind == "str_opt":
        # Empty string coerces to None so `AGENTIC_ML_SERVICE_URL=` clears
        # a previously-set value cleanly.
        return raw if raw else None
    if kind == "str":
        if choices and raw not in choices:
            logger.warning(
                "%s=%r not in %s; using default %r",
                name, raw, choices, default,
            )
            return default
        return raw
    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            logger.warning("%s=%r is not int; using default %r", name, raw, default)
            return default
    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            logger.warning("%s=%r is not float; using default %r", name, raw, default)
            return default
    if kind == "bool_01":
        # "1" -> True, "0" -> False, anything else -> default. Chose "1"/"0"
        # over the pydantic-style "true/false/yes/no" because every existing
        # call site already uses "1"/"0" and I don't want to break docker/CI
        # configs that rely on that literal shape.
        if raw == "1":
            return True
        if raw == "0":
            return False
        logger.warning(
            "%s=%r is not 0/1; using default %r", name, raw, default,
        )
        return default
    if kind == "csv":
        return [t.strip() for t in raw.split(",") if t.strip()]
    raise ValueError(f"unknown flag kind {kind!r} for {name}")


# --- registry --------------------------------------------------------------
#
# Order matters only for the `agentic-investor flags` output; alpha-sorted
# there anyway. Group related flags for reading here.

_FLAGS: dict[str, Flag] = {}


def _register(name: str, default: Any, kind: str, help_: str,
              choices: tuple[str, ...] | None = None) -> None:
    if name in _FLAGS:
        raise RuntimeError(f"flag {name!r} already registered")
    _FLAGS[name] = Flag(name=name, default=default, kind=kind,
                        help=help_, choices=choices)


# Arm identity + experiment routing --------------------------------------

_register(
    "AGENTIC_ARM_ID", "solo", "str",
    "Which arm this process is (used to scope M17 retrieval, tag session "
    "events, etc.). 'solo' when running as a single-arm paper-loop; "
    "'A'/'B'/'C' etc. in a paper-experiment run.",
)
_register(
    "AGENTIC_NEWS_BUS", None, "str_opt",
    "sqlite:/// URL of the shared news bus. Set by paper-experiment "
    "supervisor; unset arms fall through to a direct Alpaca websocket.",
)
_register(
    "AGENTIC_PRICE_BUS", None, "str_opt",
    "sqlite:/// URL of the shared price bus (same pattern as news bus).",
)
_register(
    "AGENTIC_ML_SERVICE_URL", None, "str_opt",
    "HTTP URL of the shared paper-ml-service (task #145). When set, arms "
    "route finBERT + embed calls through the service instead of loading "
    "the models locally. Unset = local fallback.",
)


# Memory / M17 retrieval -------------------------------------------------

_register(
    "AGENTIC_MEMORY_RAG", True, "bool_01",
    "Whether the allocator prompt includes retrieved past decisions "
    "(M17). '1' enables, '0' disables. Kill-switch for A/B tests.",
)
_register(
    "AGENTIC_MEMORY_RAG_K", 4, "int",
    "How many past-decision precedents to retrieve on each regen.",
)


# Memory watchdog (task #144/#145 defensive) -----------------------------

_register(
    "AGENTIC_MEM_RECYCLE_MB", 0, "int",
    "Watchdog threshold in MB. When this process's PrivateUsage crosses "
    "the threshold, the arm exits 42 for supervisor respawn. 0 disables "
    "the watchdog (default outside paper-experiment supervisor).",
)


# LLM ensemble / reasoning-quality experiments ---------------------------

_register(
    "AGENTIC_ENSEMBLE_MODELS", [], "csv",
    "Cross-family ensemble models (CSV of LiteLLM model strings). Empty "
    "= single-model. Example: 'gpt-4o-mini,anthropic/claude-haiku-4-5'.",
)
_register(
    "AGENTIC_SELF_CONSISTENCY_N", 0, "int",
    "Number of self-consistency samples for the allocator. 0/1 = single "
    "call. Overridden by AGENTIC_ENSEMBLE_MODELS when both set.",
)
_register(
    "AGENTIC_MAX_PROMPT_TOKENS", 100000, "int",
    "Soft cap on total allocator prompt tokens. When exceeded, the regen "
    "raises PromptTooLargeError and the tick is skipped rather than "
    "letting the provider 400 us.",
)
_register(
    "AGENTIC_CITE_TO_TRADE", True, "bool_01",
    "Trade-gate: '1' requires news citations for order-flow; '0' opens "
    "the gate (arm A in the reasoning-quality experiment).",
)


# Bus / store TTLs (task #147) -------------------------------------------

_register(
    "AGENTIC_NEWS_BUS_TTL_HOURS", 4.0, "float",
    "Retention window for news_bus.db bus_events rows. Sweeper drops "
    "rows older than this every 5 min. 4h = 4x the STALE cutoff.",
)
_register(
    "AGENTIC_PRICE_BUS_TTL_HOURS", 2.0, "float",
    "Retention window for price_bus.db price_ticks rows.",
)
_register(
    "AGENTIC_NEWS_STORE_TTL_DAYS", 30.0, "float",
    "Retention window for the news-article sqlite-vec store (news_articles "
    "+ vec_news tables). Older rows are dropped from both tables in "
    "lockstep every hour.",
)


# Deterministic replay ---------------------------------------------------

_register(
    "AGENTIC_REPLAY_NEWS_SPEED", 1.0, "float",
    "News-replay playback multiplier. 1.0 = real-time (respects "
    "recorded inter-event gaps), 2.0 = twice as fast, 0 = fire "
    "everything as fast as the queue drains. Only consulted when "
    "AGENTIC_REPLAY_FROM is set.",
)


# Session-log rotation (task #147) ---------------------------------------

_register(
    "AGENTIC_LOG_ROTATE_MB", 20, "int",
    "Per-arm log file max size in MB before rotation. On rotate, the "
    "current file is renamed .1, prior .1 becomes .2, etc.",
)
_register(
    "AGENTIC_LOG_ROTATE_KEEP", 5, "int",
    "How many rotated log backups to keep. Total per-arm on-disk log "
    "budget = ROTATE_MB * (ROTATE_KEEP + 1).",
)


# --- ergonomic accessor -------------------------------------------------
#
# `flags.MEMORY_RAG` -> reads AGENTIC_MEMORY_RAG at each access.
# Fresh read every time so `monkeypatch.setenv` works in tests.


@dataclass(frozen=True)
class _FlagsAccessor:
    """Namespace for flag reads. Do not instantiate directly; import
    `flags` from this module.
    """

    # Present as a field for repr / help; not used internally.
    _registered: tuple[str, ...] = field(
        default_factory=lambda: tuple(sorted(_FLAGS.keys())),
    )

    def __getattr__(self, attr: str) -> Any:
        env_name = f"AGENTIC_{attr}"
        flag = _FLAGS.get(env_name)
        if flag is None:
            raise AttributeError(
                f"unknown flag {attr!r} (looked for env {env_name!r}); "
                f"registered flags: {sorted(f[len('AGENTIC_'):] for f in _FLAGS)}"
            )
        return flag.read()


flags = _FlagsAccessor()


# --- introspection ------------------------------------------------------

def all_flags() -> list[Flag]:
    """Return every registered Flag, alphabetized by env-var name."""
    return sorted(_FLAGS.values(), key=lambda f: f.name)


def _fmt_default(f: Flag) -> str:
    if f.default is None:
        return "(unset)"
    if f.kind == "csv":
        return "(empty)" if not f.default else ",".join(f.default)
    if f.kind == "bool_01":
        return "1" if f.default else "0"
    return str(f.default)


def format_flags_table() -> str:
    """Render every flag as a markdown table string. Used by the CLI's
    `agentic-investor flags` subcommand and can be piped into docs.
    """
    rows = all_flags()
    header = "| Env var | Kind | Default | Description |"
    sep = "| --- | --- | --- | --- |"
    lines = [header, sep]
    for f in rows:
        # Trim any pipes inside help to keep the table valid.
        help_ = f.help.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{f.name}` | {f.kind} | `{_fmt_default(f)}` | {help_} |"
        )
    return "\n".join(lines)
