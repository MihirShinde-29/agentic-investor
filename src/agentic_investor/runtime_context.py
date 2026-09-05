"""Per-request/per-thread runtime context for arm routing.

The dashboard's HTTP middleware sets these ContextVars for the duration
of a request when the client passes `?arm=X`; internal helpers in
paper_store / paper_broker / orchestrator.store transparently pick them
up so we don't have to thread arm_id through every function signature.

Everything defaults to None. When None, the helpers fall back to the
process's ambient defaults (settings.database_url + primary Alpaca).
So this module is a no-op for single-arm paper-loop runs; only the
experiment dashboard actively sets these.

ContextVars are async-safe and thread-local, so concurrent HTTP requests
each see their own arm without leaking across.
"""

from __future__ import annotations

from contextvars import ContextVar

_active_db_url: ContextVar[str | None] = ContextVar(
    "active_db_url", default=None,
)
_active_alpaca_account: ContextVar[str | None] = ContextVar(
    "active_alpaca_account", default=None,
)


def get_active_db_url() -> str | None:
    """Return the DB URL for the current arm context, or None if unset."""
    return _active_db_url.get()


def get_active_alpaca_account() -> str | None:
    """Return the Alpaca account label for the current arm context.

    Returns one of "primary" | "secondary" | "tertiary" when set by
    dashboard middleware. When None, get_broker() falls back to
    "primary" (its own default).
    """
    return _active_alpaca_account.get()


def set_arm_context(
    db_url: str | None, alpaca_account: str | None,
) -> tuple:
    """Set both context vars atomically; returns (tok_db, tok_acct) for reset."""
    tok_db = _active_db_url.set(db_url)
    tok_acct = _active_alpaca_account.set(alpaca_account)
    return (tok_db, tok_acct)


def reset_arm_context(tokens: tuple) -> None:
    """Reset both context vars from the tokens returned by set_arm_context."""
    tok_db, tok_acct = tokens
    _active_db_url.reset(tok_db)
    _active_alpaca_account.reset(tok_acct)
