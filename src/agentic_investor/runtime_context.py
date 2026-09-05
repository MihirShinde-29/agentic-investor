"""Per-request ContextVars for the multi-arm dashboard.

Middleware sets these when the client passes `?arm=X`. paper_store /
paper_broker / orchestrator.store helpers pick them up transparently
so we don't thread arm routing through every function signature.

Defaults are None; unset means fall back to the process ambient
(settings.database_url + primary Alpaca account).
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
    return _active_db_url.get()


def get_active_alpaca_account() -> str | None:
    return _active_alpaca_account.get()


def set_arm_context(
    db_url: str | None, alpaca_account: str | None,
) -> tuple:
    return (
        _active_db_url.set(db_url),
        _active_alpaca_account.set(alpaca_account),
    )


def reset_arm_context(tokens: tuple) -> None:
    tok_db, tok_acct = tokens
    _active_db_url.reset(tok_db)
    _active_alpaca_account.reset(tok_acct)
