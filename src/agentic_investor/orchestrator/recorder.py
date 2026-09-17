"""LLM call recorder + replayer for deterministic session rerun.

Scope note: this module captures the *LLM* input/output side of a paper
session. Full deterministic replay (news + price + clock capture too)
is documented as follow-up in the task. LLM-only is the highest-value
slice because:
  1. LLM calls are what cost money and can vary run-to-run
  2. A/B testing prompt changes against a fixed session is the most
     common "let me rerun this" use case
  3. The other inputs (news events, price ticks, market clock) are
     already logged in session.jsonl for post-hoc reconstruction

Design:
- Recording is a JSONL file (`recording.jsonl`) next to session.jsonl.
  Each line is one LLM call: model, prompt hash, response JSON.
- Two orthogonal env-var switches (both can be set at once):
    AGENTIC_REPLAY_FROM=<dir>   read + serve cached responses from
                                <dir>/recording.jsonl
    AGENTIC_RECORD_TO=<dir>     append every completed call to
                                <dir>/recording.jsonl
- Miss policy (only relevant when RECORD_TO isn't set to catch new
  calls; when RECORD_TO is set, misses fall through to the live LLM
  and get captured):
    AGENTIC_REPLAY_MISS=strict  (default) raise ReplayMiss on any
                                hash miss - guarantees deterministic
                                replay, catches accidental drift
    AGENTIC_REPLAY_MISS=live    fall through to the real LLM on miss.
                                Combined with RECORD_TO, this is the
                                "let this prompt change replay against
                                the same market state" A/B mode.

Prompt hash key: sha256 over the normalized JSON of
(model, response_model.__name__, messages). Deep-normalization is
important: LiteLLM's OpenAI adapter mutates messages in place
(cache_control markers), so we hash BEFORE the call not after.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_ENV_REPLAY_FROM = "AGENTIC_REPLAY_FROM"
_ENV_RECORD_TO = "AGENTIC_RECORD_TO"
_ENV_MISS_POLICY = "AGENTIC_REPLAY_MISS"


class ReplayMiss(Exception):
    """Raised in strict-miss mode when a prompt hash isn't in the
    recording. Contains the offending model + hash + first N chars of
    the first message so the caller can eyeball the mismatch.
    """


def _hash_prompt(model: str, response_model_name: str,
                 messages: list[dict]) -> str:
    """sha256 over the tuple (model, response_model_name, messages).

    `json.dumps(sort_keys=True)` for canonicalization: dict key order
    is irrelevant to the semantic prompt, so we sort. Nested dicts +
    lists preserve order.
    """
    payload = {
        "model": model,
        "response_model": response_model_name,
        "messages": messages,
    }
    canon = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _recording_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / "recording.jsonl"


# --- replay side (read cached response) ---------------------------------

_replay_lock = threading.Lock()
_replay_cache: dict[str, dict] | None = None
_replay_source: Path | None = None


def _load_replay_cache() -> dict[str, dict]:
    """Load and index the recording once per process. Cached; re-reads
    are no-ops.
    """
    global _replay_cache, _replay_source
    src = os.environ.get(_ENV_REPLAY_FROM)
    if not src:
        return {}
    with _replay_lock:
        if _replay_cache is not None and _replay_source == Path(src):
            return _replay_cache
        cache: dict[str, dict] = {}
        rec_path = _recording_path(src)
        if not rec_path.exists():
            logger.warning(
                "replay source %s has no recording.jsonl; will fall through",
                rec_path,
            )
        else:
            n = 0
            with rec_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("kind") != "llm":
                        continue
                    h = row.get("prompt_hash")
                    if not h:
                        continue
                    cache[h] = row
                    n += 1
            logger.info("loaded %d LLM records from %s", n, rec_path)
        _replay_cache = cache
        _replay_source = Path(src)
        return cache


def try_replay(
    model: str,
    response_model_name: str,
    messages: list[dict],
    response_model_type: type,
) -> tuple[bool, Any]:
    """Attempt to serve this call from the recording.

    Returns (hit, response). `hit=True` means the caller should return
    `response` directly. `hit=False` means the caller must proceed with
    the real LLM call (either because RECORD_TO isn't set to replay
    from, or because the miss policy is 'live').

    Raises ReplayMiss when the miss policy is 'strict' and the hash
    isn't in the recording. Strict is the default; it catches
    accidental prompt drift between record and replay runs.
    """
    src = os.environ.get(_ENV_REPLAY_FROM)
    if not src:
        return (False, None)
    cache = _load_replay_cache()
    h = _hash_prompt(model, response_model_name, messages)
    row = cache.get(h)
    if row is None:
        policy = os.environ.get(_ENV_MISS_POLICY, "strict").lower()
        if policy == "live":
            return (False, None)
        first_msg = (messages[0].get("content", "") if messages else "")
        raise ReplayMiss(
            f"no recorded response for hash={h[:12]}... "
            f"model={model} response_model={response_model_name} "
            f"first_msg={str(first_msg)[:120]!r}. "
            f"Set AGENTIC_REPLAY_MISS=live to fall through to the "
            f"real LLM (useful for A/B testing prompt changes)."
        )
    resp_json = row.get("response_json")
    if not resp_json:
        raise ReplayMiss(
            f"recording hash={h[:12]} has no response_json field",
        )
    try:
        obj = response_model_type.model_validate_json(resp_json)
    except Exception as e:
        raise ReplayMiss(
            f"recorded response for hash={h[:12]} failed to deserialize "
            f"into {response_model_name}: {e}"
        ) from e
    return (True, obj)


# --- record side (append + persist) -------------------------------------

_record_lock = threading.Lock()
_record_seq = 0


def is_recording() -> bool:
    return bool(os.environ.get(_ENV_RECORD_TO))


def record_call(
    model: str,
    response_model_name: str,
    messages: list[dict],
    response: Any,
    *,
    served_from_replay: bool = False,
) -> None:
    """Append one LLM call to `AGENTIC_RECORD_TO/recording.jsonl`.

    `response` must be a Pydantic BaseModel (or something with
    `.model_dump_json()`). Anything else is str()'d as a last resort so
    telemetry never blocks the loop.

    `served_from_replay=True` is stamped into the row so a re-record of
    a replay can distinguish "this was a live call" from "this came
    out of an upstream recording" - useful in the RECORD + REPLAY
    combined mode for chains of replay-of-replay.
    """
    dest_dir = os.environ.get(_ENV_RECORD_TO)
    if not dest_dir:
        return
    dest = _recording_path(dest_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    global _record_seq
    try:
        try:
            resp_json = response.model_dump_json()
        except AttributeError:
            resp_json = json.dumps(response, default=str)
        row = {
            "kind": "llm",
            "ts": datetime.now(UTC).isoformat(),
            "seq": None,  # filled inside the lock
            "model": model,
            "response_model": response_model_name,
            "prompt_hash": _hash_prompt(model, response_model_name, messages),
            "response_json": resp_json,
            "served_from_replay": bool(served_from_replay),
        }
        with _record_lock:
            _record_seq += 1
            row["seq"] = _record_seq
            with dest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    except Exception as e:  # noqa: BLE001 - telemetry never blocks the loop
        logger.debug("record_call failed: %s", e)


def reset_for_tests() -> None:
    """Clear process-wide record + replay state. Tests use this so a
    prior test's recording doesn't leak into the next.
    """
    global _replay_cache, _replay_source, _record_seq
    with _replay_lock:
        _replay_cache = None
        _replay_source = None
    with _record_lock:
        _record_seq = 0
