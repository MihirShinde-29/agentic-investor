"""Recorder + replayer for deterministic paper-session rerun.

Captures FOUR kinds of session inputs so a full replay is bit-exact
against the recorded market conditions:

  llm    - one row per structured_complete call (model + prompt hash +
           response JSON). Retrieval by prompt hash (random-access).
  clock  - one row per PaperBroker.get_clock() (is_open + next_open +
           next_close). Retrieval by FIFO order.
  price  - one row per PriceBusClient.get_latest(ticker) (ticker +
           price). Retrieval by FIFO order per-ticker (see keying note).
  news   - one row per NewsStreamer callback firing (ticker + headline
           + summary + published_at). Retrieval by FIFO order.

All four kinds share the same recording.jsonl file, distinguished by
`kind`. The LLM slice landed in task #151 MVP; the other three landed
in task #153 as the "full replay" follow-up.

Design:
- Recording is a JSONL file (`recording.jsonl`) next to session.jsonl.
- Two orthogonal env-var switches (both can be set at once):
    AGENTIC_REPLAY_FROM=<dir>   read from <dir>/recording.jsonl
    AGENTIC_RECORD_TO=<dir>     append every capture to
                                <dir>/recording.jsonl
- LLM miss policy (only relevant when RECORD_TO isn't set to catch new
  calls; when RECORD_TO is set, misses fall through to the live LLM
  and get captured):
    AGENTIC_REPLAY_MISS=strict  (default) raise ReplayMiss on hash miss
    AGENTIC_REPLAY_MISS=live    fall through to the real LLM on miss

Source-kind (clock/price/news) semantics:
- FIFO-ordered replay: nth call returns nth recorded row of that kind.
- Empty queue -> return None -> caller falls through to live source.
  Rationale: hitting end-of-recording is a signal to switch back to
  live, not a hard error. Strict mode isn't wired for source kinds
  because the fallthrough is usually what you want.
- Per-kind queues so a hot path (price) doesn't starve a cold one
  (clock).

Prompt hash key (LLM only): sha256 over the normalized JSON of
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
from collections import deque
from collections.abc import Iterator
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
    _reset_source_queues()


# --- source (clock/price/news) side ---------------------------------------
#
# LLM records are keyed by prompt hash (random-access lookup). Source
# records are keyed by (kind, arrival-order) - the nth invocation of
# get_clock() replays the nth recorded clock row. Separate FIFO queue
# per kind so a hot source (price) doesn't starve a cold one (clock).

_source_queues_lock = threading.Lock()
_source_queues: dict[str, deque[dict]] | None = None
_source_queues_from: Path | None = None


def _load_source_queues() -> dict[str, deque[dict]]:
    """Read every non-LLM row from the current AGENTIC_REPLAY_FROM
    recording and bucket into per-kind FIFO queues. Loaded once per
    replay source; re-reads are no-ops.
    """
    global _source_queues, _source_queues_from
    src = os.environ.get(_ENV_REPLAY_FROM)
    if not src:
        return {}
    with _source_queues_lock:
        if _source_queues is not None and _source_queues_from == Path(src):
            return _source_queues
        queues: dict[str, deque[dict]] = {}
        rec_path = _recording_path(src)
        if rec_path.exists():
            n_total = 0
            with rec_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = row.get("kind")
                    if kind in (None, "llm"):
                        # LLM has its own hash-keyed cache path.
                        continue
                    queues.setdefault(kind, deque()).append(row)
                    n_total += 1
            logger.info(
                "loaded %d source events across %d kinds from %s",
                n_total, len(queues), rec_path,
            )
        _source_queues = queues
        _source_queues_from = Path(src)
        return queues


def _reset_source_queues() -> None:
    global _source_queues, _source_queues_from
    with _source_queues_lock:
        _source_queues = None
        _source_queues_from = None


def next_from_source(
    kind: str,
    *,
    match: dict | None = None,
) -> dict | None:
    """Return the next recorded row for this kind, or None if the
    queue is exhausted or the recording had no rows of this kind.

    `match` (optional): key/value pairs the row must equal. Enables
    per-ticker price retrieval when arms poll in a different order
    than the recording produced. Non-matching rows are LEFT IN PLACE
    so a later request with a different match can still find them.

    Caller falls through to the live source on None. This is a
    'graceful trailoff' pattern: a replay can safely run past the
    end of a recording by resuming live capture.
    """
    src = os.environ.get(_ENV_REPLAY_FROM)
    if not src:
        return None
    queues = _load_source_queues()
    q = queues.get(kind)
    if not q:
        return None
    with _source_queues_lock:
        if match is None:
            try:
                return q.popleft()
            except IndexError:
                return None
        # Linear scan for the first row that matches every key/value in
        # `match`. On hit, remove that row from the deque (preserves
        # order for the remaining rows) and return it. O(n) per call
        # in the worst case but acceptable at our scale (recordings
        # rarely exceed ~10k rows).
        for i, row in enumerate(q):
            if all(row.get(k) == v for k, v in match.items()):
                del q[i]
                return row
        return None


def iter_recorded_news() -> Iterator[dict]:
    """Yield every recorded 'news' row in seq order from the current
    AGENTIC_REPLAY_FROM recording, WITHOUT consuming the FIFO queue
    that `next_from_source('news')` walks.

    Used by news-replay drivers that want to inject recorded events
    into the arm's news queue at startup (see the follow-up to
    task #153). Iterator-style so a caller can gate on published_at
    for time-based playback rather than dumping the whole log at once.
    """
    src = os.environ.get(_ENV_REPLAY_FROM)
    if not src:
        return iter(())
    rec_path = _recording_path(src)
    if not rec_path.exists():
        return iter(())

    def _gen() -> Iterator[dict]:
        with rec_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "news":
                    yield row

    return _gen()


def recorded_now() -> datetime:
    """Deterministic-replay-aware `datetime.now(UTC)`.

    - Replay mode: pop the next recorded 'now' from the FIFO. Empty
      queue falls through to real datetime.now (graceful trailoff so a
      replay past the recording's end still works).
    - Record mode: get the real time, capture it, return it.
    - Both env vars off: identical to datetime.now(UTC), zero overhead
      beyond the wrapper call.

    Use for any wall-clock reads in the decision path (retrieval max-
    age filters, correlation lookback windows, rebalancer day stamp)
    so a bit-exact replay is possible. Ordinary datetime.now(UTC) in
    telemetry / logging paths is fine to leave unchanged.
    """
    replayed = next_from_source("now")
    if replayed is not None:
        iso = replayed.get("iso")
        if iso:
            try:
                return datetime.fromisoformat(iso.replace("Z", "+00:00"))
            except ValueError:
                pass
    now = datetime.now(UTC)
    if os.environ.get(_ENV_RECORD_TO):
        record_source("now", {"iso": now.isoformat()})
    return now


def recorded_uuid_hex(n_chars: int = 16) -> str:
    """Deterministic-replay-aware short UUID hex.

    Same shape as `uuid.uuid4().hex[:n_chars]`. Same rules as
    `recorded_now`: replay pops from FIFO, record captures the fresh
    value.

    Used for the client_order_id fallback in paper_broker (production
    code path derives a deterministic id from rec_id + ticker + side +
    day + qty via _client_order_id, so this only bites manual /
    one-off submissions; wrapping it makes replay reproducible for
    those too).
    """
    replayed = next_from_source("uuid_hex")
    if replayed is not None:
        h = replayed.get("hex")
        if isinstance(h, str):
            return h[:n_chars]
    import uuid
    h = uuid.uuid4().hex[:n_chars]
    if os.environ.get(_ENV_RECORD_TO):
        record_source("uuid_hex", {"hex": h})
    return h


def record_source(kind: str, data: dict) -> None:
    """Append a source-event row to AGENTIC_RECORD_TO/recording.jsonl.

    `data` must be JSON-serializable (or fall through to str via
    json.dumps default). `kind` should be one of the well-known
    strings ('clock', 'price', 'news') so replay's FIFO lookup finds
    it, but the module doesn't enforce this - new kinds can be added
    at their capture sites.
    """
    dest_dir = os.environ.get(_ENV_RECORD_TO)
    if not dest_dir:
        return
    dest = _recording_path(dest_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    global _record_seq
    try:
        row = {
            "kind": kind,
            "ts": datetime.now(UTC).isoformat(),
            "seq": None,  # filled inside the lock
            **data,
        }
        with _record_lock:
            _record_seq += 1
            row["seq"] = _record_seq
            with dest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:  # noqa: BLE001 - telemetry never blocks the loop
        logger.debug("record_source(%s) failed: %s", kind, e)
