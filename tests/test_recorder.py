"""Tests for the LLM record/replay module (task #151 MVP)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from agentic_investor.orchestrator.recorder import (
    ReplayMiss,
    _hash_prompt,
    is_recording,
    record_call,
    reset_for_tests,
    try_replay,
)


class _Sample(BaseModel):
    answer: str
    confidence: float


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with recorder state fully cleared and no env
    leaking in from a prior test run."""
    for name in ("AGENTIC_REPLAY_FROM", "AGENTIC_RECORD_TO", "AGENTIC_REPLAY_MISS"):
        monkeypatch.delenv(name, raising=False)
    reset_for_tests()
    yield
    reset_for_tests()


# --- hash --------------------------------------------------------------

def test_prompt_hash_is_stable_across_dict_key_orders():
    """Semantically-identical prompts must hash to the same key even
    when dict keys arrive in different order.
    """
    msgs_a = [{"role": "system", "content": "you are helpful"},
              {"role": "user", "content": "hi"}]
    msgs_b = [{"content": "you are helpful", "role": "system"},
              {"content": "hi", "role": "user"}]
    h_a = _hash_prompt("gpt-4o-mini", "Sample", msgs_a)
    h_b = _hash_prompt("gpt-4o-mini", "Sample", msgs_b)
    assert h_a == h_b


def test_prompt_hash_differs_on_model_change():
    msgs = [{"role": "user", "content": "hi"}]
    assert _hash_prompt("gpt-4o-mini", "S", msgs) != \
           _hash_prompt("anthropic/claude-haiku-4-5", "S", msgs)


def test_prompt_hash_differs_on_response_model_change():
    msgs = [{"role": "user", "content": "hi"}]
    assert _hash_prompt("gpt-4o-mini", "A", msgs) != \
           _hash_prompt("gpt-4o-mini", "B", msgs)


# --- record ------------------------------------------------------------

def test_record_call_appends_jsonl_row(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(tmp_path))
    assert is_recording() is True
    resp = _Sample(answer="yes", confidence=0.9)
    record_call(
        "gpt-4o-mini", "Sample",
        [{"role": "user", "content": "hi"}], resp,
    )
    rec_path = tmp_path / "recording.jsonl"
    assert rec_path.exists()
    rows = [json.loads(ln) for ln in rec_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "llm"
    assert row["model"] == "gpt-4o-mini"
    assert row["response_model"] == "Sample"
    assert row["seq"] == 1
    assert row["prompt_hash"]
    assert row["served_from_replay"] is False
    # response_json round-trips into the same pydantic model
    got = _Sample.model_validate_json(row["response_json"])
    assert got == resp


def test_record_call_noop_when_env_unset(tmp_path):
    """No AGENTIC_RECORD_TO -> record_call is a silent no-op. No file
    is created; is_recording() is False.
    """
    assert is_recording() is False
    record_call("m", "R", [], _Sample(answer="x", confidence=0.5))
    assert not (tmp_path / "recording.jsonl").exists()


def test_record_call_seq_increments(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(tmp_path))
    for i in range(3):
        record_call(
            "m", "Sample",
            [{"role": "user", "content": f"msg {i}"}],
            _Sample(answer=str(i), confidence=0.1 * i),
        )
    rows = [json.loads(ln)
            for ln in (tmp_path / "recording.jsonl").read_text(
                encoding="utf-8").splitlines()]
    assert [r["seq"] for r in rows] == [1, 2, 3]


def test_record_call_survives_non_pydantic_response(tmp_path, monkeypatch):
    """Response without .model_dump_json() falls back to json.dumps
    with default=str. Telemetry must never block the loop.
    """
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(tmp_path))
    record_call("m", "R", [{"role": "user", "content": "x"}], {"weird": "dict"})
    assert (tmp_path / "recording.jsonl").exists()


# --- replay ------------------------------------------------------------

def _seed_recording(tmp_path, entries: list[dict]) -> None:
    """Write a synthetic recording.jsonl."""
    rec = tmp_path / "recording.jsonl"
    with rec.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_try_replay_returns_false_when_env_unset():
    """No AGENTIC_REPLAY_FROM -> try_replay is a no-op that returns
    (False, None) so the caller falls through to the real LLM.
    """
    hit, obj = try_replay(
        "gpt-4o-mini", "Sample",
        [{"role": "user", "content": "hi"}], _Sample,
    )
    assert hit is False
    assert obj is None


def test_try_replay_serves_cached_response_on_hash_match(tmp_path, monkeypatch):
    msgs = [{"role": "user", "content": "what is 2+2?"}]
    resp = _Sample(answer="4", confidence=0.99)
    h = _hash_prompt("gpt-4o-mini", "Sample", msgs)
    _seed_recording(tmp_path, [{
        "kind": "llm",
        "ts": "2026-09-17T15:00:00+00:00",
        "seq": 1,
        "model": "gpt-4o-mini",
        "response_model": "Sample",
        "prompt_hash": h,
        "response_json": resp.model_dump_json(),
    }])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    hit, obj = try_replay("gpt-4o-mini", "Sample", msgs, _Sample)
    assert hit is True
    assert isinstance(obj, _Sample)
    assert obj == resp


def test_try_replay_strict_miss_raises(tmp_path, monkeypatch):
    _seed_recording(tmp_path, [])  # empty recording -> any call is a miss
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    # Default miss policy is 'strict' - anything not in the recording raises.
    with pytest.raises(ReplayMiss) as exc:
        try_replay(
            "gpt-4o-mini", "Sample",
            [{"role": "user", "content": "new prompt"}], _Sample,
        )
    assert "no recorded response" in str(exc.value)
    assert "AGENTIC_REPLAY_MISS=live" in str(exc.value)


def test_try_replay_live_miss_falls_through(tmp_path, monkeypatch):
    """AGENTIC_REPLAY_MISS=live -> hash miss returns (False, None) and
    the caller falls through to the real LLM. Used for A/B testing a
    prompt change against the same market state.
    """
    _seed_recording(tmp_path, [])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    monkeypatch.setenv("AGENTIC_REPLAY_MISS", "live")
    hit, obj = try_replay(
        "gpt-4o-mini", "Sample",
        [{"role": "user", "content": "new prompt"}], _Sample,
    )
    assert hit is False
    assert obj is None


def test_try_replay_missing_response_json_raises(tmp_path, monkeypatch):
    """A recording row with the right hash but no response_json field
    is a corrupt record - strict-miss raises to surface it.
    """
    msgs = [{"role": "user", "content": "hi"}]
    h = _hash_prompt("gpt-4o-mini", "Sample", msgs)
    _seed_recording(tmp_path, [{
        "kind": "llm",
        "seq": 1,
        "model": "gpt-4o-mini",
        "response_model": "Sample",
        "prompt_hash": h,
        # response_json intentionally missing
    }])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    with pytest.raises(ReplayMiss):
        try_replay("gpt-4o-mini", "Sample", msgs, _Sample)


def test_try_replay_bad_response_json_raises(tmp_path, monkeypatch):
    """A hash-hit whose response_json doesn't deserialize into the
    requested model is a schema-drift signal - strict-miss raises with
    a clear message.
    """
    msgs = [{"role": "user", "content": "hi"}]
    h = _hash_prompt("gpt-4o-mini", "Sample", msgs)
    _seed_recording(tmp_path, [{
        "kind": "llm",
        "seq": 1,
        "prompt_hash": h,
        "response_model": "Sample",
        "response_json": '{"answer": "x"}',  # missing 'confidence' field
    }])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    with pytest.raises(ReplayMiss) as exc:
        try_replay("gpt-4o-mini", "Sample", msgs, _Sample)
    assert "failed to deserialize" in str(exc.value)


def test_try_replay_missing_recording_file_falls_through_in_live_mode(
    tmp_path, monkeypatch,
):
    """A missing recording.jsonl behaves like an empty recording:
    strict-miss raises, live-miss falls through.
    """
    # Don't create the file.
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    monkeypatch.setenv("AGENTIC_REPLAY_MISS", "live")
    hit, obj = try_replay(
        "m", "Sample", [{"role": "user", "content": "x"}], _Sample,
    )
    assert hit is False
    assert obj is None


# --- record + replay round-trip ---------------------------------------

def test_record_then_replay_round_trip(tmp_path, monkeypatch):
    """The end-to-end contract: record a call, then replay against the
    same recording, and the same response comes back.
    """
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(tmp_path))

    # 1. Record
    msgs = [{"role": "user", "content": "recall this response"}]
    original = _Sample(answer="captured", confidence=0.42)
    record_call("gpt-4o-mini", "Sample", msgs, original)

    # 2. Replay - clear seq state so a re-record has a clean counter,
    # but keep the recording.jsonl on disk.
    monkeypatch.delenv("AGENTIC_RECORD_TO", raising=False)
    reset_for_tests()
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    hit, obj = try_replay("gpt-4o-mini", "Sample", msgs, _Sample)
    assert hit is True
    assert obj == original


def test_replay_and_record_chain_stamps_served_from_replay(tmp_path, monkeypatch):
    """RECORD_TO active during a replay stamps served_from_replay=True
    on the re-recorded row so replay-of-replay chains stay traceable.
    """
    # Seed the source recording.
    src = tmp_path / "src"
    src.mkdir()
    msgs = [{"role": "user", "content": "hi"}]
    orig = _Sample(answer="cached", confidence=0.7)
    _seed_recording(src, [{
        "kind": "llm",
        "seq": 1,
        "prompt_hash": _hash_prompt("m", "Sample", msgs),
        "model": "m",
        "response_model": "Sample",
        "response_json": orig.model_dump_json(),
    }])
    dst = tmp_path / "dst"
    dst.mkdir()
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(src))
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(dst))

    hit, obj = try_replay("m", "Sample", msgs, _Sample)
    assert hit is True
    record_call("m", "Sample", msgs, obj, served_from_replay=True)
    rows = [
        json.loads(ln)
        for ln in (dst / "recording.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["served_from_replay"] is True


# --- source recording + FIFO replay (clock/price/news) -------------------

def test_record_source_appends_row(tmp_path, monkeypatch):
    from agentic_investor.orchestrator.recorder import record_source
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(tmp_path))
    record_source("clock", {
        "now": "2026-09-17T15:00:00+00:00",
        "is_open": True,
        "next_open": "2026-09-18T13:30:00+00:00",
        "next_close": "2026-09-17T20:00:00+00:00",
    })
    rows = [json.loads(ln) for ln in
            (tmp_path / "recording.jsonl").read_text(
                encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "clock"
    assert rows[0]["is_open"] is True
    assert rows[0]["seq"] == 1


def test_record_source_noop_when_env_unset(tmp_path):
    """No AGENTIC_RECORD_TO -> no file written."""
    from agentic_investor.orchestrator.recorder import record_source
    record_source("clock", {"now": "2026-01-01T00:00:00Z"})
    assert not (tmp_path / "recording.jsonl").exists()


def test_next_from_source_fifo_pop(tmp_path, monkeypatch):
    from agentic_investor.orchestrator.recorder import next_from_source
    _seed_recording(tmp_path, [
        {"kind": "clock", "seq": 1, "now": "T1", "is_open": True},
        {"kind": "clock", "seq": 2, "now": "T2", "is_open": True},
        {"kind": "clock", "seq": 3, "now": "T3", "is_open": False},
    ])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    a = next_from_source("clock")
    b = next_from_source("clock")
    c = next_from_source("clock")
    d = next_from_source("clock")
    assert a["now"] == "T1"
    assert b["now"] == "T2"
    assert c["now"] == "T3"
    assert d is None  # queue exhausted -> caller falls through to live


def test_next_from_source_returns_none_when_env_unset():
    """No AGENTIC_REPLAY_FROM -> always None so caller uses live source."""
    from agentic_investor.orchestrator.recorder import next_from_source
    assert next_from_source("clock") is None


def test_next_from_source_per_kind_queues_dont_starve_each_other(
    tmp_path, monkeypatch,
):
    from agentic_investor.orchestrator.recorder import next_from_source
    _seed_recording(tmp_path, [
        {"kind": "clock", "seq": 1, "now": "T1"},
        {"kind": "price", "seq": 2, "ticker": "AAPL", "price": 200.0},
        {"kind": "clock", "seq": 3, "now": "T2"},
        {"kind": "price", "seq": 4, "ticker": "MSFT", "price": 400.0},
    ])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    # Pop clocks in order; price queue untouched.
    assert next_from_source("clock")["now"] == "T1"
    assert next_from_source("clock")["now"] == "T2"
    assert next_from_source("clock") is None
    # Prices still available in their own order.
    assert next_from_source("price")["ticker"] == "AAPL"
    assert next_from_source("price")["ticker"] == "MSFT"


def test_next_from_source_match_filter_preserves_other_rows(
    tmp_path, monkeypatch,
):
    """Match-based lookup lets arm A pop AAPL-price without consuming
    the MSFT-price rows arm B still needs. Non-matches stay in place.
    """
    from agentic_investor.orchestrator.recorder import next_from_source
    _seed_recording(tmp_path, [
        {"kind": "price", "seq": 1, "ticker": "AAPL", "price": 200.0},
        {"kind": "price", "seq": 2, "ticker": "MSFT", "price": 400.0},
        {"kind": "price", "seq": 3, "ticker": "AAPL", "price": 201.0},
    ])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    # First AAPL request pops seq 1
    a1 = next_from_source("price", match={"ticker": "AAPL"})
    assert a1["price"] == 200.0
    # MSFT request pops seq 2 (still available; not consumed by prior AAPL)
    m1 = next_from_source("price", match={"ticker": "MSFT"})
    assert m1["price"] == 400.0
    # Second AAPL request pops seq 3
    a2 = next_from_source("price", match={"ticker": "AAPL"})
    assert a2["price"] == 201.0
    # All exhausted
    assert next_from_source("price", match={"ticker": "AAPL"}) is None
    assert next_from_source("price", match={"ticker": "MSFT"}) is None


def test_iter_recorded_news_yields_only_news_rows(tmp_path, monkeypatch):
    """iter_recorded_news is separate from next_from_source's FIFO -
    it doesn't consume the queue, so replay drivers can preview or
    inject on their own schedule.
    """
    from agentic_investor.orchestrator.recorder import (
        iter_recorded_news,
        next_from_source,
    )
    _seed_recording(tmp_path, [
        {"kind": "clock", "seq": 1, "now": "T1"},
        {"kind": "news", "seq": 2, "ticker": "AAPL", "headline": "h1"},
        {"kind": "price", "seq": 3, "ticker": "AAPL", "price": 200.0},
        {"kind": "news", "seq": 4, "ticker": "MSFT", "headline": "h2"},
    ])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    news_rows = list(iter_recorded_news())
    assert [r["headline"] for r in news_rows] == ["h1", "h2"]
    # And the FIFO queue for news still has both rows available
    # (iter_recorded_news is read-only, doesn't consume).
    assert next_from_source("news")["headline"] == "h1"
    assert next_from_source("news")["headline"] == "h2"


def test_iter_recorded_news_empty_when_env_unset():
    from agentic_investor.orchestrator.recorder import iter_recorded_news
    assert list(iter_recorded_news()) == []
