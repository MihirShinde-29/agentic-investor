"""End-to-end deterministic-replay integration test (task #160 / B).

Records every recorder-integrated source in one recording, then plays
it back and asserts bit-exact match:

  llm       via record_call + try_replay (hash-keyed)
  clock     via record_source("clock") + next_from_source("clock")
  price     via record_source("price") + next_from_source("price",
            match={ticker}) - per-ticker ordering
  now       via recorded_now (round-trip through datetime.isoformat)
  uuid_hex  via recorded_uuid_hex (32-char hex, sliced)
  news      via record_source("news") + NewsReplayDriver

The point is not to re-cover the per-module unit tests but to catch
integration bugs that only show when several kinds live in the same
recording.jsonl and are consumed by different code paths on replay.
"""

from __future__ import annotations

import json
import queue as _q
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from agentic_investor.orchestrator import recorder
from agentic_investor.orchestrator.news_replay import NewsReplayDriver


class _AllocSample(BaseModel):
    action: str
    weight: float


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "AGENTIC_REPLAY_FROM", "AGENTIC_RECORD_TO", "AGENTIC_REPLAY_MISS",
        "AGENTIC_REPLAY_NEWS_SPEED",
    ):
        monkeypatch.delenv(name, raising=False)
    recorder.reset_for_tests()
    yield
    recorder.reset_for_tests()


def _do_recording_phase(rec_dir, monkeypatch):
    """Populate a full recording. Returns the values we captured so the
    replay phase can assert bit-exact equality.
    """
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(rec_dir))
    recorder.reset_for_tests()

    # --- LLM: two distinct prompts.
    llm_a_msgs = [{"role": "user", "content": "allocate AAPL"}]
    llm_b_msgs = [{"role": "user", "content": "allocate NVDA"}]
    resp_a = _AllocSample(action="buy", weight=0.6)
    resp_b = _AllocSample(action="hold", weight=0.4)
    recorder.record_call("gpt-4o-mini", "_AllocSample", llm_a_msgs, resp_a)
    recorder.record_call("gpt-4o-mini", "_AllocSample", llm_b_msgs, resp_b)

    # --- Clock ticks (two).
    clocks = [
        {"now": "2026-09-17T14:30:00+00:00", "is_open": True,
         "next_open": "2026-09-18T13:30:00+00:00",
         "next_close": "2026-09-17T20:00:00+00:00"},
        {"now": "2026-09-17T14:31:00+00:00", "is_open": True,
         "next_open": "2026-09-18T13:30:00+00:00",
         "next_close": "2026-09-17T20:00:00+00:00"},
    ]
    for c in clocks:
        recorder.record_source("clock", c)

    # --- Prices (interleaved tickers, per-ticker order matters).
    prices = [
        {"ticker": "AAPL", "price": 175.10},
        {"ticker": "NVDA", "price": 120.55},
        {"ticker": "AAPL", "price": 175.25},
        {"ticker": "NVDA", "price": 120.90},
    ]
    for p in prices:
        recorder.record_source("price", p)

    # --- Recorded now (three samples).
    nows_recorded: list[datetime] = []
    for _ in range(3):
        nows_recorded.append(recorder.recorded_now())

    # --- Recorded uuid_hex (three samples).
    uuids_recorded: list[str] = [
        recorder.recorded_uuid_hex() for _ in range(3)
    ]

    # --- News (three events, ascending received_at so replay ordering
    # matches recording order).
    base = datetime(2026, 9, 17, 14, 30, 0, tzinfo=UTC)
    news_rows = []
    for i, tk in enumerate(("AAPL", "NVDA", "TSLA")):
        received = (base + timedelta(seconds=i * 5)).isoformat()
        row = {
            "ticker": tk,
            "headline": f"headline-{tk}",
            "summary": f"summary-{tk}",
            "published_at": received,
            "received_at": received,
            "url": f"https://ex/{tk}",
            "source": "test",
        }
        news_rows.append(row)
        recorder.record_source("news", row)

    monkeypatch.delenv("AGENTIC_RECORD_TO", raising=False)
    return {
        "llm": [
            ("gpt-4o-mini", "_AllocSample", llm_a_msgs, resp_a),
            ("gpt-4o-mini", "_AllocSample", llm_b_msgs, resp_b),
        ],
        "clocks": clocks,
        "prices": prices,
        "nows": nows_recorded,
        "uuids": uuids_recorded,
        "news": news_rows,
    }


def test_full_recording_replays_bit_exact(tmp_path, monkeypatch):
    captured = _do_recording_phase(tmp_path, monkeypatch)

    # Sanity: the recording file exists and holds every kind we wrote.
    rec_path = tmp_path / "recording.jsonl"
    assert rec_path.exists()
    rows = [
        json.loads(line)
        for line in rec_path.read_text(encoding="utf-8").splitlines()
    ]
    kinds = {r.get("kind") for r in rows}
    assert kinds == {"llm", "clock", "price", "now", "uuid_hex", "news"}
    # Seq must be monotone starting at 1 (per-process counter, single-thread
    # here so no gaps possible).
    seqs = [r["seq"] for r in rows]
    assert seqs == list(range(1, len(seqs) + 1))

    # --- Flip to replay mode.
    recorder.reset_for_tests()
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    monkeypatch.setenv("AGENTIC_REPLAY_MISS", "strict")

    # LLM: both prompts hit + return exact recorded payload.
    for model, resp_name, msgs, expected_resp in captured["llm"]:
        hit, obj = recorder.try_replay(
            model, resp_name, msgs, _AllocSample,
        )
        assert hit is True
        assert obj == expected_resp

    # LLM strict-miss: an unrecorded prompt raises.
    with pytest.raises(recorder.ReplayMiss):
        recorder.try_replay(
            "gpt-4o-mini", "_AllocSample",
            [{"role": "user", "content": "unrecorded prompt"}],
            _AllocSample,
        )

    # Clocks: FIFO replay.
    for expected in captured["clocks"]:
        row = recorder.next_from_source("clock")
        assert row is not None
        for k, v in expected.items():
            assert row[k] == v
    # Queue exhausted -> None (graceful trailoff).
    assert recorder.next_from_source("clock") is None

    # Prices: per-ticker FIFO. Ask in a different order than we
    # recorded to verify the `match` filter picks the right row and
    # leaves the others in place.
    p1 = recorder.next_from_source("price", match={"ticker": "AAPL"})
    p2 = recorder.next_from_source("price", match={"ticker": "AAPL"})
    p3 = recorder.next_from_source("price", match={"ticker": "NVDA"})
    p4 = recorder.next_from_source("price", match={"ticker": "NVDA"})
    assert [p1["price"], p2["price"]] == [175.10, 175.25]
    assert [p3["price"], p4["price"]] == [120.55, 120.90]
    assert recorder.next_from_source(
        "price", match={"ticker": "AAPL"},
    ) is None

    # now: three replays should return the recorded datetimes exactly.
    replayed_nows = [recorder.recorded_now() for _ in range(3)]
    for got, expected in zip(replayed_nows, captured["nows"], strict=True):
        # Compare isoformats to sidestep tzinfo identity - the recorder
        # writes/reads via .isoformat() so the round-trip is exact.
        assert got.isoformat() == expected.isoformat()
    # Post-exhaustion recorded_now falls through to live datetime.now.
    tail = recorder.recorded_now()
    assert isinstance(tail, datetime)

    # uuid_hex: three replays match exactly.
    for expected in captured["uuids"]:
        assert recorder.recorded_uuid_hex() == expected

    # News: NewsReplayDriver drains recorded rows into a queue in the
    # order they were recorded.
    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=0)
    drv.start()
    drv.join(timeout=5.0)
    got_events = []
    while not q.empty():
        got_events.append(q.get_nowait())
    assert [e.ticker for e in got_events] == [
        r["ticker"] for r in captured["news"]
    ]
    assert [e.headline for e in got_events] == [
        r["headline"] for r in captured["news"]
    ]


def test_hash_stability_survives_message_dict_key_order(tmp_path, monkeypatch):
    """A recorded prompt with keys {role, content} must still replay a
    hit when the caller sends {content, role}. Regression guard so a
    future dict-construction reshuffle in llm.client doesn't silently
    invalidate every recording.
    """
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(tmp_path))
    recorder.reset_for_tests()

    msgs_recorded = [{"role": "user", "content": "hello"}]
    resp = _AllocSample(action="buy", weight=0.5)
    recorder.record_call("gpt-4o-mini", "_AllocSample", msgs_recorded, resp)

    recorder.reset_for_tests()
    monkeypatch.delenv("AGENTIC_RECORD_TO", raising=False)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    msgs_reordered = [{"content": "hello", "role": "user"}]
    hit, obj = recorder.try_replay(
        "gpt-4o-mini", "_AllocSample", msgs_reordered, _AllocSample,
    )
    assert hit is True
    assert obj == resp


def test_replay_miss_live_policy_falls_through(tmp_path, monkeypatch):
    """AGENTIC_REPLAY_MISS=live returns (False, None) instead of raising,
    so a caller can A/B-test a prompt change against a recording.
    """
    monkeypatch.setenv("AGENTIC_RECORD_TO", str(tmp_path))
    recorder.reset_for_tests()
    recorder.record_call(
        "gpt-4o-mini", "_AllocSample",
        [{"role": "user", "content": "old prompt"}],
        _AllocSample(action="buy", weight=0.5),
    )
    recorder.reset_for_tests()
    monkeypatch.delenv("AGENTIC_RECORD_TO", raising=False)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))
    monkeypatch.setenv("AGENTIC_REPLAY_MISS", "live")

    hit, obj = recorder.try_replay(
        "gpt-4o-mini", "_AllocSample",
        [{"role": "user", "content": "new prompt"}],
        _AllocSample,
    )
    assert hit is False
    assert obj is None
