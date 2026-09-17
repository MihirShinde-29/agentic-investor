"""News-path memory-leak regression harness (task #154 result).

Runs 6 variants across NewsStreamer._on_news + surrounding chain and
reports private VM growth per event. Fresh subprocess per variant so
PyTorch arena / recorder state don't cross-contaminate.

**Result on 2026-09-17** (post-#148 finBERT padding + #146 chroma fix):
all variants flat at 0-1 KB per event even at N=1000 events. The
"residual ~109 MB/min" figure from the Day-4 observation was
contamination from the corrupt chroma 44 GB one-shot spike, smoothed
into a per-minute rate over the observation window - NOT a genuine
per-event leak.

Kept as regression: run this after any change to news processing
(NewsStreamer, decision_engine, finBERT prefilter, session recorder,
recorder.record_source) and confirm per-event delta stays under
~5 KB. If a real per-event leak reappears this harness will surface
it in seconds.

Variants:
  a_baseline               NewsStreamer._on_news only, no recording
  b_with_record_source     +record_source (task #153) writing jsonl
  c_with_session_log       +SessionRecorder.log per event
  d_with_dedup_growth      pushes unique headlines so _seen dict grows
  e_with_finbert_padded    +score_single on each event (post-#148 fix)
  f_full_chain             all of the above stacked (most realistic)

Usage:
  python scripts/hunt_news_path_leak.py             # run all variants
  python scripts/hunt_news_path_leak.py a_baseline  # single variant
"""

from __future__ import annotations

import ctypes
import gc
import os
import subprocess
import sys
from ctypes import wintypes


def _private_bytes() -> int:
    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    psapi = ctypes.WinDLL("psapi.dll")
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32 = ctypes.WinDLL("kernel32.dll")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(c), ctypes.sizeof(c),
    )
    return int(c.PrivateUsage)


def _mb(b: int) -> int:
    return b // (1024 * 1024)


N_WARMUP = 20
N_MEASURE = 1000


# Real-shape synthetic events; headlines vary in length + ticker so
# dedup + fanout both exercise realistically.
_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
            "JPM", "V", "MA", "CVX", "XOM", "PG", "KO", "PFE"]


def _make_event_class():
    """Build the minimal shim of the Alpaca news item that
    NewsStreamer._on_news expects (has .symbols, .headline, .summary,
    .created_at, .url, .source attrs).
    """
    class _Item:
        __slots__ = ("symbols", "headline", "summary", "created_at",
                     "url", "source")
        def __init__(self, symbols, headline, summary, created_at,
                     url, source):
            self.symbols = symbols
            self.headline = headline
            self.summary = summary
            self.created_at = created_at
            self.url = url
            self.source = source
    return _Item


def _make_events(n: int, *, unique: bool = False):
    """Yield n synthetic news items. `unique=True` produces unseen
    headlines every call to force dedup dict growth.
    """
    _Item = _make_event_class()
    from datetime import UTC, datetime, timedelta
    now = datetime.now(UTC)
    events = []
    for i in range(n):
        ticker = _TICKERS[i % len(_TICKERS)]
        if unique:
            headline = f"{ticker} unique-story-{i} headline text"
        else:
            # Cycle through 20 templates so dedup catches most.
            headline = f"{ticker} template-{i % 20} beats earnings"
        summary = "some summary text " * 20  # ~330 chars, realistic
        ts = now + timedelta(seconds=i)
        events.append(_Item(
            symbols=[ticker],
            headline=headline,
            summary=summary,
            created_at=ts.isoformat(),
            url=f"https://example.com/{i}",
            source="wire",
        ))
    return events


def _make_streamer():
    """Build a NewsStreamer that skips the real websocket."""
    import queue

    from agentic_investor.tools.news_stream import NewsStreamer
    q: queue.Queue = queue.Queue()
    # Wildcard mode so filter doesn't drop anything.
    return NewsStreamer(["*"], event_queue=q, stream_factory=lambda: None), q


async def _drive_events(streamer, events):
    """Push events through the async _on_news callback in-process."""
    for e in events:
        await streamer._on_news(e)


def variant_a_baseline():
    """Just NewsStreamer._on_news, no recording, no session log,
    templated headlines (dedup catches most).
    """
    import asyncio
    import logging
    logging.disable(logging.CRITICAL)
    streamer, _q = _make_streamer()
    for e in _make_events(N_WARMUP):
        asyncio.run(_drive_events(streamer, [e]))
    gc.collect()
    before = _private_bytes()
    events = _make_events(N_MEASURE)
    asyncio.run(_drive_events(streamer, events))
    gc.collect()
    after = _private_bytes()
    _report("a_baseline", before, after, extra=f"queue={_q.qsize()}")


def variant_b_with_record_source():
    """Baseline + task #153 record_source writing to a temp
    recording.jsonl. Tests whether the recording path leaks.
    """
    import asyncio
    import logging
    import tempfile
    logging.disable(logging.CRITICAL)
    tmpdir = tempfile.mkdtemp(prefix="hunt_")
    os.environ["AGENTIC_RECORD_TO"] = tmpdir
    from agentic_investor.orchestrator.recorder import reset_for_tests
    reset_for_tests()
    streamer, _q = _make_streamer()
    for e in _make_events(N_WARMUP):
        asyncio.run(_drive_events(streamer, [e]))
    gc.collect()
    before = _private_bytes()
    events = _make_events(N_MEASURE)
    asyncio.run(_drive_events(streamer, events))
    gc.collect()
    after = _private_bytes()
    _report("b_with_record_source", before, after,
            extra=f"queue={_q.qsize()}, tmpdir={tmpdir}")


def variant_c_with_session_log():
    """Baseline + session.log('news_received', ...) per event.
    Isolates the SessionRecorder + JSONL write cost.
    """
    import asyncio
    import logging
    import tempfile
    logging.disable(logging.CRITICAL)
    from agentic_investor.ops.session import SessionRecorder
    rec = SessionRecorder.start(base_dir=tempfile.mkdtemp(prefix="hunt_sess_"))
    streamer, q = _make_streamer()
    events = _make_events(N_WARMUP)
    for e in events:
        asyncio.run(_drive_events(streamer, [e]))
        # Drain + log like the arm would
        while not q.empty():
            evt = q.get()
            rec.log("news_received", {"ticker": evt.ticker,
                                       "headline": evt.headline[:80]})
    gc.collect()
    before = _private_bytes()
    events = _make_events(N_MEASURE)
    for e in events:
        asyncio.run(_drive_events(streamer, [e]))
        while not q.empty():
            evt = q.get()
            rec.log("news_received", {"ticker": evt.ticker,
                                       "headline": evt.headline[:80]})
    gc.collect()
    after = _private_bytes()
    _report("c_with_session_log", before, after,
            extra=f"queue={q.qsize()}")


def variant_d_with_dedup_growth():
    """Push UNIQUE headlines every event so _seen dedup dict grows
    unboundedly (well, TTL-bounded but under the test window it just
    accumulates).
    """
    import asyncio
    import logging
    logging.disable(logging.CRITICAL)
    streamer, _q = _make_streamer()
    for e in _make_events(N_WARMUP, unique=True):
        asyncio.run(_drive_events(streamer, [e]))
    gc.collect()
    before = _private_bytes()
    events = _make_events(N_MEASURE, unique=True)
    asyncio.run(_drive_events(streamer, events))
    gc.collect()
    after = _private_bytes()
    seen_size = len(streamer._seen)
    _report("d_with_dedup_growth", before, after,
            extra=f"seen_dict={seen_size} entries")


def variant_e_with_finbert_padded():
    """Baseline + score_single (post-#148 padded finBERT) per event."""
    import asyncio
    import logging
    logging.disable(logging.CRITICAL)
    from agentic_investor.orchestrator.finbert_prefilter import score_single
    streamer, q = _make_streamer()
    warmup = _make_events(N_WARMUP)
    for e in warmup:
        asyncio.run(_drive_events(streamer, [e]))
        while not q.empty():
            evt = q.get()
            score_single(f"{evt.headline}. {evt.summary}"[:400])
    gc.collect()
    before = _private_bytes()
    events = _make_events(N_MEASURE)
    for e in events:
        asyncio.run(_drive_events(streamer, [e]))
        while not q.empty():
            evt = q.get()
            score_single(f"{evt.headline}. {evt.summary}"[:400])
    gc.collect()
    after = _private_bytes()
    _report("e_with_finbert_padded", before, after,
            extra=f"queue={q.qsize()}")


def variant_f_full_chain():
    """Everything stacked: dedup growth + session log + finBERT +
    record_source. Closest to what a real arm at market-close does.
    """
    import asyncio
    import logging
    import tempfile
    logging.disable(logging.CRITICAL)
    tmpdir = tempfile.mkdtemp(prefix="hunt_")
    os.environ["AGENTIC_RECORD_TO"] = tmpdir
    from agentic_investor.ops.session import SessionRecorder
    from agentic_investor.orchestrator.finbert_prefilter import score_single
    from agentic_investor.orchestrator.recorder import reset_for_tests
    reset_for_tests()
    rec = SessionRecorder.start(base_dir=tempfile.mkdtemp(prefix="hunt_sess_"))
    streamer, q = _make_streamer()
    warmup = _make_events(N_WARMUP, unique=True)
    for e in warmup:
        asyncio.run(_drive_events(streamer, [e]))
        while not q.empty():
            evt = q.get()
            rec.log("news_received", {"ticker": evt.ticker,
                                       "headline": evt.headline[:80]})
            score_single(f"{evt.headline}. {evt.summary}"[:400])
    gc.collect()
    before = _private_bytes()
    events = _make_events(N_MEASURE, unique=True)
    for e in events:
        asyncio.run(_drive_events(streamer, [e]))
        while not q.empty():
            evt = q.get()
            rec.log("news_received", {"ticker": evt.ticker,
                                       "headline": evt.headline[:80]})
            score_single(f"{evt.headline}. {evt.summary}"[:400])
    gc.collect()
    after = _private_bytes()
    _report("f_full_chain", before, after,
            extra=f"queue={q.qsize()}, seen={len(streamer._seen)}")


def _report(name, before, after, *, extra=""):
    delta = after - before
    per_call = delta // N_MEASURE
    print(
        f"{name}: delta={_mb(delta)} MB "
        f"({per_call // 1024} KB/event, {per_call} B/event) "
        f"[{_mb(before)} -> {_mb(after)}] {extra}"
    )


VARIANTS = {
    "a_baseline": variant_a_baseline,
    "b_with_record_source": variant_b_with_record_source,
    "c_with_session_log": variant_c_with_session_log,
    "d_with_dedup_growth": variant_d_with_dedup_growth,
    "e_with_finbert_padded": variant_e_with_finbert_padded,
    "f_full_chain": variant_f_full_chain,
}


def _run_all():
    print(f"hunt_residual_leak: {len(VARIANTS)} variants, "
          f"N_MEASURE={N_MEASURE}, N_WARMUP={N_WARMUP}")
    print("-" * 78)
    for name in VARIANTS:
        cp = subprocess.run(
            [sys.executable, __file__, name],
            capture_output=True, text=True, timeout=900,
            env={**os.environ},
        )
        if cp.returncode != 0:
            print(f"{name}: SUBPROCESS FAILED rc={cp.returncode}")
            print(cp.stdout[-500:])
            print(cp.stderr[-500:])
            continue
        for line in cp.stdout.strip().splitlines():
            print(line)


def main():
    if len(sys.argv) == 2 and sys.argv[1] in VARIANTS:
        VARIANTS[sys.argv[1]]()
        return
    _run_all()


if __name__ == "__main__":
    main()
