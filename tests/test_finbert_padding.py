"""Regression test: finBERT scoring doesn't grow the PyTorch arena
per-call after task #148's padding fix.

Pre-fix (2026-09-17): every score_single/score_headlines call could
grow PyTorch's caching allocator arena because tensor shapes varied
with input length (real headlines range 10-500 chars). At 100 news
events/min, arms grew ~6 MB/min just from finBERT alone.

Post-fix: pipeline is called with padding='max_length', max_length=128
so tensors are always shape (batch, 128) and the arena stabilizes
after the first inference call. This test pins that shape by measuring
committed private VM before + after 50 varied-length calls.

Skipped when finBERT isn't installed / can't load (CI without model
weights) so it stays as an opt-in local regression signal.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import sys

import pytest


def _private_bytes() -> int:
    """Best-effort own-process PrivateUsage. Windows-only via
    GetProcessMemoryInfo; Linux via /proc/self/status VmData; other
    platforms return 0.
    """
    import platform
    if platform.system() == "Windows":
        try:
            from ctypes import wintypes

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
            counters = _PMC()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi.dll")
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            k32 = ctypes.WinDLL("kernel32.dll")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.GetCurrentProcess.argtypes = []
            ok = psapi.GetProcessMemoryInfo(
                k32.GetCurrentProcess(),
                ctypes.byref(counters),
                ctypes.sizeof(counters),
            )
            return int(counters.PrivateUsage) if ok else 0
        except Exception:  # noqa: BLE001
            return 0
    if platform.system() == "Linux":
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmData:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return 0


_SAMPLES = [
    "Apple beats Q3 earnings estimates on strong iPhone sales",
    "Federal Reserve signals rate cut",
    "Nvidia stock jumps 5%",
    "Boeing 737 MAX faces new FAA scrutiny after door plug incident on Alaska "
    "Airlines flight and multiple prior safety investigations by the FAA and "
    "pressure from Congress on the manufacturer's supply chain and QA",
    "Tesla",
    "JPMorgan raises price target on Amazon to $250",
    "Oil prices retreat 2% after OPEC+ production announcement",
]


@pytest.mark.slow
def test_score_single_arena_stays_bounded_after_padding_fix():
    """Post-#148: pipeline is padded to max_length=128 so tensor shape
    is (1, 128) regardless of input. Arena should stabilize by
    warmup + 50 more calls should add less than 3 MB (was ~6 MB pre-fix).

    Marked slow because it loads finBERT (~1.5 GB weights) - skips
    gracefully when transformers/torch isn't available.
    """
    logging.disable(logging.CRITICAL)
    try:
        from agentic_investor.orchestrator.finbert_prefilter import score_single
    except ImportError:  # pragma: no cover - defensive
        pytest.skip("finbert_prefilter unavailable")

    # Warmup: load weights + let the first call size the arena.
    warmup_result = score_single(_SAMPLES[0])
    if warmup_result is None:
        pytest.skip("finBERT pipeline could not be initialized")
    for i in range(1, 5):
        score_single(_SAMPLES[i % len(_SAMPLES)])
    gc.collect()

    before = _private_bytes()
    if before == 0:
        pytest.skip("PrivateUsage sampling unavailable on this platform")

    for i in range(50):
        score_single(_SAMPLES[i % len(_SAMPLES)])
    gc.collect()

    after = _private_bytes()
    delta_mb = (after - before) / (1024 * 1024)
    # 3 MB is the guard rail: pre-fix was ~3 MB in 50 calls (6 MB/100);
    # post-fix should be near zero. Bounded at 3 MB gives us a signal
    # if the fix regresses without being flaky on noise.
    assert delta_mb < 3.0, (
        f"finBERT arena grew {delta_mb:.1f} MB over 50 padded calls; "
        f"pre-#148 baseline was ~3 MB and post-fix is near 0. "
        f"Padding likely regressed - check "
        f"finbert_prefilter._pipeline_scores for padding='max_length'."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
