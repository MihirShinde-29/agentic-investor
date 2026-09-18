"""Retrieval-quality regression harness (task #163 / F).

Golden queries with expected top-K rec_ids over a stable seed corpus.
Guards against drift in the M17 retrieval SQL, similarity math, arm
filtering, date cutoff, and outcome-tiebreak logic.

Runs two ways:

- **Deterministic mode (default, fast, always on):** uses a keyword-
  bag embedder so a bit-flip in ranking / filter / SQL is caught
  without needing sentence-transformers on disk. Good enough for a
  CI gate.

- **Real-embed mode (opt-in via `AGENTIC_RUN_REAL_EMBED_HARNESS=1`):**
  routes through the actual `_default_embed`, which pulls
  sentence-transformers weights and exercises the same code path as
  production. Skipped by default because the weights are ~90 MB and
  the test would flake on a fresh CI machine. Run locally with
  `AGENTIC_RUN_REAL_EMBED_HARNESS=1 pytest tests/test_retrieval_regression.py`
  before shipping a change to the embedder or the retrieval SQL.

Recall floor rationale: the top-2 golden retrieval for each query
must include the ground-truth-relevant rec_id. Anything looser
wouldn't catch a real regression; anything tighter starts to test
the embedder's ordering rather than the wrapper.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import pytest

from agentic_investor.memory.retrieval import retrieve_similar
from agentic_investor.memory.store import (
    EMBED_DIM,
    open_memory_conn,
)

# --- Deterministic keyword-bag embedder ---------------------------------
#
# Not a semantic model - it hashes each word into K slots and adds a 1.0.
# Similarity between two texts is dominated by shared vocabulary. That's
# enough to exercise the retrieval wrapper (SQL, filters, tiebreak,
# dedup) with a rigid ground truth, without the flake of a real model.

_TOKENIZE_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def _keyword_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-words -> `EMBED_DIM`-vector. Same word ends up in the
    same slot across calls (sha1-keyed), so semantically-adjacent
    queries score high on the intended docs.
    """
    out = []
    for t in texts:
        vec = [0.0] * EMBED_DIM
        words = {w.lower() for w in _TOKENIZE_RE.findall(t)}
        for w in words:
            slot = int(hashlib.sha1(w.encode()).hexdigest()[:8], 16) % EMBED_DIM
            vec[slot] = 1.0
        # L2-normalize so retrieval math (L2 distance -> similarity)
        # behaves like a cosine on unit vectors.
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        out.append(vec)
    return out


class _Doc(NamedTuple):
    rec_id: int
    text: str
    tickers: str  # comma-joined
    tags: set[str]  # ground-truth topic tags for golden mapping


# Seed corpus - each doc has a distinct semantic center + tags. Kept
# small so a human can eyeball a regression without decoding the
# hash-slot embedder.
CORPUS: list[_Doc] = [
    _Doc(1, "NVIDIA beats AI GPU demand earnings estimates record",
         "NVDA", {"ai", "nvda", "earnings"}),
    _Doc(2, "AMD accelerator revenue Instinct AI chip datacenter",
         "AMD", {"ai", "amd", "chip"}),
    _Doc(3, "TSMC foundry advanced node yields customer volumes",
         "TSM", {"chip", "tsm", "foundry"}),
    _Doc(4, "OPEC oil supply cut Brent crude price barrel",
         "XOM,CVX", {"oil", "energy", "opec"}),
    _Doc(5, "Chevron Permian production dividend energy sector",
         "CVX", {"oil", "cvx", "dividend"}),
    _Doc(6, "First Republic bank deposit outflow FDIC receivership",
         "FRC", {"bank", "crisis", "frc"}),
    _Doc(7, "Regional bank stress deposits KRE ETF concerns",
         "KRE", {"bank", "crisis", "kre"}),
    _Doc(8, "Moderna cancer vaccine phase trial data results",
         "MRNA", {"biotech", "trial", "mrna"}),
    _Doc(9, "Pfizer FDA approval rare disease drug launch",
         "PFE", {"biotech", "fda", "pfe"}),
    _Doc(10, "Tesla vehicle tariff China production plant Shanghai",
          "TSLA", {"ev", "tsla", "tariff"}),
    _Doc(11, "Ford Rivian electric truck production ramp",
          "F,RIVN", {"ev", "truck"}),
    _Doc(12, "Apple iPhone launch services revenue quarter guidance",
          "AAPL", {"aapl", "consumer", "earnings"}),
]


# Golden queries + expected topics: retrieval's top-2 must contain a rec
# whose tag set overlaps with the expected topic set.
GOLDEN = [
    ("NVIDIA earnings AI GPU demand", {"ai", "nvda"}),
    ("AMD accelerator AI chip datacenter", {"ai", "amd"}),
    ("OPEC oil supply cut Brent barrel", {"oil", "opec"}),
    ("bank deposit outflow crisis regional", {"bank", "crisis"}),
    ("Moderna trial phase cancer vaccine", {"biotech", "mrna"}),
    ("Tesla tariff China Shanghai production", {"ev", "tsla"}),
    ("Apple iPhone launch quarterly guidance", {"aapl", "consumer"}),
]


@pytest.fixture
def seeded_conn():
    """Fresh in-memory rec-store, seeded with the corpus above under
    source=historical so retrieve_similar's arm filter picks it up
    with include_historical=True.
    """
    conn = open_memory_conn()
    created_at = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    # Insert into recs + vec_recs directly (rather than the higher-level
    # upsert path) so the test doesn't depend on Recommendation shape.
    embeddings = _keyword_embed([d.text for d in CORPUS])
    import struct
    for doc, emb in zip(CORPUS, embeddings, strict=True):
        conn.execute(
            "INSERT INTO recs (rec_id, source, created_at, tickers, text, "
            " n_positions, avg_confidence, cash_pct, risk) "
            "VALUES (?, 'historical', ?, ?, ?, ?, ?, ?, ?)",
            (doc.rec_id, created_at, doc.tickers, doc.text,
             len(doc.tickers.split(",")), 0.7, 5.0, "moderate"),
        )
        conn.execute(
            "INSERT INTO vec_recs (rowid, embedding) VALUES (?, ?)",
            (doc.rec_id, struct.pack(f"{EMBED_DIM}f", *emb)),
        )
    yield conn
    conn.close()


def _tags_for(rec_id: int) -> set[str]:
    for d in CORPUS:
        if d.rec_id == rec_id:
            return d.tags
    return set()


@pytest.mark.parametrize("query,expected_tags", GOLDEN)
def test_top2_contains_expected_topic(query, expected_tags, seeded_conn):
    """Recall@2 gate: at least one of the top-2 results carries the
    expected topic tag. Tight enough to catch a broken query / SQL /
    similarity flip; loose enough that a ranking swap between the #1
    and #2 candidates on the same topic doesn't false-alarm.
    """
    got = retrieve_similar(
        query, arm_id="test",
        k=2,
        conn=seeded_conn,
        embedder=_keyword_embed,
    )
    assert got, f"no results returned for query {query!r}"
    top_tags = set()
    for r in got:
        top_tags |= _tags_for(r.rec_id)
    assert top_tags & expected_tags, (
        f"query={query!r} top rec_ids={[r.rec_id for r in got]} "
        f"tags={top_tags} did not intersect expected {expected_tags}"
    )


def test_mean_recall_at_2_meets_floor(seeded_conn):
    """Aggregate floor: at least 85 % of golden queries must land at
    least one expected-tag doc in the top-2. Below that we probably
    regressed the pipeline; above that a single golden hiccup on a
    hash collision doesn't fail CI.
    """
    hits = 0
    for query, expected_tags in GOLDEN:
        got = retrieve_similar(
            query, arm_id="test",
            k=2,
            conn=seeded_conn,
            embedder=_keyword_embed,
        )
        top_tags = set()
        for r in got:
            top_tags |= _tags_for(r.rec_id)
        if top_tags & expected_tags:
            hits += 1
    recall = hits / len(GOLDEN)
    assert recall >= 0.85, (
        f"recall@2 = {recall:.2%} across {len(GOLDEN)} golden queries "
        f"is below the 85% regression floor"
    )


def test_arm_source_filter_excludes_other_arms(seeded_conn):
    """Sanity: a doc tagged arm_B must NOT surface for arm_id='A'
    even if it's a perfect match. Cross-arm leakage would poison the
    A/B experiment.
    """
    import struct
    # Insert an arm_B doc that would win on similarity to the query.
    emb = _keyword_embed(["special super rare unique word xyzzy"])[0]
    seeded_conn.execute(
        "INSERT INTO recs (rec_id, source, created_at, tickers, text, "
        " n_positions, avg_confidence, cash_pct, risk) "
        "VALUES (?, 'arm_B', ?, ?, ?, 1, 0.5, 0.0, 'moderate')",
        (999, datetime.now(UTC).isoformat(), "XYZ",
         "special super rare unique word xyzzy"),
    )
    seeded_conn.execute(
        "INSERT INTO vec_recs (rowid, embedding) VALUES (?, ?)",
        (999, struct.pack(f"{EMBED_DIM}f", *emb)),
    )
    got = retrieve_similar(
        "special super rare unique xyzzy",
        arm_id="A",
        k=3,
        conn=seeded_conn,
        embedder=_keyword_embed,
    )
    assert all(r.rec_id != 999 for r in got), (
        f"arm_B doc leaked into arm_A retrieval: {[r.rec_id for r in got]}"
    )


def test_max_age_days_filter_drops_stale_docs(seeded_conn):
    """A rec older than max_age_days must be dropped even if it's
    the top match on similarity. Guards the age-cutoff clamp we
    added when retrieval started surfacing 2-year-old recs.
    """
    import struct
    # Old but perfect-match doc.
    old_ts = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    emb = _keyword_embed(["ancient corpus unique marker phrase"])[0]
    seeded_conn.execute(
        "INSERT INTO recs (rec_id, source, created_at, tickers, text, "
        " n_positions, avg_confidence, cash_pct, risk) "
        "VALUES (?, 'historical', ?, ?, ?, 1, 0.5, 0.0, 'moderate')",
        (500, old_ts, "OLD", "ancient corpus unique marker phrase"),
    )
    seeded_conn.execute(
        "INSERT INTO vec_recs (rowid, embedding) VALUES (?, ?)",
        (500, struct.pack(f"{EMBED_DIM}f", *emb)),
    )
    got = retrieve_similar(
        "ancient corpus unique marker phrase",
        arm_id="test",
        k=3,
        max_age_days=90,
        conn=seeded_conn,
        embedder=_keyword_embed,
    )
    assert all(r.rec_id != 500 for r in got), (
        f"stale doc leaked past max_age_days=90: {[r.rec_id for r in got]}"
    )


@pytest.mark.skipif(
    os.environ.get("AGENTIC_RUN_REAL_EMBED_HARNESS") != "1",
    reason="opt-in only: run with AGENTIC_RUN_REAL_EMBED_HARNESS=1 to "
           "exercise the real sentence-transformer embedder (needs "
           "~90 MB of model weights on disk)",
)
def test_real_embedder_hits_top1_on_direct_paraphrase(seeded_conn):
    """Opt-in: pull the real embedder + a single golden that a real
    semantic model would obviously get right. Failing this means the
    embedder itself regressed (model swap, wrong tokenizer, etc.).
    """
    # Re-seed with real embeddings so distances make sense.
    import struct

    from agentic_investor.memory.rec_index import _default_embed
    seeded_conn.execute("DELETE FROM vec_recs")
    real_embs = _default_embed([d.text for d in CORPUS])
    for doc, emb in zip(CORPUS, real_embs, strict=True):
        seeded_conn.execute(
            "INSERT INTO vec_recs (rowid, embedding) VALUES (?, ?)",
            (doc.rec_id, struct.pack(f"{EMBED_DIM}f", *emb)),
        )
    got = retrieve_similar(
        "NVIDIA earnings AI GPU demand beat estimates",
        arm_id="test",
        k=1,
        conn=seeded_conn,
        embedder=_default_embed,
    )
    assert got and got[0].rec_id == 1, (
        f"real embedder no longer picks the NVIDIA doc top-1: "
        f"got {[r.rec_id for r in got]}"
    )
