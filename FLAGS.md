<!--
This file is auto-generated from src/agentic_investor/flags.py.
Do NOT edit by hand: run `python scripts/regen_flags_md.py`
after adding or changing a flag. Task #158 wires this into a
pre-commit hook (see scripts/precommit.sh) so drift can't ship.
-->

# Feature flags

Every `AGENTIC_*` env var the paper-loop reads, sourced from the
central `flags` registry. Values shown are defaults.

| Env var | Kind | Default | Description |
| --- | --- | --- | --- |
| `AGENTIC_ARM_ID` | str | `solo` | Which arm this process is (used to scope M17 retrieval, tag session events, etc.). 'solo' when running as a single-arm paper-loop; 'A'/'B'/'C' etc. in a paper-experiment run. |
| `AGENTIC_CITE_TO_TRADE` | bool_01 | `1` | Trade-gate: '1' requires news citations for order-flow; '0' opens the gate (arm A in the reasoning-quality experiment). |
| `AGENTIC_ENSEMBLE_MODELS` | csv | `(empty)` | Cross-family ensemble models (CSV of LiteLLM model strings). Empty = single-model. Example: 'gpt-4o-mini,anthropic/claude-haiku-4-5'. |
| `AGENTIC_JEV_MATERIALITY_ENABLED` | bool_01 | `0` | When '1', an arm routes the news-materiality prefilter decision through Jev's Noul primitive instead of the finBERT + threshold stack. Cheaper AND typed - no hallucination possible. Arm C in the current 3-arm A/B enables this; A stays off as control. Falls back to the existing finBERT prefilter path if Jev fails. |
| `AGENTIC_JEV_VERDICT_ENABLED` | bool_01 | `0` | When '1', an arm renders verdict labels (WORKING/WRONG/UNCLEAR/TOO-EARLY) on its own recent fills + drift-skips into the allocator prompt's fast_tail so the LLM sees which of its own recent decisions worked. Backed by Jev (Choice primitive). Falls back to a deterministic threshold heuristic if the Jev call fails. Arm B in the current 3-arm A/B enables this; A stays off as control. |
| `AGENTIC_LOG_ROTATE_KEEP` | int | `5` | How many rotated log backups to keep. Total per-arm on-disk log budget = ROTATE_MB * (ROTATE_KEEP + 1). |
| `AGENTIC_LOG_ROTATE_MB` | int | `20` | Per-arm log file max size in MB before rotation. On rotate, the current file is renamed .1, prior .1 becomes .2, etc. |
| `AGENTIC_MAX_PROMPT_TOKENS` | int | `100000` | Soft cap on total allocator prompt tokens. When exceeded, the regen raises PromptTooLargeError and the tick is skipped rather than letting the provider 400 us. |
| `AGENTIC_MEMORY_RAG` | bool_01 | `1` | Whether the allocator prompt includes retrieved past decisions (M17). '1' enables, '0' disables. Kill-switch for A/B tests. |
| `AGENTIC_MEMORY_RAG_K` | int | `4` | How many past-decision precedents to retrieve on each regen. |
| `AGENTIC_MEM_RECYCLE_MB` | int | `0` | Watchdog threshold in MB. When this process's PrivateUsage crosses the threshold, the arm exits 42 for supervisor respawn. 0 disables the watchdog (default outside paper-experiment supervisor). |
| `AGENTIC_ML_SERVICE_URL` | str_opt | `(unset)` | HTTP URL of the shared paper-ml-service (task #145). When set, arms route finBERT + embed calls through the service instead of loading the models locally. Unset = local fallback. |
| `AGENTIC_NEWS_BUS` | str_opt | `(unset)` | sqlite:/// URL of the shared news bus. Set by paper-experiment supervisor; unset arms fall through to a direct Alpaca websocket. |
| `AGENTIC_NEWS_BUS_TTL_HOURS` | float | `4.0` | Retention window for news_bus.db bus_events rows. Sweeper drops rows older than this every 5 min. 4h = 4x the STALE cutoff. |
| `AGENTIC_NEWS_STORE_TTL_DAYS` | float | `30.0` | Retention window for the news-article sqlite-vec store (news_articles + vec_news tables). Older rows are dropped from both tables in lockstep every hour. |
| `AGENTIC_PRICE_BUS` | str_opt | `(unset)` | sqlite:/// URL of the shared price bus (same pattern as news bus). |
| `AGENTIC_PRICE_BUS_MAX_SYMBOLS` | int | `15` | Max concurrent Alpaca websocket trade-subscriptions the shared price bus will hold. Alpaca paper returns 'symbol limit exceeded (405)' on the whole subscribe call once the cap is exceeded. Nominal documented cap is 30 but paper accounts on IEX-only feeds hit 405 at lower thresholds in practice - 15 is the safe default confirmed against a live paper-experiment run. Freshness-based eviction: newest updated_at wins. Bump if you upgrade to a data plan with a higher cap. |
| `AGENTIC_PRICE_BUS_TTL_HOURS` | float | `2.0` | Retention window for price_bus.db price_ticks rows. |
| `AGENTIC_REPLAY_NEWS_SPEED` | float | `1.0` | News-replay playback multiplier. 1.0 = real-time (respects recorded inter-event gaps), 2.0 = twice as fast, 0 = fire everything as fast as the queue drains. Only consulted when AGENTIC_REPLAY_FROM is set. |
| `AGENTIC_SELF_CONSISTENCY_N` | int | `0` | Number of self-consistency samples for the allocator. 0/1 = single call. Overridden by AGENTIC_ENSEMBLE_MODELS when both set. |
| `AGENTIC_WHIPSAW_GUARD_WINDOW_MIN` | int | `15` | Minutes of look-back for the whipsaw guard. When a fresh order on ticker T would reverse the direction of the last trade on T within this window AND T isn't in the current news batch's trigger_tickers (i.e. no fresh news to justify the flip), the order is dropped and a knob_fired name=whipsaw_guard event is logged. Motivated by the 2026-09-14/18 A/B: arm C flipped CRM 6+ times/day and B's best-hit-rate arm still net-lost from asymmetric single-tick reversals. 0 disables (arm A baseline). |
