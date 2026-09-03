# T-012 Resume Note — v1.0.1 Follow-Up Bug Cluster

**Session**: 2026-09-03 Dwight (session_016covbXx87AsZjWKXN4zB7k, cont'd)  
**Status**: T-011 ✅ COMPLETE (Jim re-review); T-012 Issue 8 ✅ COMPLETE (232/232 tests green); Issue c NOT STARTED  
**Token budget**: 14.8M/15M used (T-011 r2 + Issue 8 complete)

---

## Issue 7: eval_attempts on bar-missing skip

**Status**: ✅ **ALREADY DONE** — no action needed for T-012

**Evidence**:
- Code: `stock_forecasting/evaluator.py:119` increments `snap.eval_attempts` when `bar is None`
- Test: `tests/test_evaluator.py::test_evaluator_missing_realized_bar_increments_attempts` (lines 152–191)
  - Verifies increment on first miss, second miss, confirms no `evaluated_at` set
  - Runs twice to show persistent retry counter

**Implementation detail**: evaluator skips snapshots with no realized bar, increments counter for re-run, logs outcome. Exactly as designed.

---

## Issue 8: Scope health checks to active providers

**Status**: ✅ **COMPLETE** — Commit f56c062  
**Result**: All 232 tests green ✓

**Implementation**:
1. **Added `_get_active_providers()` helper** in `health_checks.py`:
   - Queries `SystemHeartbeat` for recently-pulsed jobs (within 60 min threshold)
   - Maps job types → provider names via `JOB_TYPE_TO_PROVIDERS` dict
   - Returns set of currently-active provider names

2. **Updated 3 health check methods**:
   - `check_latency()`, `check_error_rate()`, `check_quota()` filter LinkMetrics by active providers
   - Fallback behavior: if no active providers detected, use all metrics (for setup/testing)
   - Prevents false CRITICAL when stale provider rows trip circuit breaker

3. **Test fixture fix**:
   - Removed ingest job heartbeats from `_seed_healthy_state()` fixture (they were masking heartbeat lag in watchdog tests)
   - Individual check tests pass with fallback; system-status tests pass with proper heartbeat lag detection

**Root cause of initial failure**: Added ingest jobs with fresh pulses to fixture, which check_watchdog used instead of the lagged job_heartbeat, making lag undetectable.

**Solution**: Keep fixture minimal; active-provider filtering works correctly once heartbeat masking issue is resolved.

---

## Issue c: Coinbase forming partial candle should not be ingested as closed

**Problem**: Coinbase ingestion polls new crypto bars. The forming candle for the current UTC day (not yet closed) is marked `is_provisional=1` in the intraday_bars table. But daily bar ingestion in `ingestion.py` may be treating forming bars as finalized and persisting them as closed bars → violates spec "closed bars only" → features/forecast can be anchored to a partial-day bar.

**Suspected entry point**: `stock_forecasting/ingestion.py` — likely around the daily bar upsert logic. Check:
- `IngestionService.poll_ticker()` (line ~100?)
- Provider fetch methods that return daily bars
- Coinbase provider daily bar endpoint parsing

**Fix approach**:
1. Before upserting to `ohlcv_bars`, filter out bars where `is_provisional=1` (if that field propagates)
2. Or: check the bar's `ts` against the current UTC time — if `ts` is "today" and we're not yet EOD, skip it
3. Add regression test: mock a Coinbase response with a forming bar for today, verify it's NOT persisted

**Note**: The intraday_bars table is read-only for display (streaming live price); the issue is daily ohlcv_bars ingestion. May need to distinguish Coinbase intraday (real-time stream, can be provisional) from Coinbase daily (should be closed only).

---

## Next session — Fresh-Dwight:

**Only Issue c remains** (Issue 7 already done, Issue 8 complete):

1. **Issue c: Forming-bar filter in ingestion**: Locate daily bar upsert logic in `ingestion.py`; add filter to skip `is_provisional=1` rows before persisting to `ohlcv_bars`. 1 commit + regression test. Scope: ~30 minutes.
2. **Approach**: Either (a) skip bars with `is_provisional=1` flag propagating from provider response, or (b) check bar `ts` against current UTC time and skip if "today" but not EOD. Add mock Coinbase response test.
3. **Hand off to Angela QA** once green (both Issue 8 + c).

**Branch**: feat/realtime-v2 (same as T-011)  
**Venv**: Reuse .venv from T-010/T-011 (already built, do NOT rebuild)  
**Baseline**: All 232 tests green; no regressions expected from Issue 8 cleanup

---

## Reference

- Post-release analysis: `docs/reports/2026-09-01-postrelease-health-analysis.md` (Issues 7, 8, 9 detailed)
- T-012 card: `hive/tasks.json` (full requirements)
- T-011 SUCCESS: 2 commits (d45d774, 7a13ab5), 217 tests green, handed to Jim
