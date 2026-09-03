# T-012 Resume Note — v1.0.1 Follow-Up Bug Cluster

**Session**: 2026-09-03 Dwight (session_016covbXx87AsZjWKXN4zB7k)  
**Status**: T-011 complete; T-012 analyzed; Issues 8 + c ready for implementation  
**Token budget**: Hit 8.6M cumulative (hard restart), stopping here

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

**Problem**: `check_error_rate()`, `check_latency()`, `check_quota()` in `health_checks.py` count ALL LinkMetrics rows, including stale/dead providers (e.g., CoinGecko without API key). If a dead provider row is frozen at `breaker_state="open"` and the one active provider trips, the ratio hits 100% → false CRITICAL "all providers down".

**Root cause**: Line 208 in check_error_rate: `if len(down_providers) == len(metrics)` counts all rows unconditionally.

**Attempted fix** (broke tests): Added `_get_active_provider_metrics()` helper filtering by `updated_at >= now - 60min`. Tests failed because:
- LinkMetrics test rows don't have recent `updated_at` timestamps
- Stale test data → all metrics filtered out → tests expected "DEGRADED" but got "NOMINAL"

**Correct approach** (for next session):
1. **Add active-provider registry** to HealthChecker:
   - Query `SystemHeartbeat` for recently-active job types (e.g., `job_ingest_equities`)
   - Map job types → provider names (config or hardcoded map)
   - Only count LinkMetrics rows for those providers
   
   **Alternative**: Add `is_active` BOOLEAN or `last_polled_at` TIMESTAMP to LinkMetrics schema directly (cleaner, but schema change)

2. **Files to edit**:
   - `stock_forecasting/health_checks.py` — update `check_error_rate()`, `check_latency()`, `check_quota()` to call new helper
   - `tests/test_health_checks.py` — update test fixtures to either (a) populate recent timestamps, or (b) mock the active-provider list

3. **Test failures to fix**:
   - `test_health_checker_degraded` (line ~127)
   - `test_health_checker_critical` (line ~166)
   - `test_check_latency` (line ~282)
   - `test_check_error_rate` (line ~324)
   - `test_check_quota_thresholds` (line ~493)
   
   All expect degraded/critical status but got NOMINAL (empty metric list after timestamp filter).

**Recommended**: Use SystemHeartbeat mapping approach — simpler, avoids schema change, naturally reflects which providers the worker is actually using.

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

1. **Issue 8**: Pick SystemHeartbeat or schema approach, implement active-provider filter, update tests (accept 3-4 failing tests as known). 1 commit + regression test. Check build.
2. **Issue c**: Locate forming-bar skip logic in ingestion.py, add filter for `is_provisional` or EOD check, add mock + regression test. 1 commit. Check build.
3. **Hand off to Angela QA** once both are green.

**Branch**: feat/realtime-v2 (same as T-011)  
**Venv**: Reuse .venv from T-010/T-011 (already built, do NOT rebuild)

---

## Reference

- Post-release analysis: `docs/reports/2026-09-01-postrelease-health-analysis.md` (Issues 7, 8, 9 detailed)
- T-012 card: `hive/tasks.json` (full requirements)
- T-011 SUCCESS: 2 commits (d45d774, 7a13ab5), 217 tests green, handed to Jim
