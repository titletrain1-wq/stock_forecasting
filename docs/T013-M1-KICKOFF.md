# T-013 M1 Kickoff — Intraday Data Pipeline

**Milestone**: M1 (Intraday data pipeline + feature matrix)  
**Design reference**: Design doc §3.2, §4.1, §F8, §F9  
**Status**: Ready to implement  

## Objective

Implement `intraday_trainer.py` to fetch and prepare 365-day intraday training data for M2 (feature engineering) and M3 (model training).

## Scope

1. **Coinbase REST backfill** (design doc §3.2):
   - Fetch 365 calendar days of historical intraday bars (BTC-USD, ETH-USD).
   - Granularity: 5-minute bars (fixed per scope).
   - Write to `intraday_bars_history` table (new immutable ML store, M0 schema).
   - Cost: ~351 requests/ticker (~702 both, acceptable).

2. **dYdX funding-rate as-of join** (design doc §4.1, §F9):
   - Fetch 365d funding-rate snapshots from `crypto_derivatives` table.
   - **Critical**: As-of join (no lookahead, no forward-fill). Value at feature-time $t$ = last funding rate published at or before $t$.
   - Applied as zero-order hold to 5m bars (same value for 12 consecutive 5m bars within each hour).

3. **Closed-bar anchor filtering** (design doc §3.2, §F8):
   - Filter data to closed-bar boundaries only:
     - **1-hour horizon**: `ts` ends at :00 UTC (e.g., 2026-09-01T10:00:00Z).
     - **4-hour horizon**: `ts` ends at :00, :04, :08, :12, :16, :20 UTC (e.g., 2026-09-01T10:00:00Z, 2026-09-01T10:04:00Z, ...).
   - Rationale: Labels (k-step log-returns) must anchor only at closed bars to avoid lookahead bias (§F8).

4. **Immutable ML store** (design doc §3.5, §F2):
   - Data written to `intraday_bars_history` (365d retention).
   - Training pipeline reads ONLY from this table, never from `intraday_bars` (7d display cache).
   - Single-writer pattern (v2): data fetched and written once, then immutable for training/grading.

## Implementation Tasks

### Task 1: Data fetch (Coinbase REST)
- Write function to query Coinbase REST for 365d of 5m bars (both BTC-USD, ETH-USD).
- Use keyless pagination (no API key required; yfinance or curl fallback if Coinbase REST unavailable).
- Error handling: log quota/rate-limit violations; graceful degradation (alert if <90d fetched).

### Task 2: Funding-rate as-of join
- Query `crypto_derivatives` table for 365d of funding snapshots (hourly).
- Merge with 5m bars using as-of join (pandas `merge_asof`):
  - Left: 5m bars, indexed by `ts`.
  - Right: hourly funding rates, indexed by `ts`.
  - Direction: 'backward' (each 5m bar gets the last funding rate published at or before its time).
  - **No forward-fill**; rows with no prior funding rate should be NaN (handled in M2 imputation if needed).

### Task 3: Anchor filtering
- After as-of join, filter DataFrame to closed-bar anchors:
  - For 1h labels: keep rows where `ts.minute == 0 and ts.second == 0`.
  - For 4h labels: keep rows where `(ts.hour % 4 == 0 and ts.minute == 0)` OR `(ts.hour % 4 != 0 and ts.minute in [4, 8, 12, 16, 20])`.
  - Alternative (cleaner): pre-compute valid anchor timestamps, filter to membership.

### Task 4: Tests
- **Test 1**: Mock Coinbase REST response; verify fetched shape is ~8,760 rows/ticker (365d * 24h * 5m granularity).
- **Test 2**: Funding as-of join; verify:
  - No forward-fill (if funding_rate[t] is None and prior rate exists, use prior).
  - Lookahead canary: verify no funding value appears at `t` before its actual publication time.
- **Test 3**: Anchor filtering; verify only valid :00 (1h) and :00/:04/:08/:12/:16/:20 (4h) timestamps remain.
- **Test 4**: End-to-end; write result to temp `intraday_bars_history`, verify rows inserted.

## Dependencies & Constraints

- **Design doc constraints** (all mandatory):
  - §F8 (closed-bar anchors only).
  - §F9 (as-of funding, no lookahead/forward-fill).
  - §3.5 (immutable 365d store, write once).

- **External data sources**:
  - Coinbase REST (or yfinance fallback for historical 5m bars).
  - `crypto_derivatives` table (existing, written by dYdX ingestion job).

- **Output**:
  - `intraday_bars_history` table populated with 365d of closed-bar-anchored bars.
  - Ready for M2 (feature engineering).

## Definition of Done (M1 DoD)

- [ ] `intraday_trainer.py` module created with data fetch + as-of join + anchor filtering.
- [ ] 365d bars fetched for BTC-USD, ETH-USD; written to `intraday_bars_history`.
- [ ] As-of join verified (no forward-fill lookahead per F9).
- [ ] Anchor filtering verified (1h :00, 4h :00/:04/:08/:12/:16/:20 only).
- [ ] All 4 tests passing.
- [ ] Committed on `feat/intraday-t013`.

## Real Failure Criteria (M1 RFC)

- Missing module (intraday_trainer.py).
- Funding forward-filled (lookahead leak per F9).
- Non-closed-bar anchors included (violates §F8).
- Fewer than 90 days fetched (data insufficient for training).
- Tests fail or missing.

## Notes for Fresh Session

- Start from commit 4b9d707 on `feat/intraday-t013`.
- Working tree is clean; no loose changes.
- M0 (schema, config) is complete and approved by god; Jim reviewing async.
- M1 builds on M0 schema immediately (writes to `intraday_bars_history`).
- M2 (feature engineering) will consume output of M1.
