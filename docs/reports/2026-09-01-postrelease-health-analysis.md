# Post-Release Health Analysis — stock_forecasting v1.0.0

**Date:** 2026-09-01T08:40Z
**Analyst:** Toby (hive)
**Repo state:** branch `main`, HEAD `0e15758` (tag `v1.0.0` @ `b10ffe9`), 149/149 tests green, ruff clean.
**Scope:** ANALYSIS ONLY — no source edits, no commits. Live DB read at `~/Desktop/stock_forecasting/data/app.db`; worker pid 93555 running (heartbeats current as of 08:35Z).

---

## Environment facts established first (so verdicts are correct)

- **2026-09-01 is a Tuesday and a normal US trading day.** US Labor Day 2026 is **Sept 7** (first Monday). The task's "equity markets CLOSED" note is wrong. At 08:40Z it is 04:40 ET — ~5 h before the 13:30Z (09:30 ET) equity open.
- **The system ingests DAILY bars only.** `YFinanceProvider.get_daily_history` / `get_latest_bars`, `CoinbaseProvider` daily candles. There is no intraday/streaming bar anywhere. Latest bars in DB: equities `2026-08-31T00:00Z` (Mon close — correct, today's bar cannot exist pre-open), crypto `2026-09-01T00:00Z` (source `coinbase`, from a manual backfill at ~07:03Z, not the worker).
- **The running worker only has two providers registered:** `yfinance` + `fake` (`worker.py:93-98 _init_default_providers`). `coingecko`, `coinbase`, `tiingo`, `finnhub`, and the dYdX derivatives provider are never instantiated by the worker. The 731 crypto bars and the `link_metrics` rows for coingecko/coinbase came from an out-of-band backfill earlier this session, not from the worker loop.
- **Worker (re)started ~08:30Z** (heartbeat rows) — god's re-run. `job_evaluate_hourly` (1 h interval) and `job_retrain_nightly` (cron h=22) have not fired yet this session; no heartbeat rows for them.

---

## Issues

### ISSUE 1 — Freshness check has no market-hours / bar-interval awareness → system is CRITICAL by construction
- **Symptom:** System health = CRITICAL. `[FRESHNESS]` "severely stale" — AAPL/NVDA/SPY ~1952 m, BTC-USD/ETH-USD ~512 m.
- **Root cause:** `health_checks.py:54-123 check_freshness`. It computes `age_sec = now - latest_bar.ts` and applies flat wall-clock thresholds: NOMINAL `<= 300 s` crypto / `<= 1200 s` equity, DEGRADED `<= 3600 s`, else **CRITICAL** (`health_checks.py:89-104`). There is no market calendar, no "next expected bar" logic, no bar-interval input. Against **daily** bar timestamps this can never pass:
  - Equity: last bar `2026-08-31T00:00Z`, age ≈ 1960 min → CRITICAL. This is the *correct* pre-market state of a daily feed, flagged as a severe fault.
  - Crypto: last bar `2026-09-01T00:00Z`, age ≈ 520 min → CRITICAL. A once-a-day bar is > 1 h old for 23 of every 24 h even when the feed is perfectly healthy.
- **Spec conflict (this is a real deviation, not a judgement call):** design spec §6 / freshness row (`docs/2026-09-01-stock-forecasting-design.md:296`) says *"bar age vs expected (**market-hours + 15m delay aware**)"* and *"🔴 >1h **in-hours**"*; line 210 mandates *"Equities: `pandas-market-calendars` for market hours + holidays. Crypto: 24/7 calendar."* None of that is implemented.
- **Classification:** REAL BUG.
- **Severity:** CRITICAL — it is the sole driver of the overall CRITICAL status (`compute_system_status` promotes any CRITICAL check to system-CRITICAL, `health_checks.py:463`). Every other check is NOMINAL.
- **Recommended fix:** Make freshness "expected next bar" aware. Minimum viable: for equities, if now is outside market hours (or the market was closed on the most recent session and today's session bar isn't due yet) treat the last session's close bar as NOMINAL; only apply the >1 h → CRITICAL rule when the market is open and a bar is genuinely overdue. For crypto (daily bars), the threshold must be ~1 day + grace, not 5 min. Introduce a per-ticker `expected_bar_interval` and derive thresholds from it. Use `pandas-market-calendars` as the spec already requires. **Effort:** M (0.5–1 day incl. tests; `pandas-market-calendars` is already a documented dependency in the spec).

### ISSUE 2 — Worker registers only `yfinance` + `fake`; documented provider failover is entirely non-functional
- **Symptom:** worker log `No registered provider for ticker BTC-USD/ETH-USD (primary coingecko)`; crypto data never updates; no failover to Coinbase.
- **Root cause:** `worker.py:93-98 _init_default_providers()` returns `{"yfinance": ..., "fake": ...}` only. `job_ingest_crypto` / `job_ingest_equities` build `IngestionService(session, self.providers)` with that dict and pass **no** `derivatives_provider` (`worker.py:127`, `worker.py:152`). Then:
  - `ingestion.py:80-89 _provider_chain("coingecko")` = `["coingecko", "coinbase"]` filtered by `pid in self.providers` → **empty** → `ingestion.py:100-110` logs the "No registered provider" warning and returns `{"inserted": 0, "error": "Provider 'coingecko' not found"}`.
  - Equity path "works" only because `yfinance` happens to be the one real provider present; `FALLBACK_PROVIDERS["yfinance"] = ["tiingo", "finnhub"]` (`ingestion.py:33`) are also absent, so there is **no** equity failover either.
- **Sub-issues:**
  - **2a. Crypto bar ingest is 100 % dead in the running app** (every 60 s poll fails). CRITICAL.
  - **2b. Equity ingest has no fallback** — single point of failure. `KNOWN_LIMITATIONS.md §2` claims *"Tiingo is configured as the functional primary fallback for Yahoo Finance"* and *"Coinbase … serves as the primary working fallback for cryptocurrency"* — both false for the worker. HIGH.
  - **2c. Crypto derivatives (M9) ingest is dead** — no `derivatives_provider` passed, and separately there is **no scheduled job** that calls `poll_all_derivatives` at all (`worker.py:start` registers ingest_crypto, ingest_equities, retrain_nightly, evaluate_hourly, heartbeat — nothing for derivatives). MED.
- **Classification:** REAL BUG (2a/2b/2c all).
- **Severity:** CRITICAL (2a), HIGH (2b), MED (2c).
- **Recommended fix:** Build the full provider map in the worker from settings/env: `yfinance`, `tiingo` (if `TIINGO_API_KEY`), `finnhub` (if key), `coingecko` (if `COINGECKO_API_KEY`), `coinbase` (keyless — always), and a dYdX `DerivativesProvider`. Pass `derivatives_provider=` to `IngestionService`. Add a `job_ingest_derivatives` scheduled job (crypto cadence). Add a startup log line enumerating which providers registered so a missing key is visible. **Effort:** M (0.5–1 day; provider classes already exist and are tested — this is wiring + a config helper + one new job).

### ISSUE 3 — `job_ingest_crypto` reports heartbeat SUCCESS while every crypto poll fails
- **Symptom:** Watchdog + error-rate health checks stay green; `system_heartbeat.job_ingest_crypto` shows `consecutive_failures=0`, `last_success_ts` current — yet crypto data is hours stale. "Everything looks fine but the data isn't moving."
- **Root cause:** `ingestion.py` `poll_ticker` / `_fetch_with_failover` return an **error dict** (`{"inserted": 0, "error": ...}`), they never raise. `worker.py:115-138 job_ingest_crypto` loops `ingestion_service.poll_ticker(...)`, ignores the return values, and unconditionally calls `_update_heartbeat(session, "job_ingest_crypto", success=True)` (`worker.py:130`). The `except` at `worker.py:131` is never entered. Same pattern in `job_ingest_equities`.
- **Classification:** REAL BUG.
- **Severity:** HIGH — this defeats the entire watchdog/observability story; a total ingest outage is invisible.
- **Recommended fix:** Inspect poll results — if all tickers for a job returned `error` (or `inserted==0` when an insert was expected), record `success=False` with an aggregated `error_msg`, or emit a distinct degraded signal. Feed per-provider failures into `CircuitBreaker.record_failure` / `LinkMetrics` even on the "provider not found" path so `check_error_rate` can see it. **Effort:** S–M (0.5 day).

### ISSUE 4 — Confidence-interval bands double-count time: `residual_std * sqrt(h)` applied twice
- **Symptom:** 30-day CI band explodes the chart y-axis. Observed snapshot bounds:
  - AAPL 30d: anchor 316.85, predicted 318.4, **band 63.9 – 1585.95**.
  - NVDA 30d: anchor 220.78, **band 33.8 – 1621.97**.
  - SPY 30d: anchor 767.05, **band 295.2 – 1893.25**.
  - BTC-USD 30d: anchor 78 722, **band 11 456 – 619 574**.
  - ETH-USD 30d: anchor 2 472, **band 74.6 – 97 135** (≈ 40× spot).
- **Root cause:** The training target is the **h-day cumulative log return**: `trainer.py:187` `target_series = np.log(close_series.shift(-h) / close_series)`. So `residual_std` (`trainer.py:249-252`, std of walk-forward residuals) is **already in h-horizon units**. It is then scaled by `sqrt(h)` **a second time** in two places:
  - `trainer.py:256` `scaled_std = residual_std * np.sqrt(h)` — used for the `wf_ci_cov` back-test (`trainer.py:258-262`).
  - `forecaster.py:238` `scaled_std = float(residual_std * np.sqrt(h_days))` → `forecaster.py:239-240` `lower/upper = predicted_price * exp(∓1.96 * scaled_std)`.
  - The raw (unscaled) `residual_std` is what's persisted to the artifact and `model_runs` (`trainer.py:291, 305, 345, 363`), so the forecaster re-inflates it.
- **Effect:** 1d correct (√1 = 1); 5d ≈ 2.24× too wide; 30d ≈ 5.48× too wide. Empirically confirmed: AAPL 30d `residual_std` 0.1496 → `*√30` = 0.819 → `exp(±1.96·0.819)` = ×0.20 / ×4.96 → 63.9 / 1586. Matches exactly.
- **Self-masking:** `wf_ci_cov` in `model_runs` reads **1.0** for every 5d and 30d model — the metric that should have flagged over-wide bands is computed with the *same* bug (`trainer.py:256`), so it always looks perfectly covered. 1d coverage is a healthy ~0.94–0.96.
- **Classification:** REAL BUG.
- **Severity:** HIGH — forecast correctness (bands are meaningless) + primary chart UX is broken at the 30d view.
- **Recommended fix:** Remove the `* np.sqrt(h)` at both `forecaster.py:238` and `trainer.py:256` (use `scaled_std = residual_std`). Re-run the nightly trainer to regenerate artifacts + `wf_ci_cov` (expect 5d/30d coverage to drop toward ~0.95). Add a test asserting `wf_ci_cov` is in ~[0.90, 0.98] for a known fixture so a future re-introduction fails CI. **Effort:** S for the fix, M including retrain + coverage test + regenerating the 15 model artifacts.

### ISSUE 5 — `coingecko` provider stuck showing "🟡 RECOVERING" indefinitely
- **Symptom:** coingecko provider chip = RECOVERING and never clears.
- **Root cause:** two layers.
  - `link_metrics.coingecko` has `consecutive_failures = 2`, `breaker_state = "closed"`, `updated_at = 2026-09-01T07:03:15Z` (≈ 1.5 h stale). Those 2 failures are from the ~07:03 manual backfill hitting CoinGecko without a Demo key (HTTP 401 — see `KNOWN_LIMITATIONS.md §2`). Because the worker never registers coingecko, it is never called again, so `CircuitBreaker.record_success` (`circuit_breaker.py:108-120`, which zeroes `consecutive_failures`) never runs. The counter is frozen.
  - `health_view.py:92-93`: `if m.breaker_state == "half_open" or (m.consecutive_failures or 0) > 0: return "🟡 RECOVERING"`. Any non-zero failure count on an otherwise-**closed** breaker with 0 % error rate is rendered as "recovering", which is misleading.
- **Classification:** REAL BUG (display + stale-row) layered on an ENVIRONMENTAL cause (no `COINGECKO_API_KEY`, documented). If Issue 2 is fixed so `coinbase` is the working crypto provider, `coingecko` should not be a primary at all.
- **Severity:** LOW–MED (cosmetic/misleading; no data impact once Issue 2 is fixed).
- **Recommended fix:** (a) Don't surface providers in the health strip that aren't in the active provider set / haven't been polled within N minutes (stale `updated_at`). (b) Tighten the RECOVERING rule: only when `breaker_state == "half_open"`, or breaker recently transitioned closed← open, not merely `consecutive_failures > 0` on a long-closed breaker. (c) Make `coinbase` the crypto primary (swap `tickers.provider` for BTC-USD/ETH-USD, or make coingecko→coinbase failover actually work per Issue 2). **Effort:** S.

### ISSUE 6 — `input_is_stale` is hardcoded to 0; the staleness signal never fires
- **Symptom:** All 15 `prediction_snapshots` have `input_is_stale = 0`, even though they were made at `2026-09-01T07:03Z` from a bar dated `2026-08-31T00:00Z` (≈ 31 h old).
- **Root cause:** `forecaster.py:279` passes `input_is_stale=0` as a literal. It is never computed. Spec §? (`design.md:157`) defines it as *"features built from data > 1h old at prediction time."*
- **Classification:** REAL BUG.
- **Severity:** MED — the "serve last-good + staleness badge / defer retrain" degradation story (spec §6) depends on this flag, and with daily bars the input is *routinely* 12–36 h old, so it should very often be 1.
- **Recommended fix:** Compute `input_is_stale = 1 if (now - made_from_ts) > threshold else 0`, where `threshold` is the expected bar interval for the ticker's asset class (not a flat 1 h — a daily-bar system needs ~1 day + grace, same reasoning as Issue 1). **Effort:** S.

### ISSUE 7 — Accuracy panel empty
- **Symptom:** Accuracy panel shows nothing; `accuracy_records` table has 0 rows; 0 snapshots evaluated.
- **Root cause (multi-factor, mostly benign):**
  - Worker restarted ~08:30Z; `job_evaluate_hourly` (1 h interval, first fire ≈ start + 1 h) has not run yet this session — no `system_heartbeat` row for it. Predominantly **environmental / too-soon**.
  - Of the 15 snapshots, only the 3 equity **1d** snapshots (target_ts `2026-09-01T00:00Z`) are even matured. When evaluate does run, `evaluator.py:74-90` looks for the nearest `OhlcvBar` with `ts >= target_ts` for that ticker — **none exists** (equity bars stop at `2026-08-31`; today's close won't print until ~13:30Z). So those will be skipped until the real bar lands. This is the same daily-feed reality as Issue 1, not a new bug.
  - All crypto snapshots and all 5d/30d snapshots have `target_ts` in the future — correctly not evaluated.
  - Minor real concern: `prediction_snapshots.eval_attempts` is still 0 for the matured-but-unresolvable equity 1d rows — confirm `evaluator.run` increments `eval_attempts` on the "no realized bar" skip path (`evaluator.py` around line 74-95, not fully traced) so a permanently-missing bar is eventually abandoned rather than retried forever, and so an operator can see "tried, no data".
- **Classification:** ENVIRONMENTAL (timing) + small REAL concern (eval_attempts accounting).
- **Severity:** LOW — expect records to appear naturally after the first post-open evaluate run tomorrow. Re-check after 13:30Z + 1 h.
- **Recommended fix:** No urgent code change. Optionally: shorten first-run delay for `job_evaluate_hourly` (run once at startup), and verify `eval_attempts` increments on skips.

### ISSUE 8 (latent) — `check_error_rate` "all providers down" can false-alarm off dead provider rows
- **Symptom:** none yet.
- **Root cause:** `health_checks.py:205` `if len(down_providers) == len(metrics)` → CRITICAL "All primary providers down". `link_metrics` accumulates rows for providers that are no longer used (e.g. `coingecko`). If a stale/dead provider row sits at `breaker_state="open"` (or ≥5 failures) and the one genuinely-active provider also trips, the ratio hits 100 % and fires a system-CRITICAL that overstates the outage. Conversely a dead provider frozen at "closed" masks nothing but pollutes the denominator.
- **Classification:** LATENT BUG.
- **Severity:** LOW now; MED once more providers are registered (Issue 2) and some legitimately go idle.
- **Recommended fix:** Scope `check_error_rate` / `check_latency` / `check_quota` to the *currently-active* provider set (providers actually polled in the last N minutes), or add an `active`/`last_polled_at` filter on `LinkMetrics`. **Effort:** S.

### ISSUE 9 (design gap) — daily bar granularity vs intraday poll/freshness/quota model
- **Observation:** `poll_interval_crypto_sec = 60`, `poll_interval_equity_min = 5`, freshness thresholds 5 m/20 m, quota budgeting "10 equities × every 5 min ≈ 2,880 calls/day" (`design.md:364`) — all assume an intraday/streaming feed. Every provider actually returns **daily** OHLCV. Polling a daily endpoint every 60 s re-fetches an unchanged bar ~1,400 times/day per crypto ticker, burning quota for nothing, and makes the freshness model (Issue 1) unsatisfiable.
- **Classification:** DESIGN GAP — decide + document, or re-scope.
- **Severity:** MED.
- **Recommended fix (pick one):** (a) Accept daily bars: drop poll cadence to ~hourly, rewrite freshness around "expected next daily bar", document that the app is end-of-day, not real-time. (b) Add a genuine intraday source (e.g. Finnhub/Tiingo intraday, Coinbase ticker) for the "live price" number and keep daily bars for the model. (a) is far less work and matches what's built.

---

## RANKED FIX BACKLOG

### Do first (before any "v1.0.1")
1. **Issue 2 — register the real provider set in the worker** (+ derivatives provider + `job_ingest_derivatives`). Without this, crypto ingest and all failover are dead. Effort M. Unblocks live crypto data and Issue 5.
2. **Issue 1 — market-hours / bar-interval-aware freshness check.** This is what makes the whole app self-report CRITICAL. Effort M. Needs `pandas-market-calendars` (already spec'd).
3. **Issue 4 — remove the double `sqrt(h)` in CI bands** (`forecaster.py:238`, `trainer.py:256`) + regenerate model artifacts + add a `wf_ci_cov` sanity test. Effort S–M. Fixes forecast correctness and the 30d chart blow-up.
4. **Issue 3 — stop `job_ingest_*` reporting heartbeat success on total failure.** Effort S–M. Restores observability so 1 and 2 can't silently regress.

### Do next
5. **Issue 6 — compute `input_is_stale` for real.** Effort S. Pairs naturally with Issue 1 (shared "expected bar interval" helper).
6. **Issue 5 — fix the stuck "RECOVERING" chip** (stale-row filter + tighter RECOVERING predicate + make coinbase the crypto primary). Effort S. Mostly falls out of Issue 2.
7. **Issue 8 — scope provider health checks to the active provider set.** Effort S. Prevents false CRITICALs once Issue 2 adds more provider rows.

### Decision needed (not code-first)
8. **Issue 9 — daily-bar vs real-time.** Product call: embrace end-of-day (cheap) or add an intraday price source (expensive). Recommend documenting as end-of-day and dropping poll cadence to hourly. Feeds the exact thresholds used in Issues 1 and 6.

### Wontfix / document only
9. **Issue 7 — empty accuracy panel.** No code change; it is a fresh-worker timing artifact. Records should appear after the first evaluate run following the 2026-09-02 equity close. Only action: confirm `eval_attempts` increments on the "no realized bar" skip path so unresolvable maturities aren't retried forever (fold into Issue 3's observability pass).
10. **CoinGecko 401 without a Demo key** — already in `KNOWN_LIMITATIONS.md §2`; the fix is "use Coinbase as crypto primary" (Issue 2), not "make CoinGecko work".

---

## Notes for whoever picks up the fixes
- All findings are from static reading + live DB inspection. I did **not** run a second worker/streamlit instance (god has instances on :8501; a second worker writing the shared `data/app.db` risked confusing the very state being analysed).
- The 731 `coinbase`-sourced crypto bars prove the Coinbase provider *works* — it just isn't wired into the worker.
- Issues 1, 6, and 9 all revolve around one missing concept: **"expected next bar time per ticker."** Introduce it once and three issues get cleaner fixes.
- No destructive behaviour or security issues found.
