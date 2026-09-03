# Changelog

## Unreleased

Chart technical indicators for on-demand analysis on the live price pane and
dedicated sub-panes (T-011). Three critical bug fixes (T-012): accuracy
evaluator handling of pending forecasts, health checks filtering stale provider
metrics, and daily model protection against live forming-candle leakage. Fixed a
pre-existing cross-platform charset encoding bug that broke the v2.0.0 test
suite on Windows. Suite: 236 tests green.

### Added

- **Chart technical indicators (T-011).** Five new indicators on the price pane
  (`SMA20`, `SMA50`, Bollinger Bands with 2σ envelope) and three sub-panes
  (`RSI(14)`, `MACD(12,26,9)`, `Volume`). Built with `make_subplots`: price
  pane 60% height, sub-panes ~13% each, all share the x-axis. Each indicator is
  independently toggleable (default on). Integrates with existing daily/live
  ribbon layers. `viz.add_technical_indicators()` entrypoint. 16 new tests added
  to the suite.

### Fixed

- **Cross-platform charset encoding (T-011).** `viz.py` defined
  `CI_DISCLAIMER` as a string literal with non-ASCII characters (©, ±), stored
  with implicit Latin-1 encoding, but `KNOWN_LIMITATIONS.md` and other docs are
  UTF-8. The v2.0.0 test suite passed on Unix (UTF-8 filesystem default) but
  failed on Windows (Latin-1 default) with encoding mismatches during chart
  render. Re-encoded the disclaimer and all string constants to UTF-8 explicitly.
  Closes the T-010 Windows verification test failure.

- **Stuck accuracy panel (Issue 7, T-012).** The accuracy evaluator job
  previously did nothing when a forecast had matured but the realized bar was
  not yet available (e.g. due to ingestion lag or weekend gaps). The accuracy
  panel appeared frozen with stale data. Now the evaluator increments an
  `eval_attempts` counter on each check; the accuracy panel displays "N/A" or
  "evaluating…" while pending, and shows results once the bar arrives. No user
  sees a phantom "graded" state that's actually waiting.

- **Provider metrics masking outages (Issue 8, T-012).** Health checks aggregated
  error rates, latencies, and quota metrics across all registered providers,
  including retired ones (e.g. a CoinGecko row no longer in the active provider
  list). A stale metric row from a disconnected provider could shade the
  aggregated status `NOMINAL` even when all active providers were down. Health
  checks now filter metrics to the currently-active provider set only. Added a
  fallback-hardening safety net: if the active filter matches nothing but
  metrics exist (a potential mapping regression), all metrics are used rather than
  reporting `NOMINAL` — no outage can hide behind a filtering bug.

- **Forming candle leaked into daily training (Issue c, T-012).** Coinbase WS
  ticks include the current open (forming) candle while it is still in-session.
  The worker propagated this to both `intraday_bars` (display cache, correct) and
  the daily `ohlcv_bars` ML store (incorrect — the daily model expects immutable
  closed bars only). The forming candle then inflated daily-bar `volume` and
  `close` during the session, introducing live-price leakage into training
  labels. Now the worker filters: only completed/closed bars flow to `ohlcv_bars`.
  `intraday_bars` retains the forming candle for display continuity. Trade-off:
  the freshest daily EOD close may lag ~20 hours for US equities (UTC-scoped
  close gate, safe/conservative direction).

## v2.0.0 — 2026-09-02

Real-time-capable display layer. The daily ML pipeline
(`forecaster.py`, `trainer.py`, `features.py`, `evaluator.py`, `accuracy.py`)
is **unchanged** — live prices never feed training, features, or the forecast
ledger. Design: `docs/2026-09-01-realtime-v2-design.md`; plan + milestone gates:
`docs/plans/2026-09-01-realtime-v2.md`. Branch built locally, tagged `v2.0.0`.

### Added

- **Intraday storage (M0).** New tables `intraday_bars` (short-retention
  sub-daily OHLCV buckets, `is_provisional` flag, worker-pruned) and
  `live_quotes` (one row per ticker, the current-price anchor). `ohlcv_bars`
  and the ML ledger are untouched. 8 new `Settings` fields + `.env.example`.
- **Coinbase WebSocket feed (M1–M2).** `live_feed.CoinbaseWSClient` — the
  worker owns one keyless Advanced-Trade WS connection (`ticker_batch` +
  `heartbeats`) on a daemon thread; each tick is a short transaction writing
  `live_quotes` + the forming `intraday_bars` bucket. Auto-reconnect with
  capped backoff; `coinbase_rest_candles()` REST fallback. A WAL per-tick-write
  contention spike (`docs/spikes/2026-09-01-M2-wal-contention.md`) confirmed the
  per-tick DB write model is safe (0 lock errors, write p95 ≈ 5 ms).
- **Equity intraday poller (M3).** `providers/yfinance.get_intraday_bars()` +
  `job_ingest_equity_intraday` — 5-minute bars, ~15-minute delayed.
- **Two-path health (M4).** `HealthChecker.compute_system_status(display_only_checks=…)`
  — display-path checks (`live_feed_crypto`, `live_feed_equity`, `ws_connection`,
  `intraday_prune`) contribute at most `DEGRADED`, so a live-feed outage never
  marks the training/prediction core `CRITICAL`. The training-data path keeps
  the v1.0.1 trading-calendar freshness model. `job_check_ws_idle` falls back to
  REST when the socket goes quiet.
- **Streaming chart (M5).** `app.py` wraps the price header + chart in
  `st.fragment(run_every=_refresh_for(asset_class))` (2 s crypto / 15 s equity),
  stable `key="live_price_chart"`, `uirevision=True`.
  `viz.add_live_price_line()` overlays a `live` line + a faded provisional
  `forming` candle. Equity charts show a `🟡 15-min delayed` badge.
- **ML overlay integrity fence (M6).** `viz.CI_DISCLAIMER` (verbatim calibration
  disclaimer) on every figure + the app caption + `KNOWN_LIMITATIONS.md` §5.
  `tests/test_ml_overlay_integrity.py`: mutating every `live_quotes.price`
  leaves `lower_bound` / `upper_bound` and every ribbon point byte-identical —
  only the `live` trace moves. The band is always anchored to `P_close`.
- **Chaos suite + docs (M7).** `tests/test_chaos.py` covers WS drop
  mid-session (reconnect), WS silent / heartbeat stop (REST fallback + feed
  `DEGRADED`), Mac sleep/wake, equity 429 storm, and the close-time reconcile
  step.

### Deviations from the design (recorded)

- **M2 — `ingestion.py` thin wrappers skipped.** The design (§6.2) sketched
  `upsert_live_quote` / `ingest_intraday_bar` wrappers as the intraday ingest
  seam. `IntradayRepository` / `LiveQuoteRepository` in `intraday_store.py` are
  the seam directly, called from the worker's `_on_tick` and
  `job_ingest_equity_intraday`. v1's `ingestion.py` orchestrates provider
  failover for the daily path; the intraday path does WS-idle→REST in the
  worker, so a second wrapper layer added indirection with no failover value.
  See `docs/ARCHITECTURE.md`.

### Known limitations

- **Crypto has no true 00:00 UTC close.** The "daily close" is the provider's
  cutoff bar, so the intraday-line → forecast-ribbon handover can show a small
  visual step when the cutoff close differs from the last live tick. Equities
  (real 16:00 ET close) do not. See `KNOWN_LIMITATIONS.md` §0 / §5.
- **Two processes.** `worker.py` and `streamlit run app.py` are separate OS
  processes; without the worker, no live ticks stream.
- Real-time paid equity feeds, intraday forecasting models, and trade signals
  are out of scope for v2.0.0.

### Upgrade note

Adds two tables (`intraday_bars`, `live_quotes`) — run the worker once to create
them. The Coinbase WebSocket is keyless; no new API key is required. New
`Settings` fields have safe defaults (see `.env.example`). Model artifacts under
`model_store/` are unchanged and remain gitignored.

## v1.0.1 — 2026-09-01

Post-release reliability fixes from the live-run health analysis
(`docs/reports/2026-09-01-postrelease-health-analysis.md`). No new features.

### Fixed

- **Confidence-interval bands double-counted time (Issue 4).** Models are
  trained on the h-day cumulative log return, so `residual_std` is already a
  h-horizon quantity; both the walk-forward `wf_ci_cov` back-test and the
  published bands then multiplied it by `sqrt(h)` again. 30d bands were ~5.5×
  too wide (e.g. AAPL 30d band was 64–1586 around a 317 price) and `wf_ci_cov`
  pinned at 1.0 for every 5d/30d model, self-masking the bug. Removed the extra
  `* np.sqrt(h)` in `forecaster.py` and `trainer.py`; retrained models now
  report `wf_ci_cov` ≈ 0.87–0.96. A regression test locks the calibration.

- **Worker registered only yfinance + fake (Issue 2).** Crypto ingest was 100%
  dead ("No registered provider for BTC-USD"), equities had no failover, and
  crypto derivatives never ingested. The worker now registers yfinance +
  Coinbase (keyless) always, and Tiingo / Finnhub / CoinGecko when their API key
  is set; it wires the dYdX derivatives provider and schedules a
  `job_ingest_derivatives` job; it logs the registered provider set at startup;
  and it moves any crypto ticker off an unregistered primary (e.g. CoinGecko
  with no key) onto keyless Coinbase.

- **`check_freshness` used wall-clock minutes on daily bars (Issue 1).** Flat
  5-minute (crypto) / 20-minute (equity) thresholds meant the system reported
  CRITICAL by construction — an equity feed showing yesterday's close pre-open,
  or a once-a-day crypto bar, is always more than an hour old. New
  `market_calendar.py` judges freshness against the trading calendar (NYSE
  sessions via `pandas-market-calendars` for equities, one UTC day per bar for
  crypto). CRITICAL only when genuinely overdue.

- **Ingest jobs reported heartbeat success on total failure (Issue 3).**
  `poll_ticker` returns an error dict instead of raising, so a complete ingest
  outage was recorded as a healthy pulse. The ingest jobs now inspect the poll
  results and record `success=False` with aggregated per-ticker errors when
  every ticker failed; an empty provider chain also records a circuit-breaker
  failure so `check_error_rate` sees it.

- **`input_is_stale` was hardcoded to 0 (Issue 6).** It is now computed from the
  anchor bar's calendar freshness at prediction time.

- **Provider chip stuck on "RECOVERING" forever (Issue 5).** `health_view`
  showed RECOVERING for any non-zero `consecutive_failures`, so a closed-breaker
  provider that failed once at seed time and was never re-polled stayed amber
  indefinitely. RECOVERING is now shown only while the breaker is `half_open`.

### Changed

- **End-of-day operating model (Issue 9, option A).** Default poll cadence
  dropped to hourly (`poll_interval_crypto_sec=3600`,
  `poll_interval_equity_min=60`). README and `KNOWN_LIMITATIONS.md` now state
  the app is end-of-day, not real-time.

### Upgrade note

Model metrics change with the CI-band fix. Regenerate artifacts once by letting
`job_retrain_nightly` run (or retrain manually). Model artifacts live under
`model_store/` and are gitignored, as in v1.0.0.

### Known follow-ups (not in this release)

- Coinbase returns a *forming* partial candle for the current UTC day, which is
  ingested as a normal bar; features / the forecast anchor can be built from a
  partial-day bar. Spec wants closed bars only.
- **Issue 7** (analysis report): the accuracy panel stays empty until predictions
  mature and the realized bar exists — a fresh-worker timing artifact, no code
  change, but confirm `eval_attempts` increments on the "no realized bar" skip.
- **Issue 8** (analysis report): `check_error_rate` / `check_latency` /
  `check_quota` still count `LinkMetrics` rows for providers that are no longer
  polled (e.g. a dead CoinGecko row), which can skew the "all providers down"
  ratio. Scope those checks to the active provider set.
- Issue 5 side effect: a provider now reads 🟢 (STANDBY/ACTIVE) while its breaker
  is still `closed` even if it just failed a call — the amber "RECOVERING"
  signal is only shown once the breaker actually opens.
