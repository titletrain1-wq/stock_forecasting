# Changelog

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
