# Architecture & Design Overview — stock_forecasting

## System Architecture

`stock_forecasting` is a single-user daily stock and cryptocurrency forecasting application built on Python 3.12, SQLModel/SQLite, and Streamlit.

```
                  ┌───────────────────────────────┐
                  │       Data Providers          │
                  │  (yfinance, CoinGecko,        │
                  │   Coinbase, Tiingo, dYdX)     │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │       Ingestion Service       │
                  │   + CircuitBreaker Failover   │
                  │   + Boundary Validation       │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │      BarRepository & DB       │
                  │    (SQLite WAL + Pragmas)     │
                  └──────────────┬────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
  ┌─────────────────────────────┐  ┌─────────────────────────────┐
  │      Worker Scheduler       │  │       Streamlit App         │
  │ (Nightly Retrain, Ingest,   │  │   (Watchlist, Plotly Chart, │
  │  Hourly Eval, Heartbeat)    │  │   Accuracy, Explain, Health)│
  └──────────────┬──────────────┘  └─────────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │     Model Store (.joblib)   │
  │    + FeatureBuilder         │
  │    + Forecast & Evaluator   │
  └─────────────────────────────┘
```

## Key Architectural Principles

1. **Service-Layer Separation**: Core logic (ingestion, feature calculation, model training, forecasting, evaluation, health checks) lives in pure Python service modules independent of Streamlit or CLI frameworks.
2. **Immutable Prediction Ledger**: Once created, prediction records in `prediction_snapshots` are immutable (enforced via a SQLite `enforce_snapshot_immutability` trigger). Only realization/evaluation columns may be updated as target dates mature.
3. **Zero-Lookahead Feature Engineering**: Technical indicators computed by `FeatureBuilder` strictly rely on past data ($< t$). Rolling calculations shift baselines to ensure target leakage cannot occur.
4. **Resilient Data Ingestion**: `IngestionService` instruments all API calls through `LinkMonitor` and guards requests with `CircuitBreaker`. On provider failure (HTTP 429/500), requests automatically fail over to secondary providers.
5. **Walk-Forward Validation**: Models are trained using `TimeSeriesSplit(n_splits=5)` expanding-window walk-forward validation. Each model is trained directly on the h-day cumulative log return, so the walk-forward residual standard deviation $\sigma_{\text{residual}}$ is already a h-horizon quantity; the 95% band is $\text{predicted\_price} \times e^{\pm 1.96\,\sigma_{\text{residual}}}$ with no additional $\sqrt{h}$ scaling (applying it twice was fixed in v1.0.1).
6. **Integrated System Health Monitoring**: `HealthChecker` continuously assesses 8 health dimensions (freshness, latency, error rates, data gaps, quarantine count, scheduler heartbeat, clock skew, and daily API quotas).

## v2.0.0 — Real-Time Display Layer

v2 adds a live display path that is **fully decoupled** from the daily ML pipeline. The ML core (`forecaster.py`, `trainer.py`, `features.py`, `evaluator.py`, `accuracy.py`) is unchanged; live prices never feed training, features, or the forecast ledger.

- **Storage**: two new tables — `intraday_bars` (short-retention sub-daily OHLCV buckets, worker-pruned after `intraday_retention_days`) and `live_quotes` (one row per ticker, the current-price anchor). `ohlcv_bars` remains the sole ML source of truth.
- **Crypto feed — `live_feed.CoinbaseWSClient`**: the worker owns a single Coinbase Advanced-Trade WebSocket connection (`ticker_batch` + `heartbeats`, keyless) on a daemon thread. Each tick runs a short transaction: upsert `live_quotes` + extend the forming `intraday_bars` bucket. Auto-reconnect with capped backoff. `coinbase_rest_candles()` is the REST fallback.
- **Equity feed**: `providers/yfinance.get_intraday_bars()` polls 5-minute bars (~15-minute delayed) on a 5-minute worker job (`job_ingest_equity_intraday`).
- **WS-idle fallback**: `job_check_ws_idle` — if no tick/heartbeat for `ws_idle_timeout_sec`, the worker pulls REST candles and marks the crypto feed `DEGRADED` until the socket recovers.
- **Streaming chart**: `app.py` wraps the price header + chart in `st.fragment(run_every=_refresh_for(asset_class))` (2 s crypto / 15 s equity), stable `key="live_price_chart"`, `uirevision=True`. `viz.add_live_price_line()` overlays a `live` scatter + a faded provisional `forming` candle. The forecast ribbon + CI band stay anchored to `P_close` — see `viz.CI_DISCLAIMER` and `tests/test_ml_overlay_integrity.py`.
- **Two-path health**: `HealthChecker.compute_system_status(display_only_checks=...)` — display-path checks (`live_feed_crypto`, `live_feed_equity`, `ws_connection`, `intraday_prune`) can contribute at most `DEGRADED`, so a live-feed outage never marks the training/prediction core `CRITICAL`. The training-data path keeps the v1.0.1 trading-calendar freshness model.
- **Two-process model**: `worker.py` and `streamlit run app.py` are independent OS processes communicating only through SQLite WAL. Without the worker running, no live ticks stream.

### M2 deviation from the design (recorded)

The design (`docs/2026-09-01-realtime-v2-design.md` §6.2) sketched thin `ingestion.py` wrappers (`upsert_live_quote`, `ingest_intraday_bar`) as the intraday ingest seam. **This was skipped**: `IntradayRepository` / `LiveQuoteRepository` in `intraday_store.py` *are* the seam directly, called from the worker's `_on_tick` and `job_ingest_equity_intraday`. v1's `ingestion.py` orchestrates provider failover for the daily path; the intraday path does WS-idle→REST in the worker instead, so a second wrapper layer added indirection with no failover value.
