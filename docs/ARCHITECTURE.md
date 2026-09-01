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
