# stock_forecasting (v1.0.1)

Personal single-user **end-of-day** stock and cryptocurrency forecast application with an immutable prediction ledger, walk-forward ML models, automated circuit-breaker failover, Plotly interactive visualization, and full system health monitoring.

> **End-of-day, not real-time.** Every data provider serves daily OHLCV bars.
> The worker polls hourly (enough to pick up a new daily bar shortly after it
> appears) and freshness is judged against the trading calendar, not wall-clock
> minutes. This is not an intraday/streaming trading tool.

## Features

- **Multi-Asset Support**: Ingestion for US Equities (Yahoo Finance primary, Tiingo/Finnhub fallback when keyed) and Cryptocurrencies (Coinbase primary — keyless; CoinGecko fallback when keyed; dYdX for derivatives).
- **Resilient Ingestion**: Automatic provider failover, circuit breaker state machine, boundary validation, and invalid data quarantining.
- **Zero-Lookahead Feature Engineering**: 17 technical indicators + 4 crypto-only features (funding rate, open interest, weekend volume ratio).
- **Walk-Forward ML Pipeline**: Ridge Regression & Random Forest models trained directly on the h-day cumulative log return; the 95% confidence band uses the walk-forward residual standard deviation (already a h-horizon quantity — no extra time scaling).
- **Immutable Ledger**: SQLite trigger-backed prediction snapshot ledger with automated evaluation against realized prices.
- **Interactive UI**: Streamlit interface powered by Plotly charts (candles, price overlays, forecast ribbons, accuracy records, explainability maps, system health cards).

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd ~/Desktop/stock_forecasting

# Install dependencies using uv
uv sync

# Configure environment variables (all optional):
#   TIINGO_API_KEY / FINNHUB_API_KEY  -> extra equity fallback providers
#   COINGECKO_API_KEY (Demo key)      -> extra crypto fallback provider
# With no keys the app still runs: yfinance (equities) + Coinbase (crypto), both keyless.
cp .env.example .env
```

### After upgrading

The v1.0.1 confidence-band fix changes model metrics, so the model artifacts
under `model_store/` must be regenerated once. The background worker's nightly
retrain job (`job_retrain_nightly`) does this automatically the first time it
runs; to force it immediately, start the worker and trigger that job, or run an
equivalent retrain over every active `ticker × horizon × model_type`.

### 2. Running the Application

**Terminal 1 — Background Worker Scheduler:**
```bash
uv run python -m stock_forecasting.worker
```

**Terminal 2 — Streamlit Dashboard:**
```bash
uv run streamlit run stock_forecasting/app.py
```

### 3. Running Tests & Quality Checks

```bash
# Run full test suite (170+ unit & chaos tests)
uv run pytest tests/ -v

# Run linter & code formatting check
uv run ruff check .
uv run ruff format --check .
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): System architecture, design decisions, and data flow.
- [`docs/API.md`](docs/API.md): API interface specification for core service components.
- [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md): Operational constraints and documented system boundaries.
