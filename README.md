# stock_forecasting (v1.0.0)

Personal single-user daily stock and cryptocurrency forecast application with an immutable prediction ledger, walk-forward ML models, automated circuit-breaker failover, Plotly interactive visualization, and full system health monitoring.

## Features

- **Multi-Asset Support**: Ingestion for US Equities (Yahoo Finance, Tiingo, Finnhub) and Cryptocurrencies (CoinGecko, Coinbase, dYdX).
- **Resilient Ingestion**: Automatic provider failover, circuit breaker state machine, boundary validation, and invalid data quarantining.
- **Zero-Lookahead Feature Engineering**: 17 technical indicators + 4 crypto-only features (funding rate, open interest, weekend volume ratio).
- **Walk-Forward ML Pipeline**: Ridge Regression & Random Forest models with horizon uncertainty scaling ($\sigma \times \sqrt{h}$).
- **Immutable Ledger**: SQLite trigger-backed prediction snapshot ledger with automated evaluation against realized prices.
- **Interactive UI**: Streamlit interface powered by Plotly charts (candles, price overlays, forecast ribbons, accuracy records, explainability maps, system health cards).

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd ~/Desktop/stock_forecasting

# Install dependencies using uv
uv sync

# Configure environment variables (optional API keys for Tiingo / Finnhub)
cp .env.example .env
```

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
# Run full test suite (149+ unit & chaos tests)
uv run pytest tests/ -v

# Run linter & code formatting check
uv run ruff check .
uv run ruff format --check .
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): System architecture, design decisions, and data flow.
- [`docs/API.md`](docs/API.md): API interface specification for core service components.
- [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md): Operational constraints and documented system boundaries.
