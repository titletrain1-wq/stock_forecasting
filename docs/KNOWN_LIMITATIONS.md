# Known Limitations & Operational Constraints — stock_forecasting v1.0.0

This document outlines the known operational boundaries, provider caveats, and design constraints of `stock_forecasting` v1.0.0.

## 1. Single-User / Local Architectural Scope

- **Deployment**: Designed specifically as a single-user, local application. There is no multi-tenant authentication, RBAC, or remote cluster synchronization.
- **Database Engine**: Uses SQLite (`./data/app.db`) in WAL mode (`busy_timeout=5000`). Designed for concurrent local worker and UI access, not high-concurrency remote writer pools.

## 2. Provider API Constraints & Quirks

- **Finnhub Stock Candles**: Finnhub's `/stock/candle` endpoint requires a premium plan for US equities. Free API keys return HTTP 403 or empty responses. **Tiingo** is configured as the functional primary fallback for Yahoo Finance equity data.
- **dYdX Derivatives History**: The dYdX v4 Indexer API does not support historical open interest endpoints (open interest is available as a current market snapshot only). Funding rate history is available hourly dating back to dYdX v4 mainnet launch (~Nov 2023).
- **Public API Quotas**: Free-tier API keys (CoinGecko, Coinbase, Tiingo, Finnhub) have rate limits. `LinkMonitor` and `CircuitBreaker` automatically guard quota limits and shift requests to secondary providers when thresholds are exceeded.

## 3. Visualization & UI Deferred Items

- **Indicator Overlays on Price Pane**: Technical indicators (SMA20/50, Bollinger Bands, RSI, MACD) are computed in `FeatureBuilder` and evaluated by ML models, but are deferred from being rendered directly on the main price chart (per Ruling 6). The chart explicitly renders price, predictions, confidence interval bands, and evaluated forecast markers.
- **Data Quality Percentage**: The data quality metric in the health panel applies a linear heuristic ($100\% - 5\% \times n_{\text{quarantined\_24h}}$).

## 4. Model & Data Requirements

- **Warmup Window**: Technical indicator calculation requires a minimum of 50 historical daily bars. Tickers with fewer than 50 bars will be skipped during feature extraction.
- **Trustworthiness Criteria**: Aggregate forecast accuracy records require a minimum sample size $n \ge 30$ and directional accuracy $\ge 55\%$ before marking `is_trustworthy = 1`.
