# Known Limitations & Operational Constraints — stock_forecasting v1.0.1

This document outlines the known operational boundaries, provider caveats, and design constraints of `stock_forecasting` v1.0.1.

## 0. End-of-Day, Not Real-Time

- **Bar granularity**: Every provider (yfinance, Coinbase, Tiingo, Finnhub, CoinGecko) serves **daily** OHLCV bars. There is no intraday or streaming feed.
- **Poll cadence**: The worker polls hourly by default (`poll_interval_crypto_sec=3600`, `poll_interval_equity_min=60`). Hourly is frequent enough to pick up a new daily bar shortly after it appears; polling faster only re-fetches an unchanged bar and burns the free-tier request budget.
- **Freshness model**: `check_freshness` judges each ticker's latest bar against its expected **trading-calendar** schedule — NYSE sessions (`pandas-market-calendars`) for equities, one bar per UTC day for crypto — not against a wall-clock age. A bar is only CRITICAL when genuinely overdue (equity: ≥2 missed sessions; crypto: ≥3 days behind). An equity feed showing yesterday's close before today's open is NOMINAL.
- **Forming crypto candle**: Coinbase returns a *forming* (partial) candle for the current UTC day, so the latest crypto bar in the DB may be an incomplete day until it settles at 00:00Z. Features and the forecast anchor can therefore be built from a partial-day bar. (Tracked as a follow-up; see release notes.)

## 1. Single-User / Local Architectural Scope

- **Deployment**: Designed specifically as a single-user, local application. There is no multi-tenant authentication, RBAC, or remote cluster synchronization.
- **Database Engine**: Uses SQLite (`./data/app.db`) in WAL mode (`busy_timeout=5000`). Designed for concurrent local worker and UI access, not high-concurrency remote writer pools.

## 2. Provider API Constraints & Quirks

- **Provider registration**: The worker always registers the two keyless providers — **yfinance** (equity primary) and **Coinbase** (crypto primary). Tiingo, Finnhub and CoinGecko register only when their API key is configured; without a key they are simply absent from the failover chain (not silently broken). The startup log lists exactly which providers registered.
- **Finnhub Stock Candles**: Finnhub's `/stock/candle` endpoint requires a premium plan for US equities. Free/keyless requests return HTTP 401/403. Finnhub is therefore only a *nominal* equity fallback — useful only with a paid key.
- **CoinGecko Free-Tier Authentication**: CoinGecko's `/coins/{id}/market_chart/range` endpoint returns HTTP 401 without a Demo API Key, and the Demo tier has a strict ~10k-call monthly cap. It is **not** the crypto primary — Coinbase (keyless) is. Set `COINGECKO_API_KEY` in `.env` only if you want CoinGecko as an extra fallback.
- **dYdX Derivatives History**: The dYdX v4 Indexer API does not support historical open interest endpoints (open interest is available as a current market snapshot only). Funding rate history is available hourly dating back to dYdX v4 mainnet launch (~Nov 2023).
- **Public API Quotas**: Free-tier API keys (CoinGecko, Coinbase, Tiingo, Finnhub) have rate limits. `LinkMonitor` and `CircuitBreaker` automatically guard quota limits and shift requests to secondary providers when thresholds are exceeded.


## 3. Visualization & UI Deferred Items

- **Indicator Overlays on Price Pane**: Technical indicators (SMA20/50, Bollinger Bands, RSI, MACD) are computed in `FeatureBuilder` and evaluated by ML models, but are deferred from being rendered directly on the main price chart (per Ruling 6). The chart explicitly renders price, predictions, confidence interval bands, and evaluated forecast markers.
- **Data Quality Percentage**: The data quality metric in the health panel applies a linear heuristic ($100\% - 5\% \times n_{\text{quarantined\_24h}}$).

## 4. Model & Data Requirements

- **Warmup Window**: Technical indicator calculation requires a minimum of 50 historical daily bars. Tickers with fewer than 50 bars will be skipped during feature extraction.
- **Trustworthiness Criteria**: Aggregate forecast accuracy records require a minimum sample size $n \ge 30$ and directional accuracy $\ge 55\%$ before marking `is_trustworthy = 1`.
