# Known Limitations & Operational Constraints — stock_forecasting v2.0.0

This document outlines the known operational boundaries, provider caveats, and design constraints of `stock_forecasting` v2.0.0.

## 0. Real-Time Display vs. Daily Training Models (v2 Hybrid Architecture)

- **Dual-Path Architecture**: `stock_forecasting` v2.0.0 decouples the real-time visualization layer from the daily ML forecasting pipeline. Crypto quotes stream in real time via Coinbase WebSocket, and equities poll at 5-minute intervals (~15-minute delayed). Daily OHLCV bars remain the sole source of truth for ML training, feature extraction, and forecast evaluation.
- **Delayed Equities**: Free-tier equity intraday bars from yfinance are ~15 minutes delayed. Intraday polling occurs on a 5-minute schedule with a 20-minute provisional lookback window. Real-time paid equity feeds (e.g. Polygon / IEX) are outside v2.0.0 scope.
- **Provisional Forming Candles**: The currently forming intraday candle (`1m` for crypto, `5m` for equity) is tracked in `intraday_bars` with `is_provisional = 1` and continuously updated on new ticks. Once the bucket interval elapses, the candle is finalized (`is_provisional = 0`).
- **Display-Only Live Feed**: Intraday ticks and quotes are strictly display-only. Live price movements never trigger model retraining, feature recomputation, or dynamic re-anchoring of the forecast ribbon / CI band.
- **Calibration to Daily Close (`P_close`)**: Statistical confidence intervals ($\pm 1.96 \sigma_h$) are calibrated against completed daily session closes. The CI band is anchored to `P_close`, never `P_live`.
- **Crypto No-True-Close**: Crypto markets trade continuously 24/7 without a hard session close. Daily bars use provider 00:00 UTC cutoff boundaries. A minor visual step may appear at the junction between the intraday live trace and the daily-anchored forecast ribbon if the cutoff close differs from the live tick.

## 1. Single-User & Process Architecture

- **Two-Process Model**: `worker.py` (ingestion, WebSocket feed, scheduler, retrain) and `streamlit run app.py` (visualization UI) run as two independent OS processes communicating strictly through SQLite WAL. If only Streamlit is started without the worker, live ticks will not stream. A single-command supervisor process is a possible future enhancement.
- **Deployment**: Designed specifically as a single-user, local application. There is no multi-tenant authentication, RBAC, or remote cluster synchronization.
- **Database Engine**: Uses SQLite (`./data/app.db`) in WAL mode (`busy_timeout=5000`). Concurrent worker writes and 2-second UI fragment reads operate lock-free under WAL concurrency.
- **Intraday ingest seam (M2 design deviation)**: the design's proposed `ingestion.py` thin wrappers (`upsert_live_quote`, `ingest_intraday_bar`) were not built. `IntradayRepository` / `LiveQuoteRepository` are the intraday ingest seam directly, called from the worker's `_on_tick` and `job_ingest_equity_intraday`. `ingestion.py` still owns provider failover for the **daily** path only. See `docs/ARCHITECTURE.md` and `CHANGELOG.md` v2.0.0.

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

## 5. Live Price vs. Calibrated Forecast Band (v2 streaming chart)

The streaming chart moves a live intraday price line across a **static** forecast ribbon and CI band. The band is always anchored to the last completed daily close (`P_close`) and is never re-based on the live tick. Verbatim (per design §5.2):

> "Statistical confidence intervals (±1.96 σ_h) and horizon accuracy evaluations are strictly calibrated to forecasts anchored at completed daily market closes (P_close). Plotted CI bands anchored to live intraday prices (P_live) represent informal visual projections; using P_live as a dynamic band origin invalidates the calibrated 95% walk-forward coverage guarantee."

This text is surfaced on every chart figure and in the app chart caption. A regression test (`tests/test_ml_overlay_integrity.py`) fences it: mutating every `live_quotes.price` leaves `lower_bound` / `upper_bound` and every ribbon point byte-identical — only the `live` trace moves.
