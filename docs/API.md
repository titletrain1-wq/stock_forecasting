# API Reference — stock_forecasting

## Core Components & Service Interfaces

### 1. Data Ingestion & Storage

- **`BarRepository(session: Session)`**:
  - `upsert_bars(ticker: str, bars: list[Bar], source: str) -> int`: Validates OHLCV bars (`close > 0`, `low <= high`, `volume >= 0`). Inserts invalid bars into `QuarantineBar` and upserts valid bars into `OhlcvBar`.
  - `get_range(ticker: str, start_ts: str, end_ts: str) -> list[OhlcvBar]`: Retrieves bars ordered by timestamp.
  - `get_latest(ticker: str, limit: int = 1) -> list[OhlcvBar]`: Fetches most recent N bars.

- **`IngestionService(session: Session, providers: dict, circuit_breaker: CircuitBreaker | None = None)`**:
  - `poll_watchlist() -> dict`: Polls all active watchlist tickers.
  - `poll_ticker(symbol: str) -> dict`: Fetches latest bars for symbol using primary provider with CircuitBreaker failover to secondary sources.
  - `backfill(symbol: str, years: int = 5) -> dict`: Historical backfill with failover.

### 2. Feature Engineering & Training

- **`FeatureBuilder(feature_cols: Sequence[str] | None = None)`**:
  - `build(bars_df: pd.DataFrame, train_window: tuple | None = None, scale: bool = True, asset_class: str = "equity", derivatives_df: pd.DataFrame | None = None) -> pd.DataFrame`: Calculates 17 technical indicators + 4 crypto-only features without lookahead bias.

- **`Trainer(session: Session, model_dir: str = "./model_store")`**:
  - `train(ticker: str, horizon: str, model_type: str) -> ModelArtifact`: Performs walk-forward TimeSeriesSplit validation, fits Ridge or RandomForest models, persists `.joblib` artifacts, and logs `ModelRun`.

### 3. Forecasting & Evaluation

- **`ForecastService(session: Session, model_dir: str = "./model_store")`**:
  - `generate_and_persist(ticker: str, horizons: list[str] = ["1d", "5d", "30d"], model_types: list[str] = ["ridge"]) -> dict`: Generates multi-horizon forecasts, computes scaled CI bounds, extracts feature explainability maps, and commits immutable `PredictionSnapshot` rows.

- **`EvaluatorService()`**:
  - `run(session: Session, as_of: str | None = None)`: Grades matured snapshots against realized prices and updates `evaluated_at`, `realized_return`, `is_direction_hit`, and `is_within_ci`.

- **`AccuracyService()`**:
  - `rebuild_aggregates(session: Session)`: Rebuilds per-(ticker, horizon, model_type) and global `AccuracyRecord` summaries and sets `is_trustworthy` (`dir_acc >= 0.55 AND n >= 30`).

### 4. System Health & Resiliency

- **`LinkMonitor(session: Session)`**:
  - `instrument(provider: str)`: Context manager timing call RTT, tracking error rate, and quota usage in `LinkMetrics`.

- **`CircuitBreaker(session: Session, failure_threshold: int = 5, cooldown_minutes: int = 15)`**:
  - `guard(provider: str)`: Context manager fail-fast check (`closed` -> `open` -> `half_open`).

- **`HealthChecker(session: Session)`**:
  - `compute_system_status(now=None, display_only_checks: set[str] | None = None) -> tuple[str, list[str]]`: Evaluates the health check rules and returns overall system status (`NOMINAL`, `DEGRADED`, `CRITICAL`). Checks named in `display_only_checks` (default: `live_feed_crypto`, `live_feed_equity`, `ws_connection`, `intraday_prune`) can contribute at most `DEGRADED`.

### 5. v2.0.0 Real-Time Display Layer

Display-only. None of these feed training, features, or the forecast ledger.

- **`live_feed.CoinbaseWSClient(url, product_ids, on_tick, *, idle_timeout_sec=90, connect=None)`**:
  - `run_forever()` / `_run_sync()`: connect, subscribe (`ticker_batch` + `heartbeats`), dispatch each `Tick` to `on_tick`; auto-reconnect with capped backoff.
  - `status() -> str`, `seconds_since_last_message() -> float`, `stop()`.
- **`live_feed.coinbase_rest_candles(product_id, granularity=60, ...) -> list[Bar]`**: REST fallback used when the WS goes idle.
- **`intraday_store.IntradayRepository(session)`**:
  - `bucket_start(event_ts, interval) -> str`, `upsert_forming(ticker, interval, bucket_ts, price, volume=0.0, source="coinbase_ws")`, `close_bucket(ticker, interval, bucket_ts)`, `get_recent(ticker, interval, limit=200) -> list[IntradayBar]` (oldest-last), `prune(older_than_days, now=None) -> int`.
- **`intraday_store.LiveQuoteRepository(session)`**:
  - `upsert(ticker, price, ts, source, received_at=None)`, `get(ticker) -> LiveQuote | None`, `get_all() -> list[LiveQuote]`.
- **`providers/yfinance` — `get_intraday_bars(symbol, interval="5m", lookback_bars=...) -> list[Bar]`**: closed intraday bars; raises on HTTP 429.
- **`WorkerScheduler`** live jobs: `_on_tick`, `_start_live_feed` / `_stop_live_feed`, `job_ingest_equity_intraday` (5 min), `job_check_ws_idle` (WS-idle → REST fallback + `DEGRADED`), `job_prune_intraday` (daily).
- **`viz.add_live_price_line(fig, quotes, intraday) -> go.Figure`**: appends a `live` scatter (recent `intraday_bars` closes + current `live_quotes` point) and a faded provisional `forming` candle. Band / ribbon untouched.
- **`viz.CI_DISCLAIMER`** / **`viz.build_price_figure`** `uirevision=True`: calibration disclaimer surfaced on every figure; the CI band is always anchored to `P_close`.
- **`app` helpers**: `_refresh_for(asset_class) -> int` (2 s crypto / 15 s equity), `_delayed_badge(asset_class) -> str`, `load_live(engine, symbol, asset_class) -> (quotes, intraday)` — reads only `live_quotes` + `intraday_bars`, never a provider.

**M2 deviation**: the design's `ingestion.py` thin wrappers (`upsert_live_quote`, `ingest_intraday_bar`) were skipped — the repositories above are the intraday ingest seam directly.
