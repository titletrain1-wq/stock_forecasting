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
  - `compute_system_status() -> tuple[str, list[str]]`: Evaluates 8 health check rules and returns overall system status (`NOMINAL`, `DEGRADED`, `CRITICAL`).
