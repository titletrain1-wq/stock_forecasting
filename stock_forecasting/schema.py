"""SQLModel schema definitions for stock_forecasting.

Includes explicit table names (__tablename__) and foreign keys matching
the system architecture specification.
"""

from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field, SQLModel


class Ticker(SQLModel, table=True):
    """Tracked financial assets (stocks and crypto)."""

    __tablename__ = "tickers"

    symbol: str = Field(primary_key=True)
    asset_class: str  # "equity" | "crypto"
    display_name: str
    provider: str  # primary provider id
    provider_symbol: str
    price_basis: str  # "adjusted" | "raw"
    added_at: str
    active: int = 1  # 1 = polled


class OhlcvBar(SQLModel, table=True):
    """Historical and ingested OHLCV price bars."""

    __tablename__ = "ohlcv_bars"
    __table_args__ = (
        UniqueConstraint("ticker", "interval", "ts", name="uq_ohlcv_bar"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(foreign_key="tickers.symbol")
    interval: str = "1d"
    ts: str  # ISO-8601 UTC
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: float
    source: str  # provider id
    ingested_at: str


class IntradayBar(SQLModel, table=True):
    """Sub-daily OHLCV buckets for the live display path.

    Kept separate from ``ohlcv_bars`` (daily, immutable, ML training + evaluation).
    Auto-pruned by the worker after ``intraday_retention_days``.
    """

    __tablename__ = "intraday_bars"
    __table_args__ = (
        UniqueConstraint("ticker", "interval", "ts", name="uq_intraday_bar"),
        Index("ix_intraday_lookup", "ticker", "interval", "ts"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(foreign_key="tickers.symbol")
    interval: str  # "1m" | "5m"
    ts: str  # bucket start, ISO-8601 UTC
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    is_provisional: int = 1  # 1 = bucket forming or inside the delay window
    source: str  # "coinbase_ws" | "coinbase_rest" | "yfinance_intraday"
    ingested_at: str


class LiveQuote(SQLModel, table=True):
    """Current price anchor: one row per ticker, overwritten each tick."""

    __tablename__ = "live_quotes"

    ticker: str = Field(primary_key=True, foreign_key="tickers.symbol")
    price: float
    ts: str  # provider event time, UTC
    received_at: str  # wall-clock; drives display-path freshness
    source: str


class ModelRun(SQLModel, table=True):
    """Metadata and validation scores for trained model artifacts."""

    __tablename__ = "model_runs"

    id: int | None = Field(default=None, primary_key=True)
    ticker: str
    horizon: str  # "1d" | "5d" | "30d"
    model_type: str  # "ridge" | "random_forest"
    model_version: str  # semver
    code_git_sha: str
    trained_at: str
    train_start: str
    train_end: str
    hyperparams_json: str  # JSON dict
    feature_list_json: str  # JSON list
    random_seed: int
    wf_mae: float
    wf_rmse: float
    wf_dir_acc: float
    wf_ci_cov: float
    residual_std: float
    artifact_path: str  # .joblib location
    is_active: int = 1


class PredictionSnapshot(SQLModel, table=True):
    """Immutable ledger of forecasts and post-realization evaluations."""

    __tablename__ = "prediction_snapshots"
    __table_args__ = (
        Index(
            "ix_prediction_snapshots_lookup",
            "ticker",
            "horizon",
            "model_type",
            "made_at",
        ),
        Index("ix_prediction_snapshots_target_ts", "target_ts"),
    )

    prediction_id: str = Field(primary_key=True)  # UUID
    ticker: str
    made_at: str  # ISO-8601 UTC
    made_from_ts: str  # last bar used for features
    anchor_price: float
    horizon: str
    target_ts: str
    predicted_return: float
    predicted_price: float
    lower_bound: float
    upper_bound: float
    model_type: str
    model_version: str
    model_run_id: int = Field(foreign_key="model_runs.id")
    explain_json: str  # JSON dict
    input_is_stale: int
    # Evaluator fills these (write-once):
    realized_price: float | None = None
    realized_return: float | None = None
    evaluated_at: str | None = None
    error_abs: float | None = None
    error_signed: float | None = None
    is_direction_hit: int | None = None
    is_within_ci: int | None = None
    eval_attempts: int = 0


class AccuracyRecord(SQLModel, table=True):
    """Cached rolling accuracy and evaluation metrics."""

    __tablename__ = "accuracy_records"
    __table_args__ = (
        UniqueConstraint(
            "scope",
            "ticker",
            "horizon",
            "model_type",
            "window",
            name="uq_accuracy_record",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    scope: str  # "ticker" | "global"
    ticker: str | None = None
    horizon: str
    model_type: str
    n: int
    mae: float
    rmse: float
    dir_acc: float
    ci_coverage: float
    mae_price_pct: float
    window: str = "all"
    is_trustworthy: int
    updated_at: str


class QuarantineBar(SQLModel, table=True):
    """Quarantined anomaly/malformed bars failing boundary checks."""

    __tablename__ = "quarantine_bars"

    id: int | None = Field(default=None, primary_key=True)
    ticker: str
    raw_json: str
    reason: str
    provider: str
    detected_at: str


class SystemHeartbeat(SQLModel, table=True):
    """Watchdog process heartbeats and health status."""

    __tablename__ = "system_heartbeat"

    job_type: str = Field(primary_key=True)
    worker_pid: int | None = None
    last_pulse_ts: str | None = None
    last_success_ts: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0


class LinkMetrics(SQLModel, table=True):
    """External API provider latency and circuit breaker status."""

    __tablename__ = "link_metrics"

    provider: str = Field(primary_key=True)
    rtt_p50_ms: float | None = None
    rtt_p95_ms: float | None = None
    rtt_jitter_ms: float | None = None
    error_rate: float = 0.0
    consecutive_failures: int = 0
    breaker_state: str = "closed"
    breaker_opened_at: str | None = None
    calls_today: int = 0
    daily_limit: int
    quota_pct: float = 0.0
    updated_at: str


class JobQueue(SQLModel, table=True):
    """Asynchronous and on-demand job queue for worker processing."""

    __tablename__ = "job_queue"

    id: int | None = Field(default=None, primary_key=True)
    job_type: str  # "forecast_now" | "backfill"
    payload_json: str
    status: str = "pending"
    requested_at: str
    finished_at: str | None = None
    error: str | None = None


class CryptoDerivative(SQLModel, table=True):
    """Crypto derivative metrics (funding rate, open interest)."""

    __tablename__ = "crypto_derivatives"
    __table_args__ = (UniqueConstraint("ticker", "ts", name="uq_crypto_derivative"),)

    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(foreign_key="tickers.symbol")
    ts: str  # UTC
    funding_rate: float | None = None
    open_interest: float | None = None
    source: str = "dydx"


class IntradayBarsHistory(SQLModel, table=True):
    """Immutable intraday bars store for ML training and grading (365-day retention).

    Separate from ``intraday_bars`` (7-day display cache). Written once per closed bar.
    Both training pipeline and evaluator read from this table.
    """

    __tablename__ = "intraday_bars_history"
    __table_args__ = (
        UniqueConstraint("ticker", "interval", "ts", name="uq_intraday_bars_history"),
        Index("ix_intraday_history_lookup", "ticker", "interval", "ts"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(foreign_key="tickers.symbol")
    interval: str  # "1m" | "5m"
    ts: str  # bucket start, ISO-8601 UTC (closed bars only)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    source: str  # "coinbase_rest" | "coinbase_ws_finalized"
    ingested_at: str


class IntradayPredictionSnapshot(SQLModel, table=True):
    """Intraday forecast ledger (1h and 4h horizons, crypto only).

    Written by forecast-writer worker job (sole writer, hourly).
    Graded by evaluator job when targets mature.
    """

    __tablename__ = "intraday_prediction_snapshots"
    __table_args__ = (
        UniqueConstraint("ticker", "horizon", "anchor_ts", name="uq_intraday_forecast"),
        Index("ix_intraday_pred_lookup", "ticker", "horizon", "made_at"),
        Index("ix_intraday_pred_target_ts", "target_ts"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(foreign_key="tickers.symbol")
    horizon: str  # "1h" | "4h"
    made_at: str  # wall-clock UTC when forecast was written
    anchor_ts: str  # timestamp of the closed bar used for features
    anchor_price: float  # close price of anchor bar
    predicted_return: float  # model's log-return forecast
    predicted_price: float  # anchor_price * exp(predicted_return)
    ci_lower_return: float  # HAR-RV lower bound (log-return space)
    ci_upper_return: float  # HAR-RV upper bound (log-return space)
    ci_lower_price: float  # denormalized to price space
    ci_upper_price: float  # denormalized to price space
    target_ts: str  # when prediction target matures
    model_version: str  # version string of model used
    model_sha: str  # git SHA of model training code


class IntradayAccuracyRecord(SQLModel, table=True):
    """Intraday grading results (written by evaluator job when target matures)."""

    __tablename__ = "intraday_accuracy_records"
    __table_args__ = (Index("ix_intraday_acc_pred_id", "prediction_id"),)

    id: int | None = Field(default=None, primary_key=True)
    prediction_id: int = Field(foreign_key="intraday_prediction_snapshots.id")
    ticker: str = Field(foreign_key="tickers.symbol")
    horizon: str  # "1h" | "4h"
    graded_at: str  # wall-clock when grade was computed
    realized_return: float | None = (
        None  # actual log-return (populated when bar matures)
    )
    realized_price: float | None = None  # actual close price at target_ts
    signed_error: float | None = None  # realized_return - predicted_return
    abs_error_pct: float | None = None  # |signed_error| * 100
    direction_hit: int | None = None  # 1 if sign(predicted) == sign(realized), else 0
    ci_cover: int | None = None  # 1 if realized_return in [ci_lower, ci_upper], else 0
    grading_attempts: int = 0  # incremented if bar not yet matured
