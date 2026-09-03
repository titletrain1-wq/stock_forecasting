"""M0 Tests: Intraday schema, config, and initialization.

Tests that verify:
- New intraday tables exist in schema
- Intraday config settings parse correctly
- Intraday model directory exists
- Retention triggers are configured
"""

import os
import tempfile
from pathlib import Path

import pytest
import sqlalchemy

from stock_forecasting.config import get_settings
from stock_forecasting.database import create_tables, get_engine
from stock_forecasting.schema import (
    IntradayAccuracyRecord,
    IntradayBarsHistory,
    IntradayPredictionSnapshot,
    Ticker,
)


def test_intraday_bars_history_table_exists(temp_db: sqlalchemy.Engine) -> None:
    """M0 DoD: intraday_bars_history table exists with correct schema."""
    tables = set(sqlalchemy.inspect(temp_db).get_table_names())
    assert "intraday_bars_history" in tables

    # Verify unique constraint on (ticker, interval, ts)
    inspector = sqlalchemy.inspect(temp_db)
    constraints = [c["name"] for c in inspector.get_unique_constraints("intraday_bars_history")]
    assert "uq_intraday_bars_history" in constraints


def test_intraday_prediction_snapshots_table_exists(temp_db: sqlalchemy.Engine) -> None:
    """M0 DoD: intraday_prediction_snapshots table exists with correct schema."""
    tables = set(sqlalchemy.inspect(temp_db).get_table_names())
    assert "intraday_prediction_snapshots" in tables

    # Verify unique constraint on (ticker, horizon, anchor_ts) for dedup
    inspector = sqlalchemy.inspect(temp_db)
    constraints = [c["name"] for c in inspector.get_unique_constraints("intraday_prediction_snapshots")]
    assert "uq_intraday_forecast" in constraints


def test_intraday_accuracy_records_table_exists(temp_db: sqlalchemy.Engine) -> None:
    """M0 DoD: intraday_accuracy_records table exists with correct schema."""
    tables = set(sqlalchemy.inspect(temp_db).get_table_names())
    assert "intraday_accuracy_records" in tables


def test_intraday_bars_history_insert(temp_db: sqlalchemy.Engine) -> None:
    """M0 DoD: can insert/upsert rows into intraday_bars_history."""
    from sqlmodel import Session

    with Session(temp_db) as session:
        # Create a ticker first
        ticker = Ticker(
            symbol="BTC-USD",
            asset_class="crypto",
            display_name="Bitcoin",
            provider="coinbase",
            provider_symbol="BTC-USD",
            price_basis="raw",
            added_at="2026-09-01T00:00:00Z",
            active=1,
        )
        session.add(ticker)
        session.commit()

        # Insert intraday_bars_history row
        row = IntradayBarsHistory(
            ticker="BTC-USD",
            interval="5m",
            ts="2026-09-01T00:00:00Z",
            open=45000.0,
            high=45100.0,
            low=44900.0,
            close=45050.0,
            volume=100.0,
            source="coinbase_rest",
            ingested_at="2026-09-01T00:01:00Z",
        )
        session.add(row)
        session.commit()

        # Verify row was inserted
        queried = session.query(IntradayBarsHistory).filter_by(
            ticker="BTC-USD", interval="5m", ts="2026-09-01T00:00:00Z"
        ).first()
        assert queried is not None
        assert queried.close == 45050.0


def test_intraday_prediction_snapshot_insert(temp_db: sqlalchemy.Engine) -> None:
    """M0 DoD: can insert rows into intraday_prediction_snapshots."""
    from sqlmodel import Session

    with Session(temp_db) as session:
        # Create a ticker first
        ticker = Ticker(
            symbol="ETH-USD",
            asset_class="crypto",
            display_name="Ethereum",
            provider="coinbase",
            provider_symbol="ETH-USD",
            price_basis="raw",
            added_at="2026-09-01T00:00:00Z",
            active=1,
        )
        session.add(ticker)
        session.commit()

        # Insert prediction snapshot
        row = IntradayPredictionSnapshot(
            ticker="ETH-USD",
            horizon="1h",
            made_at="2026-09-01T10:00:00Z",
            anchor_ts="2026-09-01T10:00:00Z",
            anchor_price=2500.0,
            predicted_return=0.01,
            predicted_price=2525.0,
            ci_lower_return=-0.02,
            ci_upper_return=0.04,
            ci_lower_price=2450.0,
            ci_upper_price=2600.0,
            target_ts="2026-09-01T11:00:00Z",
            model_version="1.0.0",
            model_sha="abc123",
        )
        session.add(row)
        session.commit()

        # Verify row was inserted
        queried = session.query(IntradayPredictionSnapshot).filter_by(
            ticker="ETH-USD", horizon="1h", anchor_ts="2026-09-01T10:00:00Z"
        ).first()
        assert queried is not None
        assert queried.predicted_price == 2525.0


def test_intraday_accuracy_record_insert(temp_db: sqlalchemy.Engine) -> None:
    """M0 DoD: can insert rows into intraday_accuracy_records."""
    from sqlmodel import Session

    with Session(temp_db) as session:
        # Create ticker and prediction first
        ticker = Ticker(
            symbol="BTC-USD",
            asset_class="crypto",
            display_name="Bitcoin",
            provider="coinbase",
            provider_symbol="BTC-USD",
            price_basis="raw",
            added_at="2026-09-01T00:00:00Z",
            active=1,
        )
        session.add(ticker)
        session.commit()

        prediction = IntradayPredictionSnapshot(
            ticker="BTC-USD",
            horizon="4h",
            made_at="2026-09-01T08:00:00Z",
            anchor_ts="2026-09-01T08:00:00Z",
            anchor_price=45000.0,
            predicted_return=0.02,
            predicted_price=45900.0,
            ci_lower_return=-0.03,
            ci_upper_return=0.07,
            ci_lower_price=43650.0,
            ci_upper_price=48150.0,
            target_ts="2026-09-01T12:00:00Z",
            model_version="1.0.0",
            model_sha="abc123",
        )
        session.add(prediction)
        session.commit()

        # Insert accuracy record
        record = IntradayAccuracyRecord(
            prediction_id=prediction.id,
            ticker="BTC-USD",
            horizon="4h",
            graded_at="2026-09-01T12:01:00Z",
            realized_return=0.015,
            realized_price=45675.0,
            signed_error=-0.005,
            abs_error_pct=0.5,
            direction_hit=1,
            ci_cover=1,
            grading_attempts=1,
        )
        session.add(record)
        session.commit()

        # Verify row was inserted
        queried = session.query(IntradayAccuracyRecord).filter_by(
            prediction_id=prediction.id
        ).first()
        assert queried is not None
        assert queried.direction_hit == 1


def test_intraday_config_settings_parse() -> None:
    """M0 DoD: intraday config settings parse and have correct defaults."""
    settings = get_settings()

    # Required intraday settings per design doc §9.1
    assert hasattr(settings, "intraday_forecast_enabled")
    assert settings.intraday_forecast_enabled is True

    assert hasattr(settings, "intraday_lookback_days")
    assert settings.intraday_lookback_days == 365  # god ruling F3

    assert hasattr(settings, "intraday_bars_history_retention_days")
    assert settings.intraday_bars_history_retention_days == 365  # immutable ML store (F2)

    assert hasattr(settings, "intraday_forecast_writer_interval_seconds")
    assert settings.intraday_forecast_writer_interval_seconds == 3600

    assert hasattr(settings, "intraday_evaluator_interval_seconds")
    assert settings.intraday_evaluator_interval_seconds == 3600

    assert hasattr(settings, "intraday_model_dir")


def test_intraday_model_directory_can_be_created() -> None:
    """M0 DoD: intraday model directory can be created and exists."""
    settings = get_settings()

    # Create the directory if it doesn't exist
    model_dir = Path(settings.intraday_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Verify it exists
    assert model_dir.exists()
    assert model_dir.is_dir()

    # Clean up if it was empty (optional; keep it for manual inspection)


def test_retention_trigger_retention_days_correct() -> None:
    """M0 DoD: retention configuration is set to 365d, not 90d (F2 ruling)."""
    settings = get_settings()

    # Must be 365 days, not 90 (per F2 immutable ML store requirement)
    assert settings.intraday_bars_history_retention_days == 365
    assert settings.intraday_bars_history_retention_days != 90
