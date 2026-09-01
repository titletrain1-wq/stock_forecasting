"""Unit tests for schema table creation and database utilities."""

import sqlalchemy
from sqlmodel import Session

from stock_forecasting.database import (
    create_tables,
    get_engine,
    get_session,
    get_session_factory,
)
from stock_forecasting.schema import Ticker


def test_schema_creation(temp_db: sqlalchemy.Engine) -> None:
    """Verify all 10 required tables exist in the schema."""
    inspector = sqlalchemy.inspect(temp_db)
    tables = set(inspector.get_table_names())
    required = [
        "tickers",
        "ohlcv_bars",
        "model_runs",
        "prediction_snapshots",
        "accuracy_records",
        "quarantine_bars",
        "system_heartbeat",
        "link_metrics",
        "job_queue",
        "crypto_derivatives",
    ]
    for table in required:
        assert table in tables, f"Missing table: {table}"


def test_get_engine_creates_directory(tmp_path) -> None:
    """Verify get_engine creates parent directory when given a file path."""
    nested_db = tmp_path / "nested" / "dir" / "test.db"
    engine = get_engine(str(nested_db))
    create_tables(engine)
    assert nested_db.parent.exists()


def test_session_factory_and_contextmanager(temp_db: sqlalchemy.Engine) -> None:
    """Verify get_session_factory and get_session function correctly."""
    factory = get_session_factory(temp_db)
    session1 = factory()
    assert isinstance(session1, Session)
    session1.close()

    with get_session(temp_db) as session2:
        assert isinstance(session2, Session)
        ticker = Ticker(
            symbol="AAPL",
            asset_class="equity",
            display_name="Apple Inc.",
            provider="yfinance",
            provider_symbol="AAPL",
            price_basis="adjusted",
            added_at="2026-09-01T00:00:00Z",
            active=1,
        )
        session2.add(ticker)
        session2.commit()

        queried = session2.get(Ticker, "AAPL")
        assert queried is not None
        assert queried.symbol == "AAPL"
        assert queried.display_name == "Apple Inc."

def test_snapshot_immutability_trigger(temp_db: sqlalchemy.Engine) -> None:
    """Verify prediction columns are immutable but realized columns can be updated."""
    from stock_forecasting.schema import PredictionSnapshot, ModelRun
    from sqlalchemy.exc import DatabaseError
    import pytest

    with get_session(temp_db) as session:
        # Need a ModelRun first to satisfy foreign key
        model_run = ModelRun(
            ticker="AAPL", horizon="1d", model_type="ridge", model_version="1.0",
            code_git_sha="abc", trained_at="2026-09-01T00:00:00Z",
            train_start="2020-01-01", train_end="2021-01-01",
            hyperparams_json="{}", feature_list_json="[]", random_seed=42,
            wf_mae=1.0, wf_rmse=1.0, wf_dir_acc=0.5, wf_ci_cov=0.9, residual_std=1.0,
            artifact_path="test"
        )
        session.add(model_run)
        session.commit()
        session.refresh(model_run)

        snapshot = PredictionSnapshot(
            prediction_id="test_uuid", ticker="AAPL", made_at="2026-09-01T00:00:00Z",
            made_from_ts="2026-09-01T00:00:00Z", anchor_price=100.0, horizon="1d",
            target_ts="2026-09-02T00:00:00Z", predicted_return=0.01, predicted_price=101.0,
            lower_bound=99.0, upper_bound=103.0, model_type="ridge", model_version="1.0",
            model_run_id=model_run.id, explain_json="{}", input_is_stale=0
        )
        session.add(snapshot)
        session.commit()

        # Update realized price should succeed
        snapshot.realized_price = 102.0
        session.add(snapshot)
        session.commit()
        
        # Update predicted price should fail
        snapshot.predicted_price = 105.0
        session.add(snapshot)
        with pytest.raises(DatabaseError, match="Prediction columns are immutable"):
            session.commit()
