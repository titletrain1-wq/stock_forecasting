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
