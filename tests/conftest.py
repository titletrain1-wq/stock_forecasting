"""Pytest fixtures for stock_forecasting test suite."""

from collections.abc import Generator

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, create_engine

from stock_forecasting.database import create_tables


@pytest.fixture
def temp_db() -> Engine:
    """Create in-memory SQLite DB with all schema tables created."""
    db_url = "sqlite:///:memory:"
    engine = create_engine(db_url, echo=False)
    create_tables(engine)
    return engine


@pytest.fixture
def db_session(temp_db: Engine) -> Generator[Session, None, None]:
    """Provide a database session bound to the in-memory test DB."""
    with Session(temp_db) as session:
        yield session
