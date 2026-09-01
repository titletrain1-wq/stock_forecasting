"""Database session factory, engine creation, and table initialization."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, text
from sqlmodel import Session, SQLModel
from sqlmodel import create_engine as create_sqlmodel_engine

import stock_forecasting.schema  # noqa: F401 - Register schema classes in metadata
from stock_forecasting.config import get_settings


def create_tables(engine: Engine) -> None:
    """Create all SQLModel metadata tables and configure SQLite pragmas."""
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA busy_timeout=5000"))


def get_engine(db_path: str | None = None) -> Engine:
    """Create SQLite engine ensuring parent directories exist."""
    settings = get_settings()
    path_str = db_path if db_path is not None else settings.db_path

    if path_str.startswith("sqlite://"):
        file_part = path_str.removeprefix("sqlite:///").removeprefix("sqlite://")
        if file_part and file_part != ":memory:":
            Path(file_part).parent.mkdir(parents=True, exist_ok=True)
        url = path_str
    elif path_str == ":memory:":
        url = "sqlite:///:memory:"
    else:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path_str}"

    return create_sqlmodel_engine(
        url,
        echo=False,
        connect_args={"timeout": 5},
    )


def get_session_factory(engine: Engine | None = None) -> Callable[[], Session]:
    """Return a factory function producing Session instances."""
    eng = engine if engine is not None else get_engine()
    return lambda: Session(eng)


@contextmanager
def get_session(engine: Engine | None = None) -> Generator[Session, None, None]:
    """Context manager yielding a database Session."""
    eng = engine if engine is not None else get_engine()
    with Session(eng) as session:
        yield session
