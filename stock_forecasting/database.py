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

        # Enforce prediction snapshot immutability
        trigger_sql = """
        CREATE TRIGGER IF NOT EXISTS enforce_snapshot_immutability
        BEFORE UPDATE ON prediction_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'Prediction columns are immutable')
            WHERE OLD.prediction_id != NEW.prediction_id
               OR OLD.ticker != NEW.ticker
               OR OLD.made_at != NEW.made_at
               OR OLD.made_from_ts != NEW.made_from_ts
               OR OLD.anchor_price != NEW.anchor_price
               OR OLD.horizon != NEW.horizon
               OR OLD.target_ts != NEW.target_ts
               OR OLD.predicted_return != NEW.predicted_return
               OR OLD.predicted_price != NEW.predicted_price
               OR OLD.lower_bound != NEW.lower_bound
               OR OLD.upper_bound != NEW.upper_bound
               OR OLD.model_type != NEW.model_type
               OR OLD.model_version != NEW.model_version
               OR OLD.model_run_id != NEW.model_run_id
               OR OLD.explain_json != NEW.explain_json
               OR OLD.input_is_stale != NEW.input_is_stale;
        END;
        """
        conn.execute(text(trigger_sql))


def get_engine(db_path: str | None = None) -> Engine:
    """Create SQLite or libSQL engine.

    If TURSO_DATABASE_URL is set in environment, use libSQL/Turso (remote).
    Otherwise, use local SQLite from db_path.
    """
    settings = get_settings()

    # Check if Turso is configured (both URL and preferably auth token)
    if settings.turso_database_url and settings.turso_database_url.strip():
        # Use libSQL/Turso engine (requires sqlalchemy-libsql package)
        # URL format: sqlite+libsql://[host]?authToken=[token]&secure=true
        url = settings.turso_database_url
        if settings.turso_auth_token:
            # Append auth token if provided
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}authToken={settings.turso_auth_token}"
        try:
            return create_sqlmodel_engine(
                url,
                echo=False,
                connect_args={"timeout": 5},
            )
        except ImportError as e:
            if "libsql" in str(e).lower():
                raise RuntimeError(
                    "sqlalchemy-libsql not installed. Install it with: pip install sqlalchemy-libsql"
                ) from e
            raise

    # Fall back to local SQLite
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
