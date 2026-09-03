"""Database session factory, engine creation, and table initialization."""

import logging
import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel
from sqlmodel import create_engine as create_sqlmodel_engine

import stock_forecasting.schema  # noqa: F401 - Register schema classes in metadata
from stock_forecasting.config import get_settings


class _LibsqlConnProxy:
    """Wrap a libsql (Turso) connection so SQLAlchemy's pysqlite dialect can
    drive it. The libsql Connection is a native object that rejects attribute
    assignment and lacks a few sqlite3-only hooks (create_function etc.) that
    the dialect calls on connect - stub them as no-ops (we don't use them)."""

    def __init__(self, conn: object) -> None:
        object.__setattr__(self, "_c", conn)

    def __getattr__(self, name: str) -> object:
        return getattr(self._c, name)

    def create_function(self, *a: object, **k: object) -> None:
        pass

    def create_aggregate(self, *a: object, **k: object) -> None:
        pass

    def set_progress_handler(self, *a: object, **k: object) -> None:
        pass

    def set_trace_callback(self, *a: object, **k: object) -> None:
        pass


_JWT_RE = re.compile(r"ey[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_DBURL_RE = re.compile(r"(?:libsql|wss?|https?)://[A-Za-z0-9._-]+(?::\d+)?")


def _clean_token(raw: str) -> str:
    """Extract the bare JWT from a possibly mis-pasted secret value.

    Tolerates wrapped lines, surrounding quotes, and the whole TOML/env line
    (``TURSO_AUTH_TOKEN = "eyJ..."``) being pasted in as the value - all of
    which otherwise reach Turso as a malformed JWT or an invalid HTTP header.
    """
    squashed = "".join(raw.split())
    match = _JWT_RE.search(squashed)
    return match.group(0) if match else squashed.strip("\"'")


def _clean_db_url(raw: str) -> str:
    """Extract the bare database URL from a possibly mis-pasted secret value.

    Returns a full ``libsql://<host>`` URL - ``libsql.connect()`` treats a bare
    hostname as a LOCAL file path and silently never reaches Turso.
    """
    squashed = "".join(raw.split())
    match = _DBURL_RE.search(squashed)
    if match:
        return match.group(0).rstrip("/")
    host = squashed.strip("\"'").split("?", 1)[0].rstrip("/")
    # Fall back to a bare host (possibly with the key name stripped off).
    host = host.rsplit("=", 1)[-1].strip("\"'")
    return host if "://" in host else f"libsql://{host}"


def _make_turso_engine(raw_url: str, auth_token: str) -> Engine:
    """Build a SQLAlchemy Engine backed by a remote Turso/libSQL database.

    Uses the `libsql` package (cross-platform wheels) via a creator function on
    the stock `sqlite://` dialect - avoids `sqlalchemy-libsql`, whose
    `libsql-experimental` dependency 308-redirects on connect and has no
    Windows wheel.
    """
    import libsql

    token = _clean_token(auth_token)
    conn_url = _clean_db_url(raw_url)

    def _creator() -> object:
        return _LibsqlConnProxy(libsql.connect(conn_url, auth_token=token))

    return create_engine(
        "sqlite://",
        creator=_creator,
        poolclass=StaticPool,
        isolation_level="AUTOCOMMIT",
    )


def create_tables(engine: Engine) -> None:
    """Create all SQLModel metadata tables and configure SQLite pragmas."""
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        # Local-SQLite tuning PRAGMAs. Turso/libSQL manages journaling itself
        # and rejects these ("SQL not allowed statement") - harmless to skip.
        for pragma in ("PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=5000"):
            try:
                conn.execute(text(pragma))
            except Exception as exc:  # noqa: BLE001 - PRAGMA support is backend-specific
                logging.getLogger(__name__).debug("skipped %s: %s", pragma, exc)

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


def seed_watchlist(engine: Engine) -> int:
    """Ensure the configured watchlist symbols exist as active Ticker rows.

    Idempotent. Returns the number of tickers inserted. Crypto is anything
    ending in ``-USD`` (Coinbase), everything else is a yfinance equity.
    """
    from datetime import UTC, datetime

    from sqlmodel import Session, select

    from stock_forecasting.schema import Ticker

    symbols = [
        s.strip().upper() for s in get_settings().watchlist.split(",") if s.strip()
    ]
    inserted = 0
    with Session(engine) as session:
        for sym in symbols:
            if session.exec(select(Ticker).where(Ticker.symbol == sym)).first():
                continue
            is_crypto = sym.endswith("-USD")
            session.add(
                Ticker(
                    symbol=sym,
                    asset_class="crypto" if is_crypto else "equity",
                    display_name=sym,
                    provider="coinbase" if is_crypto else "yfinance",
                    provider_symbol=sym,
                    price_basis="raw" if is_crypto else "adjusted",
                    added_at=datetime.now(UTC).isoformat(),
                    active=1,
                )
            )
            inserted += 1
        if inserted:
            session.commit()
    return inserted


def get_engine(db_path: str | None = None) -> Engine:
    """Create SQLite or libSQL engine.

    If TURSO_DATABASE_URL is set in environment, use libSQL/Turso (remote).
    Otherwise, use local SQLite from db_path.
    """
    settings = get_settings()

    # Turso configured -> remote libSQL engine.
    if settings.turso_database_url and settings.turso_database_url.strip():
        return _make_turso_engine(
            settings.turso_database_url.strip(), settings.turso_auth_token.strip()
        )

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
