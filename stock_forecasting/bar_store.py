"""Repository for storing, validating, and querying OHLCV bars."""

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime

from sqlalchemy import text
from sqlmodel import Session, select

from stock_forecasting.providers.base import Bar
from stock_forecasting.schema import OhlcvBar, QuarantineBar


class BarRepository:
    """Repository managing OhlcvBar storage and QuarantineBar filtering."""

    def __init__(self, session: Session) -> None:
        """Initialize repository with a database session."""
        self.session = session

    def _validate_bar(self, bar: Bar) -> list[str]:
        """Validate bar attributes against boundary checks.

        Returns:
            List of failure reason codes (empty list if valid).
            - 'price_le_0' if close <= 0
            - 'ohlc_inconsistent' if low > high
            - 'negative_volume' if volume < 0
        """
        reasons: list[str] = []
        try:
            if not hasattr(bar, "close") or bar.close is None or bar.close <= 0:
                reasons.append("price_le_0")
            if (
                not hasattr(bar, "low")
                or not hasattr(bar, "high")
                or bar.low is None
                or bar.high is None
                or bar.low > bar.high
            ):
                reasons.append("ohlc_inconsistent")
            if not hasattr(bar, "volume") or bar.volume is None or bar.volume < 0:
                reasons.append("negative_volume")
        except (AttributeError, TypeError, ValueError):
            reasons.append("schema_error")
        return reasons

    def upsert_bars(
        self,
        ticker: str,
        bars: list[Bar],
        source: str,
        interval: str = "1d",
    ) -> int:
        """Validate and upsert bars into the database.

        Invalid bars are written to `quarantine_bars`.
        Valid bars are upserted into `ohlcv_bars`.

        Args:
            ticker: Ticker symbol (e.g. 'AAPL').
            bars: List of Bar objects to insert or update.
            source: Identifier of the data source provider (e.g. 'yfinance').
            interval: Bar interval (default '1d').

        Returns:
            Count of newly inserted OhlcvBar rows.
        """
        if not bars:
            return 0

        now_str = datetime.now(UTC).isoformat()
        today_utc = datetime.now(UTC).date()
        inserted_count = 0
        valid_bars: list[Bar] = []

        # Filter out bars from today (forming/provisional candles).
        # Spec requires closed bars only; today's bar is still forming.
        bars_filtered = []
        for b in bars:
            ts = b.ts if hasattr(b, "ts") else b.get("ts", "")
            if not ts.startswith(today_utc.isoformat()):
                bars_filtered.append(b)

        for bar in bars_filtered:
            reasons = self._validate_bar(bar)
            if reasons:
                raw_payload = (
                    asdict(bar)
                    if is_dataclass(bar)
                    else (bar if isinstance(bar, dict) else {"raw": str(bar)})
                )
                quarantine_row = QuarantineBar(
                    ticker=ticker,
                    raw_json=json.dumps(raw_payload),
                    reason="|".join(reasons),
                    provider=source,
                    detected_at=now_str,
                )
                self.session.add(quarantine_row)
            else:
                valid_bars.append(bar)

        if valid_bars:
            # How many are genuinely new (return value / logging only).
            existing_ts = {
                row
                for row in self.session.exec(
                    select(OhlcvBar.ts).where(
                        OhlcvBar.ticker == ticker,
                        OhlcvBar.interval == interval,
                        OhlcvBar.ts.in_([b.ts for b in valid_bars]),
                    )
                ).all()
            }
            new_count = sum(1 for b in valid_bars if b.ts not in existing_ts)

            # Single idempotent upsert. INSERT .. ON CONFLICT avoids the
            # read-then-write race and SQLAlchemy's RETURNING-based batch
            # insert, which libSQL/Turso rejects on a duplicate and which then
            # poisons the session for every later ticker. Chunked so one
            # request stays small over a remote (Turso) connection.
            stmt = text(
                "INSERT INTO ohlcv_bars "
                "(ticker, interval, ts, open, high, low, close, adj_close, "
                " volume, source, ingested_at) "
                "VALUES (:ticker, :interval, :ts, :open, :high, :low, :close, "
                " :adj_close, :volume, :source, :ingested_at) "
                "ON CONFLICT(ticker, interval, ts) DO UPDATE SET "
                " open=excluded.open, high=excluded.high, low=excluded.low, "
                " close=excluded.close, adj_close=excluded.adj_close, "
                " volume=excluded.volume, source=excluded.source, "
                " ingested_at=excluded.ingested_at"
            )
            conn = self.session.connection()
            chunk = 200
            for start in range(0, len(valid_bars), chunk):
                conn.execute(
                    stmt,
                    [
                        {
                            "ticker": ticker,
                            "interval": interval,
                            "ts": b.ts,
                            "open": b.open,
                            "high": b.high,
                            "low": b.low,
                            "close": b.close,
                            "adj_close": b.adj_close,
                            "volume": b.volume,
                            "source": source,
                            "ingested_at": now_str,
                        }
                        for b in valid_bars[start : start + chunk]
                    ],
                )
            inserted_count = new_count

        self.session.commit()
        return inserted_count

    def get_range(
        self,
        ticker: str,
        start_ts: str,
        end_ts: str,
        interval: str = "1d",
    ) -> list[OhlcvBar]:
        """Fetch bars within [start_ts, end_ts] ordered chronologically.

        Args:
            ticker: Ticker symbol.
            start_ts: Start ISO-8601 UTC timestamp (inclusive).
            end_ts: End ISO-8601 UTC timestamp (inclusive).
            interval: Bar interval (default '1d').

        Returns:
            List of OhlcvBar models sorted ascending by timestamp.
        """
        statement = (
            select(OhlcvBar)
            .where(
                OhlcvBar.ticker == ticker,
                OhlcvBar.interval == interval,
                OhlcvBar.ts >= start_ts,
                OhlcvBar.ts <= end_ts,
            )
            .order_by(OhlcvBar.ts.asc())
        )
        return list(self.session.exec(statement).all())

    def get_latest(
        self,
        ticker: str,
        limit: int = 1,
        interval: str = "1d",
    ) -> list[OhlcvBar]:
        """Fetch the most recent N bars for a ticker.

        Args:
            ticker: Ticker symbol.
            limit: Maximum number of bars to retrieve.
            interval: Bar interval (default '1d').

        Returns:
            List of OhlcvBar models sorted descending by timestamp.
        """
        statement = (
            select(OhlcvBar)
            .where(
                OhlcvBar.ticker == ticker,
                OhlcvBar.interval == interval,
            )
            .order_by(OhlcvBar.ts.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def latest_ts(self, ticker: str, interval: str = "1d") -> str | None:
        """Fetch timestamp of the latest available bar for a ticker."""
        latest = self.get_latest(ticker, limit=1, interval=interval)
        return latest[0].ts if latest else None
