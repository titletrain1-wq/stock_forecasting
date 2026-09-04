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
            inserted_count = sum(1 for b in valid_bars if b.ts not in existing_ts)
            self._bulk_upsert(ticker, interval, source, now_str, valid_bars)

        self.session.commit()
        return inserted_count

    def _bulk_upsert(
        self,
        ticker: str,
        interval: str,
        source: str,
        now_str: str,
        bars: list[Bar],
    ) -> None:
        """Idempotent bulk write of `bars` as ONE multi-row INSERT per chunk.

        A single ``INSERT ... VALUES (..),(..),.. ON CONFLICT DO UPDATE`` is one
        round trip for the whole chunk. `executemany` over a remote libSQL/Turso
        connection instead does one network round trip *per row* (~200ms each),
        which turned a 500-bar backfill into ~2 minutes. `ON CONFLICT` also
        sidesteps SQLAlchemy's RETURNING batch-insert path, which libSQL rejects
        on a duplicate and which then poisons the session for later tickers.
        """
        cols = (
            "ticker",
            "interval",
            "ts",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "source",
            "ingested_at",
        )
        update = ", ".join(f"{c}=excluded.{c}" for c in cols[3:])
        conn = self.session.connection()
        chunk = 100  # 100 rows * 11 cols = 1100 params, well under libSQL's cap
        for start in range(0, len(bars), chunk):
            batch = bars[start : start + chunk]
            rows_sql = ", ".join(
                f"(:t{i}, :iv{i}, :ts{i}, :o{i}, :h{i}, :l{i}, :c{i}, :ac{i}, "
                f":v{i}, :s{i}, :ia{i})"
                for i in range(len(batch))
            )
            params: dict[str, object] = {}
            for i, b in enumerate(batch):
                params.update(
                    {
                        f"t{i}": ticker,
                        f"iv{i}": interval,
                        f"ts{i}": b.ts,
                        f"o{i}": b.open,
                        f"h{i}": b.high,
                        f"l{i}": b.low,
                        f"c{i}": b.close,
                        f"ac{i}": b.adj_close,
                        f"v{i}": b.volume,
                        f"s{i}": source,
                        f"ia{i}": now_str,
                    }
                )
            conn.execute(
                text(
                    f"INSERT INTO ohlcv_bars ({', '.join(cols)}) VALUES {rows_sql} "
                    f"ON CONFLICT(ticker, interval, ts) DO UPDATE SET {update}"
                ),
                params,
            )

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

    def get_up_to(
        self,
        ticker: str,
        cutoff_ts: str,
        limit: int | None = None,
        interval: str = "1d",
    ) -> list[OhlcvBar]:
        """Fetch bars up to and including cutoff_ts, ordered ascending by timestamp.

        Used for walk-forward backtesting: ensures forecasts made as-of T only
        consume bars with ts <= T (no lookahead).

        Args:
            ticker: Ticker symbol.
            cutoff_ts: ISO-8601 UTC timestamp (inclusive upper bound).
            limit: Maximum number of bars to retrieve (None = all).
            interval: Bar interval (default '1d').

        Returns:
            List of OhlcvBar models sorted ascending by timestamp.
        """
        statement = (
            select(OhlcvBar)
            .where(
                OhlcvBar.ticker == ticker,
                OhlcvBar.interval == interval,
                OhlcvBar.ts <= cutoff_ts,
            )
            .order_by(OhlcvBar.ts.asc())
        )
        if limit is not None:
            statement = statement.limit(limit)
        return list(self.session.exec(statement).all())
