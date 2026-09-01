"""Repositories for the v2 live display path.

Kept separate from ``bar_store.BarRepository`` (daily ``ohlcv_bars``, immutable,
ML training + evaluation). ``intraday_bars`` is a short-retention cache; the
worker prunes it. ``live_quotes`` is the current-price anchor, one row per ticker.

Per docs/spikes/2026-09-01-M2-wal-contention.md the write path is per-tick DB
writes (no in-memory flush loop): keep each call in one short transaction.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session, delete, select

from stock_forecasting.schema import IntradayBar, LiveQuote

_INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600}


class IntradayRepository:
    """CRUD for ``intraday_bars`` (forming-bucket upsert, close, prune)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def bucket_start(self, event_ts: str, interval: str) -> str:
        """Floor an ISO-8601 timestamp to the start of its interval bucket."""
        raw = event_ts.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw).astimezone(UTC)
        step = _INTERVAL_SECONDS[interval]
        floored = (int(dt.timestamp()) // step) * step
        return datetime.fromtimestamp(floored, UTC).isoformat()

    def upsert_forming(
        self,
        ticker: str,
        interval: str,
        bucket_ts: str,
        price: float,
        volume: float = 0.0,
        source: str = "coinbase_ws",
    ) -> None:
        """Create or extend the forming bucket at ``bucket_ts``.

        ``open`` is fixed at the first tick; ``high``/``low`` track; ``close`` is
        the latest tick. The row stays ``is_provisional = 1`` until ``close_bucket``.
        """
        row = self.session.exec(
            select(IntradayBar).where(
                IntradayBar.ticker == ticker,
                IntradayBar.interval == interval,
                IntradayBar.ts == bucket_ts,
            )
        ).first()
        now = datetime.now(UTC).isoformat()
        if row is None:
            row = IntradayBar(
                ticker=ticker,
                interval=interval,
                ts=bucket_ts,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                is_provisional=1,
                source=source,
                ingested_at=now,
            )
        else:
            row.high = max(row.high, price)
            row.low = min(row.low, price)
            row.close = price
            row.volume = volume or row.volume
            row.source = source
            row.ingested_at = now
        self.session.add(row)
        self.session.commit()

    def close_bucket(self, ticker: str, interval: str, bucket_ts: str) -> None:
        """Mark a bucket finalized (``is_provisional = 0``)."""
        row = self.session.exec(
            select(IntradayBar).where(
                IntradayBar.ticker == ticker,
                IntradayBar.interval == interval,
                IntradayBar.ts == bucket_ts,
            )
        ).first()
        if row is not None and row.is_provisional != 0:
            row.is_provisional = 0
            self.session.add(row)
            self.session.commit()

    def get_recent(
        self, ticker: str, interval: str, limit: int = 200
    ) -> list[IntradayBar]:
        """Most recent ``limit`` buckets, oldest first (chart-ready order)."""
        rows = self.session.exec(
            select(IntradayBar)
            .where(IntradayBar.ticker == ticker, IntradayBar.interval == interval)
            .order_by(IntradayBar.ts.desc())
            .limit(limit)
        ).all()
        return list(reversed(rows))

    def prune(self, older_than_days: int, now: datetime | None = None) -> int:
        """Delete buckets whose ``ts`` is older than the retention window."""
        now = now or datetime.now(UTC)
        cutoff = now.timestamp() - older_than_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, UTC).isoformat()
        result = self.session.exec(
            delete(IntradayBar).where(IntradayBar.ts < cutoff_iso)
        )
        self.session.commit()
        return result.rowcount


class LiveQuoteRepository:
    """One-row-per-ticker current price anchor."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        ticker: str,
        price: float,
        ts: str,
        source: str,
        received_at: str | None = None,
    ) -> None:
        received_at = received_at or datetime.now(UTC).isoformat()
        row = self.session.get(LiveQuote, ticker)
        if row is None:
            row = LiveQuote(
                ticker=ticker,
                price=price,
                ts=ts,
                received_at=received_at,
                source=source,
            )
        else:
            row.price = price
            row.ts = ts
            row.received_at = received_at
            row.source = source
        self.session.add(row)
        self.session.commit()

    def get(self, ticker: str) -> LiveQuote | None:
        return self.session.get(LiveQuote, ticker)

    def get_all(self) -> list[LiveQuote]:
        return list(self.session.exec(select(LiveQuote)).all())
