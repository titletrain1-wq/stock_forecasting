"""Repository for idempotent upserts into the ``crypto_derivatives`` table."""

from __future__ import annotations

from collections.abc import Sequence

from sqlmodel import Session, select

from stock_forecasting.providers.base import Derivative
from stock_forecasting.schema import CryptoDerivative


class CryptoDerivativeStore:
    """Writes day-aggregated funding rate + open interest, keyed on (ticker, ts)."""

    def __init__(self, session: Session) -> None:
        """Initialize with a database session."""
        self.session = session

    def upsert(
        self,
        ticker: str,
        derivatives: Sequence[Derivative],
        source: str = "dydx",
    ) -> int:
        """Insert new rows / update existing ones. Returns the count of NEW rows.

        A ``None`` field on an incoming row never overwrites an existing non-null
        value (open interest is snapshot-only, so historical rows carry ``None``).
        """
        if not derivatives:
            return 0

        timestamps = [d.ts for d in derivatives]
        existing = {
            row.ts: row
            for row in self.session.exec(
                select(CryptoDerivative).where(
                    CryptoDerivative.ticker == ticker,
                    CryptoDerivative.ts.in_(timestamps),
                )
            ).all()
        }

        inserted = 0
        for d in derivatives:
            row = existing.get(d.ts)
            if row is None:
                row = CryptoDerivative(
                    ticker=ticker,
                    ts=d.ts,
                    funding_rate=d.funding_rate,
                    open_interest=d.open_interest,
                    source=source,
                )
                self.session.add(row)
                existing[d.ts] = row
                inserted += 1
            else:
                if d.funding_rate is not None:
                    row.funding_rate = d.funding_rate
                if d.open_interest is not None:
                    row.open_interest = d.open_interest
                row.source = source
                self.session.add(row)

        self.session.commit()
        return inserted

    def get_range(
        self, ticker: str, start_ts: str, end_ts: str
    ) -> list[CryptoDerivative]:
        """Rows for ``ticker`` within [start_ts, end_ts], oldest first."""
        return list(
            self.session.exec(
                select(CryptoDerivative)
                .where(
                    CryptoDerivative.ticker == ticker,
                    CryptoDerivative.ts >= start_ts,
                    CryptoDerivative.ts <= end_ts,
                )
                .order_by(CryptoDerivative.ts.asc())
            ).all()
        )
