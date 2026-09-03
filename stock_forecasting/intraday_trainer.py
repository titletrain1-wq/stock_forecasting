"""Intraday trainer: backfill orchestration (data pipeline) and model training (future)."""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlmodel import Session, select

from stock_forecasting.config import get_settings
from stock_forecasting.database import get_session
from stock_forecasting.intraday_pipeline import (
    as_of_join_funding,
    fetch_funding_rates_from_db,
    fetch_intraday_bars_5m,
    filter_closed_bar_anchors,
)
from stock_forecasting.providers.coinbase import CoinbaseProvider
from stock_forecasting.schema import Ticker

logger = logging.getLogger(__name__)


def backfill_intraday_bars(
    session: Session | None = None,
    tickers: list[str] | None = None,
    test_mode: bool = False,
) -> None:
    """Main entry point: backfill 365 days of intraday bars into intraday_bars_history.

    Args:
        session: SQLModel session (creates new if None).
        tickers: List of tickers to backfill (defaults to ['BTC-USD', 'ETH-USD']).
        test_mode: If True, mock the API calls for testing.
    """
    if tickers is None:
        tickers = ["BTC-USD", "ETH-USD"]

    settings = get_settings()
    lookback_days = settings.intraday_lookback_days
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=lookback_days)

    session = session or get_session().__enter__()

    try:
        # Ensure tickers exist in the database
        for ticker_str in tickers:
            existing = session.exec(
                select(Ticker).where(Ticker.symbol == ticker_str)
            ).first()
            if not existing:
                session.add(
                    Ticker(
                        symbol=ticker_str,
                        asset_class="crypto",
                        display_name=ticker_str,
                        provider="coinbase",
                        provider_symbol=ticker_str,
                        price_basis="raw",
                        added_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        active=1,
                    )
                )
                session.commit()

        # Initialize providers
        coinbase = CoinbaseProvider()

        for ticker_str in tickers:
            logger.info(
                f"Backfilling {ticker_str} intraday bars: {start_date} to {end_date}"
            )

            # Fetch 5-minute bars
            bars_list = fetch_intraday_bars_5m(
                coinbase, ticker_str, start_date, end_date
            )
            logger.info(f"  Fetched {len(bars_list)} 5m bars for {ticker_str}")

            if not bars_list:
                logger.warning(f"No bars fetched for {ticker_str}")
                continue

            # Convert to DataFrame
            import pandas as pd

            bars_df = pd.DataFrame(bars_list)
            bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True)

            # Fetch funding rates from crypto_derivatives table (dYdX ingestion)
            funding_df = fetch_funding_rates_from_db(
                session, ticker_str, start_date, end_date
            )
            logger.info(
                f"  Fetched {len(funding_df)} daily funding rates for {ticker_str}"
            )

            # Perform as-of join (validates no lookahead per F9)
            _, joined_df = as_of_join_funding(bars_df, funding_df)
            logger.info(f"  After as-of join: {len(joined_df)} bars")

            # Filter to closed-bar anchors
            anchored_df = filter_closed_bar_anchors(joined_df)
            logger.info(f"  After anchor filter: {len(anchored_df)} bars")

            # Write to database with INSERT OR IGNORE (dedup via unique constraint uq_intraday_bars_history)
            ingested_now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

            insert_sql = """
                INSERT OR IGNORE INTO intraday_bars_history (ticker, interval, ts, open, high, low, close, volume, source, ingested_at)
                VALUES (:ticker, :interval, :ts, :open, :high, :low, :close, :volume, :source, :ingested_at)
            """
            rows_to_insert = []
            for _, row in anchored_df.iterrows():
                ts_str = row["ts"].isoformat().replace("+00:00", "Z")
                rows_to_insert.append(
                    {
                        "ticker": ticker_str,
                        "interval": "5m",
                        "ts": ts_str,
                        "open": float(row["o"]),
                        "high": float(row["h"]),
                        "low": float(row["l"]),
                        "close": float(row["c"]),
                        "volume": float(row["v"]),
                        "source": "coinbase_rest",
                        "ingested_at": ingested_now,
                    }
                )

            # Execute batch insert with INSERT OR IGNORE
            connection = session.connection()
            connection.execute(text(insert_sql), rows_to_insert)
            session.commit()
            logger.info(
                f"  Wrote {len(anchored_df)} anchored bars to intraday_bars_history (INSERT OR IGNORE)"
            )

    except Exception:
        logger.exception("Backfill failed")
        session.rollback()
        raise


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    backfill_intraday_bars()
    sys.exit(0)
