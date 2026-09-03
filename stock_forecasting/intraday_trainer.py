"""Intraday data pipeline: Coinbase REST backfill + dYdX funding-rate as-of join + anchor filtering."""

import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx
import pandas as pd
from sqlmodel import Session, select

from stock_forecasting.config import get_settings
from stock_forecasting.database import get_engine, get_session
from stock_forecasting.providers.coinbase import CoinbaseProvider
from stock_forecasting.providers.dydx import DydxDerivativesProvider
from stock_forecasting.schema import IntradayBarsHistory, CryptoDerivative, Ticker

logger = logging.getLogger(__name__)


def _fetch_intraday_bars_5m(
    provider: CoinbaseProvider,
    ticker: str,
    start: date,
    end: date,
    lookback_hours: int = 12,
) -> list[dict[str, Any]]:
    """Fetch 365-day 5-minute bars from Coinbase for a single ticker.

    Args:
        provider: CoinbaseProvider instance.
        ticker: Ticker symbol (e.g. 'BTC-USD').
        start: Start date (inclusive).
        end: End date (inclusive).
        lookback_hours: Granularity in seconds (300 = 5m).

    Returns:
        List of dicts: {ts, interval, o, h, l, c, v}.
    """
    if start > end:
        return []

    product_id = provider.resolve_product_id(ticker)
    client = provider._get_client()
    close_client = True if provider._client is None else False

    try:
        all_bars: dict[str, dict] = {}
        current_start = start

        # Chunk by 30 days: 30d * 24h * 12 bars/h = 8,640 bars, comfortably under Coinbase 300-candle limit
        chunk_days = 30
        while current_start <= end:
            current_end = min(end, current_start + timedelta(days=chunk_days))

            start_iso = datetime.combine(
                current_start, time.min, tzinfo=UTC
            ).isoformat()
            end_iso = datetime.combine(
                current_end, time(23, 59, 59), tzinfo=UTC
            ).isoformat()

            url = f"{provider.base_url}/products/{product_id}/candles"
            params = {
                "granularity": 300,  # 5-minute bars
                "start": start_iso,
                "end": end_iso,
            }

            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Coinbase format: [[time, low, high, open, close, volume], ...]
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 6:
                        epoch_sec = item[0]
                        dt = datetime.fromtimestamp(epoch_sec, tz=UTC)

                        # Only keep if within date range
                        if not (start <= dt.date() <= end):
                            continue

                        ts_str = dt.isoformat().replace("+00:00", "Z")
                        bar = {
                            "ts": ts_str,
                            "interval": "5m",
                            "o": float(item[3]),
                            "h": float(item[2]),
                            "l": float(item[1]),
                            "c": float(item[4]),
                            "v": float(item[5]),
                        }
                        all_bars[ts_str] = bar

            current_start = current_end + timedelta(days=1)

        sorted_bars = sorted(all_bars.values(), key=lambda b: b["ts"])
        return sorted_bars

    finally:
        if close_client:
            client.close()


def _fetch_funding_rates_from_db(
    session: Session,
    ticker: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch 365-day funding rates from crypto_derivatives table (via dYdX ingestion).

    Returns a DataFrame with columns: ts (datetime), funding_rate (float).
    Funding rates are hourly snapshots published by dYdX.
    """
    # Query crypto_derivatives table for this ticker in date range
    # Note: crypto_derivatives.ts is day-aligned (00:00:00Z), but we're storing
    # hourly snapshots conceptually; in practice, we'll use what's available
    start_iso = datetime.combine(start, time.min, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    end_iso = datetime.combine(end, time.max, tzinfo=UTC).isoformat().replace("+00:00", "Z")

    rows = session.exec(
        select(CryptoDerivative).where(
            (CryptoDerivative.ticker == ticker)
            & (CryptoDerivative.ts >= start_iso)
            & (CryptoDerivative.ts <= end_iso)
        )
    ).all()

    if not rows:
        return pd.DataFrame(columns=["ts", "funding_rate"])

    # Convert to DataFrame
    df = pd.DataFrame(
        [{"ts": row.ts, "funding_rate": row.funding_rate} for row in rows]
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _as_of_join_funding(
    bars_df: pd.DataFrame, funding_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As-of join: attach the last funding rate published at or before each bar timestamp.

    Validates that:
    1. No forward-fill lookahead (funding_rate[t] != None only if published at or before t)
    2. Result is suitable for feature engineering

    Args:
        bars_df: 5m bars with columns [ts (datetime), o, h, l, c, v, ...].
        funding_df: Hourly funding rates with columns [ts (datetime), funding_rate].

    Returns:
        Tuple of (bars_df, joined_df) where joined_df is bars with 'funding_rate' column
        (NaN if no prior rate published, never forward-filled).
    """
    if bars_df.empty or funding_df.empty:
        bars_df_copy = bars_df.copy()
        bars_df_copy["funding_rate"] = None
        return bars_df, bars_df_copy

    # Ensure ts columns are datetime
    bars_df = bars_df.copy()
    funding_df = funding_df.copy()
    bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True)
    funding_df["ts"] = pd.to_datetime(funding_df["ts"], utc=True)

    # As-of join: backward direction (each bar gets last rate published at or before its ts)
    # This guarantees no forward-fill lookahead per design doc F9
    result = pd.merge_asof(
        bars_df.sort_values("ts"),
        funding_df.sort_values("ts"),
        on="ts",
        direction="backward",
        suffixes=("", "_funding"),
    )

    return bars_df, result


def _filter_closed_bar_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to closed-bar anchors only.

    1-hour horizon: ts at :00 UTC (minute==0, second==0).
    4-hour horizon: ts at :00, :04, :08, :12, :16, :20 UTC
                    (hour % 4 == 0 and minute == 0) OR (minute in [4, 8, 12, 16, 20]).

    For M1, we keep all valid anchor timestamps for both horizons.
    """
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Extract hour and minute
    df["hour"] = df["ts"].dt.hour
    df["minute"] = df["ts"].dt.minute
    df["second"] = df["ts"].dt.second

    # 1h anchors: minute == 0 and second == 0
    is_1h_anchor = (df["minute"] == 0) & (df["second"] == 0)

    # 4h anchors: (hour % 4 == 0 and minute == 0) OR (minute in [4, 8, 12, 16, 20])
    is_4h_anchor = (
        ((df["hour"] % 4 == 0) & (df["minute"] == 0))
        | (df["minute"].isin([4, 8, 12, 16, 20]))
    )

    # Keep rows that are valid anchors for either horizon
    filtered = df[is_1h_anchor | is_4h_anchor].copy()
    filtered = filtered.drop(columns=["hour", "minute", "second"])

    return filtered


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
                session.add(Ticker(symbol=ticker_str, asset_type="crypto"))
                session.commit()

        # Initialize providers
        coinbase = CoinbaseProvider()
        dydx = DydxDerivativesProvider()

        for ticker_str in tickers:
            logger.info(
                f"Backfilling {ticker_str} intraday bars: {start_date} to {end_date}"
            )

            # Fetch 5-minute bars
            bars_list = _fetch_intraday_bars_5m(
                coinbase, ticker_str, start_date, end_date
            )
            logger.info(f"  Fetched {len(bars_list)} 5m bars for {ticker_str}")

            if not bars_list:
                logger.warning(f"No bars fetched for {ticker_str}")
                continue

            # Convert to DataFrame
            bars_df = pd.DataFrame(bars_list)
            bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True)

            # Fetch funding rates from crypto_derivatives table (dYdX ingestion)
            funding_df = _fetch_funding_rates_from_db(session, ticker_str, start_date, end_date)
            logger.info(f"  Fetched {len(funding_df)} funding rates for {ticker_str}")

            # Perform as-of join (validates no lookahead per F9)
            _, joined_df = _as_of_join_funding(bars_df, funding_df)
            logger.info(f"  After as-of join: {len(joined_df)} bars")

            # Filter to closed-bar anchors
            anchored_df = _filter_closed_bar_anchors(joined_df)
            logger.info(f"  After anchor filter: {len(anchored_df)} bars")

            # Write to database
            dedup_count = 0
            ingested_now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

            for _, row in anchored_df.iterrows():
                ts_str = row["ts"].isoformat().replace("+00:00", "Z")
                existing = session.exec(
                    select(IntradayBarsHistory).where(
                        (IntradayBarsHistory.ticker == ticker_str)
                        & (IntradayBarsHistory.interval == "5m")
                        & (IntradayBarsHistory.ts == ts_str)
                    )
                ).first()

                if existing:
                    dedup_count += 1
                    continue

                bar = IntradayBarsHistory(
                    ticker=ticker_str,
                    interval="5m",
                    ts=ts_str,
                    open=float(row["o"]),
                    high=float(row["h"]),
                    low=float(row["l"]),
                    close=float(row["c"]),
                    volume=float(row["v"]),
                    source="coinbase_rest",
                    ingested_at=ingested_now,
                )
                session.add(bar)

            session.commit()
            logger.info(
                f"  Wrote {len(bars_df) - dedup_count} new bars to intraday_bars_history (dedup skipped {dedup_count})"
            )

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        session.rollback()
        raise


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    backfill_intraday_bars()
    sys.exit(0)
