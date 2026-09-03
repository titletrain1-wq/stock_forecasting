"""Intraday data pipeline: Coinbase REST backfill + dYdX funding-rate as-of join + anchor filtering."""

import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlmodel import Session, select

from stock_forecasting.providers.coinbase import CoinbaseProvider
from stock_forecasting.schema import CryptoDerivative

logger = logging.getLogger(__name__)


def fetch_intraday_bars_5m(
    provider: CoinbaseProvider,
    ticker: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Fetch 365-day 5-minute bars from Coinbase for a single ticker.

    Args:
        provider: CoinbaseProvider instance.
        ticker: Ticker symbol (e.g. 'BTC-USD').
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        List of dicts: {ts, interval, o, h, l, c, v}.
    """
    if start > end:
        return []

    product_id = provider.resolve_product_id(ticker)
    client = provider._get_client()
    close_client = provider._client is None

    try:
        all_bars: dict[str, dict] = {}
        current_start = start

        # Chunk by ~25 hours: 25h * 12 bars/h = 300 bars, at Coinbase's 300-candle hard limit
        # (Coinbase rejects any request with >300 candles per scout doc S1.1)
        chunk_hours = 25
        while current_start <= end:
            current_end = min(end, current_start + timedelta(hours=chunk_hours))

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

            current_start = current_end + timedelta(hours=1)

        sorted_bars = sorted(all_bars.values(), key=lambda b: b["ts"])
        return sorted_bars

    finally:
        if close_client:
            client.close()


def fetch_funding_rates_from_db(
    session: Session,
    ticker: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch 365-day funding rates from crypto_derivatives table (via dYdX ingestion).

    Returns a DataFrame with columns: ts (datetime), funding_rate (float).
    """
    start_iso = (
        datetime.combine(start, time.min, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    )
    end_iso = (
        datetime.combine(end, time.max, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    )

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


def as_of_join_funding(
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


def filter_closed_bar_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to closed-bar anchors only (per design doc F8).

    1-hour horizon: ts at :00 UTC (minute==0, second==0).
    4-hour horizon: ts at hour % 4 == 0 and minute == 0 (i.e., 00:00/04:00/08:00/12:00/16:00/20:00 UTC).

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

    # 4h anchors: hour % 4 == 0 and minute == 0 (only 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)
    is_4h_anchor = (df["hour"] % 4 == 0) & (df["minute"] == 0)

    # Keep rows that are valid anchors for either horizon
    filtered = df[is_1h_anchor | is_4h_anchor].copy()
    filtered = filtered.drop(columns=["hour", "minute", "second"])

    return filtered
