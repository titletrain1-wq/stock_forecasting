"""YFinance market data provider implementation."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import yfinance as yf

from stock_forecasting.providers.base import Bar, DataProvider


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance package."""

    def get_daily_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Fetch daily historical OHLCV bars for a given symbol between start and end dates.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL').
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of Bar dataclass instances sorted chronologically.
        """
        # yfinance end date in download() is exclusive, so add 1 day to make it inclusive
        yf_end = end + timedelta(days=1)
        df = yf.download(
            symbol,
            start=start,
            end=yf_end,
            progress=False,
            auto_adjust=False,
        )
        return self._dataframe_to_bars(df)

    def get_latest_bars(self, symbol: str, lookback: int = 5) -> list[Bar]:
        """Fetch the most recent N closed daily bars for a given symbol.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL').
            lookback: Number of recent daily bars to return.

        Returns:
            List of Bar dataclass instances of length up to `lookback`.
        """
        if lookback <= 0:
            return []
        today = datetime.now(UTC).date()
        # Fetch a wider window (calendar days) to account for weekends/holidays
        start_date = today - timedelta(days=max(lookback * 3, 14))
        bars = self.get_daily_history(symbol, start=start_date, end=today)
        return bars[-lookback:]

    def _dataframe_to_bars(self, df: pd.DataFrame | None) -> list[Bar]:
        """Convert a yfinance DataFrame into a list of clean Bar objects."""
        if df is None or df.empty:
            return []

        # Handle MultiIndex columns (e.g. ('Close', 'AAPL') or ('AAPL', 'Close'))
        working_df = df.copy()
        if isinstance(working_df.columns, pd.MultiIndex):
            level_0 = [str(c).lower() for c in working_df.columns.get_level_values(0)]
            if any(k in level_0 for k in ["open", "high", "low", "close", "volume"]):
                working_df.columns = working_df.columns.get_level_values(0)
            else:
                working_df.columns = working_df.columns.get_level_values(-1)

        # Normalize column mapping
        col_map = {
            str(c).lower().replace(" ", "").replace("_", ""): c
            for c in working_df.columns
        }

        open_col = col_map.get("open")
        high_col = col_map.get("high")
        low_col = col_map.get("low")
        close_col = col_map.get("close")
        adj_close_col = col_map.get("adjclose") or col_map.get("adjustedclose")
        vol_col = col_map.get("volume")

        # If required price columns are missing, return empty
        if not open_col or not close_col:
            return []

        bars: list[Bar] = []
        for idx, row in working_df.iterrows():
            if pd.isna(row[close_col]) or pd.isna(row[open_col]):
                continue

            if hasattr(idx, "strftime"):
                ts_str = idx.strftime("%Y-%m-%dT00:00:00Z")
            else:
                ts_str = f"{str(idx)[:10]}T00:00:00Z"

            adj_val = (
                float(row[adj_close_col])
                if adj_close_col is not None and pd.notna(row[adj_close_col])
                else None
            )
            vol_val = (
                float(row[vol_col])
                if vol_col is not None and pd.notna(row[vol_col])
                else 0.0
            )

            bars.append(
                Bar(
                    ts=ts_str,
                    open=float(row[open_col]),
                    high=float(row[high_col]) if high_col else float(row[close_col]),
                    low=float(row[low_col]) if low_col else float(row[close_col]),
                    close=float(row[close_col]),
                    adj_close=adj_val,
                    volume=vol_val,
                )
            )
        return bars
