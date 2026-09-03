"""Intraday feature engineering: VWAP, volatility, EWMA, volume, returns, funding z-score, temporal."""

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlmodel import Session, select

from stock_forecasting.schema import CryptoDerivative, IntradayBarsHistory

logger = logging.getLogger(__name__)


def _compute_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute rolling VWAP over the last window bars (no lookahead)."""
    df = df.copy()
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    cumsum_tp = (df["tp"] * df["volume"]).rolling(window, min_periods=1).sum()
    cumsum_vol = df["volume"].rolling(window, min_periods=1).sum()
    vwap = cumsum_tp / (cumsum_vol + 1e-9)
    return vwap


def _compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) for volatility normalization."""
    df = df.copy()
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            np.abs(df["high"] - df["close"].shift(1)),
            np.abs(df["low"] - df["close"].shift(1)),
        ),
    )
    atr = df["tr"].rolling(window, min_periods=1).mean()
    return atr


class IntradayFeatureBuilder:
    """Compute intraday ML features from bars and funding rates (no lookahead)."""

    def __init__(self) -> None:
        """Initialize feature builder."""
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []

    def build_features(
        self, ticker: str, bars_df: pd.DataFrame, funding_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Build 8-feature matrix from intraday bars and funding rates.

        Args:
            ticker: Ticker symbol (e.g. 'BTC-USD').
            bars_df: 5-minute bars with columns [ts, open, high, low, close, volume].
            funding_df: Hourly funding rates with columns [ts, funding_rate] (optional).

        Returns:
            DataFrame with columns: ts, and 8 computed features (with NaN until warmup).
        """
        df = bars_df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)

        # 1. Intraday VWAP Distance (1h window = 12 bars, 4h window = 48 bars)
        vwap_1h = _compute_vwap(df, 12)
        vwap_std_1h = df["close"].rolling(12, min_periods=1).std()
        df["vwap_distance_1h"] = (df["close"] - vwap_1h) / (vwap_std_1h + 1e-9)

        vwap_4h = _compute_vwap(df, 48)
        vwap_std_4h = df["close"].rolling(48, min_periods=1).std()
        df["vwap_distance_4h"] = (df["close"] - vwap_4h) / (vwap_std_4h + 1e-9)

        # 2. Realized Volatility Ratio (short: 3 bars = 15m, long: 48 bars = 4h)
        vol_short = df["close"].pct_change().rolling(3, min_periods=1).std()
        vol_long = df["close"].pct_change().rolling(48, min_periods=1).std()
        df["vol_ratio"] = (vol_short + 1e-9) / (vol_long + 1e-9)

        # 3. EWMA Return Spread (multi-scale momentum, normalized by ATR)
        returns = _compute_log_returns(df["close"])
        ewma_12 = returns.ewm(span=12, min_periods=1).mean()
        ewma_48 = returns.ewm(span=48, min_periods=1).mean()
        atr = _compute_atr(df, 14)
        df["ewma_return_spread"] = (ewma_12 - ewma_48) / (atr + 1e-9)

        # 4. Volume Acceleration (current volume / 20-bar SMA)
        vol_sma = df["volume"].rolling(20, min_periods=1).mean()
        df["volume_accel"] = df["volume"] / (vol_sma + 1e-9)

        # 5. Lagged Log-Returns (3 lags: 1 bar, 5 bars, 12 bars)
        df["lag1_return"] = returns  # 1 bar (5m)
        df["lag5m_return"] = _compute_log_returns(df["close"].shift(1))  # Approx 5m lag
        df["lag1h_return"] = _compute_log_returns(df["close"].shift(12))  # 12 bars = 1h

        # 6. dYdX Funding-Rate Z-Score (14-day rolling z-score on DAILY funding, per god ruling)
        # Note: crypto_derivatives.ts is day-aligned (00:00:00Z); hourly ingestion = future T-016
        if funding_df is not None and not funding_df.empty:
            df = self._add_funding_z_score(df, funding_df)
        else:
            df["funding_zscore"] = np.nan

        # 7. Hour-of-day (cyclical encoding)
        hour = df["ts"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # 8. Day-of-week (cyclical encoding)
        dow = df["ts"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Select feature columns (drop intermediate columns)
        feature_cols = [
            "vwap_distance_1h",
            "vwap_distance_4h",
            "vol_ratio",
            "ewma_return_spread",
            "volume_accel",
            "lag1_return",
            "lag5m_return",
            "lag1h_return",
            "funding_zscore",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
        self.feature_names = feature_cols

        result = df[["ts"] + feature_cols].copy()
        return result

    def _add_funding_z_score(
        self, bars_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add funding-rate z-score feature (14-day rolling window on DAILY funding, per god ruling).

        Note: crypto_derivatives.ts is day-aligned (00:00:00Z); hourly ingestion = future T-016.
        Computes z-score of DAILY funding rates over 14-day window.

        Args:
            bars_df: 5-minute bars with ts column.
            funding_df: Daily funding rates with ts and funding_rate columns.

        Returns:
            bars_df with added 'funding_zscore' column.
        """
        bars_df = bars_df.copy()
        funding_df = funding_df.copy()

        bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True)
        funding_df["ts"] = pd.to_datetime(funding_df["ts"], utc=True)

        # Extract date from bars (for merging with daily funding)
        bars_df["date"] = bars_df["ts"].dt.date
        funding_df["date"] = pd.to_datetime(funding_df["ts"]).dt.date

        # Merge bars to daily funding rates by date (left merge keeps all bars)
        merged = bars_df.merge(
            funding_df, on="date", how="left", suffixes=("", "_funding")
        )

        # Compute z-score over 14-day rolling window of DAILY funding rates
        # (groupby date to get one value per day, then compute rolling stats)
        daily_funding = merged.groupby("date")["funding_rate"].first().reset_index()
        daily_funding["funding_mean_14d"] = (
            daily_funding["funding_rate"].rolling(14, min_periods=1).mean()
        )
        daily_funding["funding_std_14d"] = (
            daily_funding["funding_rate"].rolling(14, min_periods=1).std()
        )
        daily_funding["funding_zscore"] = (
            daily_funding["funding_rate"] - daily_funding["funding_mean_14d"]
        ) / (daily_funding["funding_std_14d"] + 1e-9)

        # Merge z-scores back to bars (one value per date)
        merged_z = merged.merge(
            daily_funding[["date", "funding_zscore"]], on="date", how="left"
        )

        # Return bars with funding_zscore
        bars_df["funding_zscore"] = merged_z["funding_zscore"]
        return bars_df

    def fit_scaler(self, train_features: pd.DataFrame) -> None:
        """Fit StandardScaler on training window.

        Args:
            train_features: Feature DataFrame (columns after 'ts').
        """
        feature_cols = [c for c in train_features.columns if c != "ts"]
        self.scaler = StandardScaler()
        self.scaler.fit(train_features[feature_cols].fillna(0))

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to features.

        Args:
            features_df: Feature DataFrame.

        Returns:
            Scaled features.
        """
        if self.scaler is None:
            logger.warning("Scaler not fitted; returning unscaled features")
            return features_df.copy()

        result = features_df.copy()
        feature_cols = [c for c in result.columns if c != "ts"]
        result[feature_cols] = self.scaler.transform(result[feature_cols].fillna(0))
        return result


def fetch_bars_and_funding(
    session: Session, ticker: str, lookback_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch intraday bars and funding rates from database.

    Args:
        session: SQLModel session.
        ticker: Ticker symbol.
        lookback_days: Number of days to fetch.

    Returns:
        Tuple of (bars_df, funding_df).
    """
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=lookback_days)

    start_iso = (
        datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    end_iso = (
        datetime.combine(end_date, datetime.max.time(), tzinfo=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # Fetch bars
    bar_rows = session.exec(
        select(IntradayBarsHistory).where(
            (IntradayBarsHistory.ticker == ticker)
            & (IntradayBarsHistory.interval == "5m")
            & (IntradayBarsHistory.ts >= start_iso)
            & (IntradayBarsHistory.ts <= end_iso)
        )
    ).all()

    bars_list = [
        {
            "ts": row.ts,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in bar_rows
    ]
    bars_df = pd.DataFrame(bars_list) if bars_list else pd.DataFrame()

    # Fetch funding rates
    funding_rows = session.exec(
        select(CryptoDerivative).where(
            (CryptoDerivative.ticker == ticker)
            & (CryptoDerivative.ts >= start_iso)
            & (CryptoDerivative.ts <= end_iso)
            & (CryptoDerivative.funding_rate.isnot(None))
        )
    ).all()

    funding_list = [
        {"ts": row.ts, "funding_rate": row.funding_rate} for row in funding_rows
    ]
    funding_df = pd.DataFrame(funding_list) if funding_list else pd.DataFrame()

    return bars_df, funding_df
