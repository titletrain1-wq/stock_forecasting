"""Feature engineering pipeline for stock and crypto price forecasting.

Builds normalized, lookahead-free technical indicator features from OHLCV bars.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: list[str] = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "sma20_stretch",
    "sma_crossover",
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "macd_hist_norm",
    "bb_pct_b",
    "bb_bandwidth",
    "norm_atr_14",
    "volume_ratio",
    "obv_10d_change",
    "day_of_week",
    "is_month_end",
    "is_quarter_end",
]


class FeatureBuilder:
    """Calculates technical indicators and features with strict zero lookahead."""

    def __init__(self, feature_cols: Sequence[str] | None = None) -> None:
        """Initialize FeatureBuilder with configured feature column names."""
        self.feature_cols: list[str] = (
            list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
        )
        self.scaler: StandardScaler | None = None

    def build(
        self,
        bars_df: pd.DataFrame,
        train_window: tuple[int, int] | None = None,
        scale: bool = True,
    ) -> pd.DataFrame:
        """Build technical indicator features from OHLCV bars.

        Args:
            bars_df: DataFrame with OHLCV data. Expected columns:
                     ['ts', 'open', 'high', 'low', 'close', 'volume']
                     (optional 'adj_close').
            train_window: Optional (start_idx, end_idx) tuple indicating slice
                          of rows for fitting StandardScaler.
            scale: Whether to scale feature columns with StandardScaler.

        Returns:
            DataFrame containing 'ts' and the computed feature columns,
            with initial warmup NaNs dropped.
        """
        if bars_df.empty:
            empty_cols = ["ts"] + [c for c in self.feature_cols if c != "ts"]
            return pd.DataFrame(columns=empty_cols)

        # Work on a shallow copy / references
        df = bars_df.copy()

        # Handle timestamp column / index
        if "ts" in df.columns:
            ts_series = pd.to_datetime(df["ts"], utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            ts_series = pd.to_datetime(df.index, utc=True)
        else:
            ts_series = pd.to_datetime(df.index, errors="coerce", utc=True)

        # Handle adjusted close vs raw close
        if "adj_close" in df.columns and df["adj_close"].notna().any():
            adj_close = df["adj_close"].fillna(df["close"]).astype(float)
            raw_close = df["close"].astype(float)
            adj_factor = (adj_close / raw_close).replace([np.inf, -np.inf], 1.0).fillna(1.0)
            close = adj_close
            high = df["high"].astype(float) * adj_factor
            low = df["low"].astype(float) * adj_factor
            volume = df["volume"].astype(float)
        else:
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)

        # 1. Log returns (1d, 5d, 20d)
        log_return_1d = np.log(close / close.shift(1))
        log_return_5d = np.log(close / close.shift(5))
        log_return_20d = np.log(close / close.shift(20))

        # 2. Moving average features
        sma20 = ta.sma(close, length=20)
        sma50 = ta.sma(close, length=50)
        sma20_stretch = (close / sma20) - 1.0 if sma20 is not None else pd.Series(np.nan, index=df.index)
        sma_crossover = (sma20 / sma50) - 1.0 if sma20 is not None and sma50 is not None else pd.Series(np.nan, index=df.index)

        # 3. Momentum indicators
        rsi_14 = ta.rsi(close, length=14)
        if rsi_14 is None:
            rsi_14 = pd.Series(np.nan, index=df.index)

        stoch = ta.stoch(high=high, low=low, close=close, k=14, d=3)
        if stoch is not None and not stoch.empty:
            k_cols = [c for c in stoch.columns if c.startswith("STOCHk_")]
            d_cols = [c for c in stoch.columns if c.startswith("STOCHd_")]
            stoch_k = stoch[k_cols[0]] if k_cols else pd.Series(np.nan, index=df.index)
            stoch_d = stoch[d_cols[0]] if d_cols else pd.Series(np.nan, index=df.index)
        else:
            stoch_k = pd.Series(np.nan, index=df.index)
            stoch_d = pd.Series(np.nan, index=df.index)

        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            h_cols = [c for c in macd.columns if c.startswith("MACDh_")]
            macd_hist = macd[h_cols[0]] if h_cols else pd.Series(np.nan, index=df.index)
            macd_hist_norm = macd_hist / close
        else:
            macd_hist_norm = pd.Series(np.nan, index=df.index)

        # 4. Volatility indicators
        bb = ta.bbands(close, length=20, std=2.0)
        if bb is not None and not bb.empty:
            bbp_cols = [c for c in bb.columns if c.startswith("BBP_")]
            bbb_cols = [c for c in bb.columns if c.startswith("BBB_")]
            bb_pct_b = bb[bbp_cols[0]] if bbp_cols else pd.Series(np.nan, index=df.index)
            bb_bandwidth = bb[bbb_cols[0]] if bbb_cols else pd.Series(np.nan, index=df.index)
        else:
            bb_pct_b = pd.Series(np.nan, index=df.index)
            bb_bandwidth = pd.Series(np.nan, index=df.index)

        atr_14 = ta.atr(high=high, low=low, close=close, length=14)
        if atr_14 is not None:
            norm_atr_14 = atr_14 / close
        else:
            norm_atr_14 = pd.Series(np.nan, index=df.index)

        # 5. Volume indicators
        vol_sma20 = ta.sma(volume, length=20)
        volume_ratio = volume / vol_sma20 if vol_sma20 is not None else pd.Series(np.nan, index=df.index)

        obv = ta.obv(close, volume)
        obv_10d_change = obv.diff(10) if obv is not None else pd.Series(np.nan, index=df.index)

        # 6. Calendar features
        day_of_week = ts_series.dt.dayofweek.astype(float)
        is_month_end = ts_series.dt.is_month_end.astype(float)
        is_quarter_end = ts_series.dt.is_quarter_end.astype(float)

        feature_dict: dict[str, pd.Series] = {
            "ts": ts_series,
            "log_return_1d": log_return_1d,
            "log_return_5d": log_return_5d,
            "log_return_20d": log_return_20d,
            "sma20_stretch": sma20_stretch,
            "sma_crossover": sma_crossover,
            "rsi_14": rsi_14,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "macd_hist_norm": macd_hist_norm,
            "bb_pct_b": bb_pct_b,
            "bb_bandwidth": bb_bandwidth,
            "norm_atr_14": norm_atr_14,
            "volume_ratio": volume_ratio,
            "obv_10d_change": obv_10d_change,
            "day_of_week": day_of_week,
            "is_month_end": is_month_end,
            "is_quarter_end": is_quarter_end,
        }

        # Filter dictionary to selected feature columns + 'ts'
        cols_to_include = ["ts"] + [col for col in self.feature_cols if col in feature_dict]
        features_df = pd.DataFrame({c: feature_dict[c] for c in cols_to_include}, index=df.index)

        # Drop warmup NaNs
        clean_df = features_df.dropna().copy()

        if clean_df.empty:
            return clean_df

        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            active_feature_cols = [c for c in self.feature_cols if c in clean_df.columns and c != "ts"]
            if train_window is not None:
                start, end = train_window
                train_slice = clean_df.iloc[start:end][active_feature_cols]
                if train_slice.empty:
                    raise ValueError(f"train_window {train_window} results in an empty training slice.")
                self.scaler.fit(train_slice)
            else:
                self.scaler.fit(clean_df[active_feature_cols])

            scaled_values = self.scaler.transform(clean_df[active_feature_cols])
            clean_df[active_feature_cols] = scaled_values
        else:
            self.scaler = None

        return clean_df
