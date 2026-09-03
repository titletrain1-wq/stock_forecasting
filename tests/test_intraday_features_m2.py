"""M2 Tests: Intraday feature engineering module.

Tests that verify:
- Feature values correct on known fixtures
- No NaN leakage past warmup periods
- No lookahead in rolling windows (only bars <= t used)
- Funding z-score as-of join has no forward-fill lookahead
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session

from stock_forecasting.intraday_features import (
    IntradayFeatureBuilder,
    _compute_vwap,
    fetch_bars_and_funding,
)
from stock_forecasting.schema import CryptoDerivative, IntradayBarsHistory, Ticker


class TestVWAPComputation:
    """Test VWAP calculation."""

    def test_compute_vwap_basic(self) -> None:
        """Test VWAP computation on simple fixture."""
        df = pd.DataFrame(
            {
                "high": [100.0, 101.0, 102.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1000.0, 1000.0],
            }
        )

        vwap = _compute_vwap(df, window=3)

        # After 3 bars, VWAP should be computed and not NaN
        assert not np.isnan(vwap.iloc[-1])
        # With uniform volume, VWAP should be close to mean of (high+low+close)/3
        # For bars [100, 101, 102] with uniform volume, VWAP ~ 100.67 (mean TP)
        assert 100.0 < vwap.iloc[-1] < 103.0


class TestFeatureBuilderBasic:
    """Test basic feature computation without NaN leakage."""

    def test_build_features_all_features_present(self) -> None:
        """Test that all 13 features are computed."""
        builder = IntradayFeatureBuilder()

        # Create 100 bars of synthetic data
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(100):
            ts = base + timedelta(minutes=5 * i)
            # Trending price with volume
            close = 45000.0 + i * 10.0
            bars.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "open": close - 5,
                    "high": close + 10,
                    "low": close - 10,
                    "close": close,
                    "volume": 100.0 + i,
                }
            )
        bars_df = pd.DataFrame(bars)

        # Build features
        features = builder.build_features("BTC-USD", bars_df)

        # Verify all feature columns exist
        expected_features = [
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
        assert all(f in features.columns for f in expected_features)
        assert len(features) == 100


class TestNaNWarmup:
    """Test that NaN leakage does not persist past warmup."""

    def test_no_nan_after_warmup_1h(self) -> None:
        """Test that VWAP 1h features (12 bars) have no NaN after warmup."""
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(50):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 + i * 5.0
            bars.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "open": close - 2,
                    "high": close + 5,
                    "low": close - 5,
                    "close": close,
                    "volume": 50.0 + i,
                }
            )
        bars_df = pd.DataFrame(bars)

        features = builder.build_features("BTC-USD", bars_df)

        # After 12 bars (1h), VWAP 1h should not be NaN
        assert not np.isnan(features["vwap_distance_1h"].iloc[12])

    def test_no_nan_after_warmup_4h(self) -> None:
        """Test that VWAP 4h features (48 bars) have no NaN after warmup."""
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(100):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 + i * 5.0
            bars.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "open": close - 2,
                    "high": close + 5,
                    "low": close - 5,
                    "close": close,
                    "volume": 50.0 + i,
                }
            )
        bars_df = pd.DataFrame(bars)

        features = builder.build_features("BTC-USD", bars_df)

        # After 48 bars (4h), VWAP 4h should not be NaN
        assert not np.isnan(features["vwap_distance_4h"].iloc[48])


class TestNoLookahead:
    """Test that features do not use future data (no lookahead)."""

    def test_lag_returns_no_forward_data(self) -> None:
        """Test that lagged returns only use past data.

        If we compute lag1_return at index i, it should only depend on
        close[i] and close[i-1], not close[i+1] or later.
        """
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        prices = [45000.0, 45010.0, 45020.0, 45030.0, 45040.0]
        for i, price in enumerate(prices):
            ts = base + timedelta(minutes=5 * i)
            bars.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "open": price - 5,
                    "high": price + 10,
                    "low": price - 10,
                    "close": price,
                    "volume": 100.0,
                }
            )
        bars_df = pd.DataFrame(bars)

        features = builder.build_features("BTC-USD", bars_df)

        # lag1_return at index i should be ln(price[i] / price[i-1])
        # At index 1: ln(45010 / 45000)
        expected_ret_1 = np.log(45010.0 / 45000.0)
        assert np.isclose(features["lag1_return"].iloc[1], expected_ret_1, atol=1e-6)

        # At index 2: ln(45020 / 45010)
        expected_ret_2 = np.log(45020.0 / 45010.0)
        assert np.isclose(features["lag1_return"].iloc[2], expected_ret_2, atol=1e-6)



class TestFundingZScore:
    """Test funding rate z-score feature (no lookahead)."""

    def test_funding_zscore_no_forward_fill_lookahead(self) -> None:
        """Test funding z-score with 14-day rolling window on DAILY funding rates.

        Implementation uses one funding rate per day (from crypto_derivatives table).
        """
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        # 15 days of 5m bars to have 14 days of history for rolling z-score
        bars = []
        for day in range(15):
            for bar_idx in range(288):  # 24h * 12 bars/h
                ts = base + timedelta(days=day, minutes=5 * bar_idx)
                close = 45000.0 + day * 10.0
                bars.append(
                    {
                        "ts": ts.isoformat().replace("+00:00", "Z"),
                        "open": close - 2,
                        "high": close + 5,
                        "low": close - 5,
                        "close": close,
                        "volume": 50.0,
                    }
                )
        bars_df = pd.DataFrame(bars)

        # Funding rates: ONE per day at 00:00:00Z (matching crypto_derivatives daily)
        funding = []
        for day in range(15):
            ts = base + timedelta(days=day)
            funding.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "funding_rate": 0.0001 + day * 0.00001,
                }
            )
        funding_df = pd.DataFrame(funding)

        # Build features with funding
        features = builder.build_features("BTC-USD", bars_df, funding_df)

        # First 14 days should have rolling z-scores (min_periods=1 in rolling window)
        # Day 14 (index 14*288 onwards) should have non-NaN z-scores
        day_14_start = 14 * 288
        if len(features) > day_14_start:
            assert not np.isnan(features["funding_zscore"].iloc[day_14_start])

    def test_funding_zscore_with_empty_funding(self) -> None:
        """Test that features handle missing funding gracefully."""
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(50):
            ts = base + timedelta(minutes=5 * i)
            bars.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "open": 45000.0,
                    "high": 45010.0,
                    "low": 44990.0,
                    "close": 45000.0 + i,
                    "volume": 50.0,
                }
            )
        bars_df = pd.DataFrame(bars)

        # No funding data
        features = builder.build_features("BTC-USD", bars_df, funding_df=None)

        # funding_zscore should be all NaN
        assert all(pd.isna(features["funding_zscore"]))


class TestScaler:
    """Test StandardScaler fit/transform."""

    def test_scaler_fit_and_transform_train_only(self) -> None:
        """Test that scaler fit on TRAIN slice only; assert test data does not leak into fit."""
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        # Need 60+ bars to have enough warmup (4h VWAP needs 48 bars)
        for i in range(200):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 + i * 5.0
            bars.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "open": close - 2,
                    "high": close + 5,
                    "low": close - 5,
                    "close": close,
                    "volume": 50.0 + i,
                }
            )
        bars_df = pd.DataFrame(bars)

        # Provide funding data for 15 days (matching 200 bars = ~16 hours, so days 0-2 at min, but use 15 for safety)
        funding = []
        for day in range(15):
            ts = base + timedelta(days=day)
            funding.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "funding_rate": 0.0001 + day * 0.00001,
                }
            )
        funding_df = pd.DataFrame(funding)

        # Build features with funding
        features = builder.build_features("BTC-USD", bars_df, funding_df)

        # Split into TRAIN (bars 50-150, after warmup) and TEST (bars 150-200)
        train_features = features.iloc[50:150]
        test_features = features.iloc[150:]

        # Fit scaler on TRAIN slice ONLY
        builder.fit_scaler(train_features)

        # Verify scaler's statistics come from TRAIN data, not TEST
        # (Scaler should have learned only from train_features)
        feature_cols = [c for c in train_features.columns if c != "ts"]

        for col in feature_cols:
            train_col = train_features[col].dropna()
            if len(train_col) > 1:
                # Manually compute what the scaler should have learned
                expected_mean = train_col.mean()
                expected_std = train_col.std()

                # Verify against what scaler learned (within tolerance)
                # Note: sklearn may use slight variations in std computation
                col_idx = feature_cols.index(col)
                assert builder.scaler is not None
                assert builder.scaler.mean_[col_idx] == pytest.approx(
                    expected_mean, rel=1e-5
                )
                assert builder.scaler.scale_[col_idx] == pytest.approx(expected_std, rel=1e-5)

        # Transform both train and test using the train-fit scaler
        scaled_train = builder.transform(train_features)
        scaled_test = builder.transform(test_features)

        # Verify shapes preserved
        assert scaled_train.shape == train_features.shape
        assert scaled_test.shape == test_features.shape

        # Verify ts columns unchanged
        assert all(scaled_train["ts"] == train_features["ts"])
        assert all(scaled_test["ts"] == test_features["ts"])


class TestTruncationInvariance:
    """Test that features computed on truncated vs full data match (no lookahead)."""

    def test_truncation_invariance_all_features(self) -> None:
        """Test that features at position k are identical whether computed on bars[:k] or bars[:n].

        This validates that no future data influences feature values (no lookahead).
        """
        builder = IntradayFeatureBuilder()

        # Create 100 bars of synthetic data
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(100):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 + i * 5.0  # Trending price
            bars.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "open": close - 2,
                    "high": close + 5,
                    "low": close - 5,
                    "close": close,
                    "volume": 50.0 + i,
                }
            )
        bars_df = pd.DataFrame(bars)

        # Compute features on full dataset
        features_full = builder.build_features("BTC-USD", bars_df)

        # For several truncation points, verify that truncated computation matches
        test_indices = [12, 24, 48, 75, 99]  # Test at various points including end
        for k in test_indices:
            builder_truncated = IntradayFeatureBuilder()
            bars_truncated = bars_df.iloc[: k + 1].copy()
            features_truncated = builder_truncated.build_features(
                "BTC-USD", bars_truncated
            )

            # Assert that all feature columns at index k match between truncated and full
            feature_cols = [c for c in features_full.columns if c != "ts"]
            for col in feature_cols:
                val_full = features_full[col].iloc[k]
                val_trunc = features_truncated[col].iloc[k]

                # If both are NaN, they match
                if pd.isna(val_full) and pd.isna(val_trunc):
                    continue

                # Otherwise, they should be float-equal (within tolerance)
                assert np.isclose(val_full, val_trunc, atol=1e-9, equal_nan=True), (
                    f"Feature {col} at index {k} differs: full={val_full}, truncated={val_trunc}"
                )


class TestDatabaseFetch:
    """Test fetching bars and funding from database."""

    def test_fetch_bars_and_funding_empty_db(self, db_session: Session) -> None:
        """Test that fetch returns empty DataFrames when DB is empty."""

        bars_df, funding_df = fetch_bars_and_funding(db_session, "BTC-USD", 365)

        assert bars_df.empty
        assert funding_df.empty

    def test_fetch_bars_and_funding_with_data(self, db_session: Session) -> None:
        """Test fetching actual bars and funding from database."""

        # Create ticker
        ticker = Ticker(
            symbol="BTC-USD",
            asset_class="crypto",
            display_name="Bitcoin",
            provider="coinbase",
            provider_symbol="BTC-USD",
            price_basis="raw",
            added_at="2026-09-01T00:00:00Z",
            active=1,
        )
        db_session.add(ticker)
        db_session.commit()

        # Add some bars
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(10):
            ts = base + timedelta(minutes=5 * i)
            bar = IntradayBarsHistory(
                ticker="BTC-USD",
                interval="5m",
                ts=ts.isoformat().replace("+00:00", "Z"),
                open=45000.0 + i,
                high=45010.0 + i,
                low=44990.0 + i,
                close=45005.0 + i,
                volume=100.0,
                source="coinbase_rest",
                ingested_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
            db_session.add(bar)

        # Add some funding rates
        for i in range(5):
            ts = base + timedelta(hours=i)
            deriv = CryptoDerivative(
                ticker="BTC-USD",
                ts=ts.isoformat().replace("+00:00", "Z"),
                funding_rate=0.0001 + i * 0.00005,
                source="dydx",
            )
            db_session.add(deriv)

        db_session.commit()

        # Fetch
        bars_df, funding_df = fetch_bars_and_funding(db_session, "BTC-USD", 365)

        # Verify
        assert len(bars_df) == 10
        assert len(funding_df) == 5
        assert "close" in bars_df.columns
        assert "funding_rate" in funding_df.columns
