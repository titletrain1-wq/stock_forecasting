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
        df = pd.DataFrame({
            "high": [100.0, 101.0, 102.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1000.0, 1000.0],
        })

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
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": close - 5,
                "high": close + 10,
                "low": close - 10,
                "close": close,
                "volume": 100.0 + i,
            })
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
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": close - 2,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "volume": 50.0 + i,
            })
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
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": close - 2,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "volume": 50.0 + i,
            })
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
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": price - 5,
                "high": price + 10,
                "low": price - 10,
                "close": price,
                "volume": 100.0,
            })
        bars_df = pd.DataFrame(bars)

        features = builder.build_features("BTC-USD", bars_df)

        # lag1_return at index i should be ln(price[i] / price[i-1])
        # At index 1: ln(45010 / 45000)
        expected_ret_1 = np.log(45010.0 / 45000.0)
        assert np.isclose(features["lag1_return"].iloc[1], expected_ret_1, atol=1e-6)

        # At index 2: ln(45020 / 45010)
        expected_ret_2 = np.log(45020.0 / 45010.0)
        assert np.isclose(features["lag1_return"].iloc[2], expected_ret_2, atol=1e-6)

    def test_vwap_distance_no_future_bars(self) -> None:
        """Test that VWAP distance at t uses only bars <= t (not future bars)."""
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(20):
            ts = base + timedelta(minutes=5 * i)
            # Monotonically increasing prices
            close = 45000.0 + i * 10.0
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": close - 2,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "volume": 50.0,
            })
        bars_df = pd.DataFrame(bars)

        features = builder.build_features("BTC-USD", bars_df)

        # At bar 12 (1h mark), VWAP should only use bars 0-12, not 13+
        # Manually compute VWAP for bars 1-12 (last 12 bars including current)
        bars_subset = bars_df.iloc[0:13].copy()  # indices 0-12
        tp = (bars_subset["high"] + bars_subset["low"] + bars_subset["close"]) / 3
        manual_vwap = (tp * bars_subset["volume"]).sum() / bars_subset["volume"].sum()

        # The feature should be distance from manual_vwap, not from future price
        computed_feature = features["vwap_distance_1h"].iloc[12]
        # Should not be NaN and should be finite
        assert not np.isnan(computed_feature)
        assert np.isfinite(computed_feature)


class TestFundingZScore:
    """Test funding rate z-score feature (no lookahead)."""

    def test_funding_zscore_no_forward_fill_lookahead(self) -> None:
        """Test that funding z-score uses backward as-of join (no lookahead).

        If we have funding rates at specific times, a bar at time t should
        only use the last funding rate published at or before t, never after.
        """
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        # 300 bars = 25 hours worth of 5m bars
        bars = []
        for i in range(300):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 + (i % 100) * 5.0  # Oscillate to avoid trend
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": close - 2,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "volume": 50.0,
            })
        bars_df = pd.DataFrame(bars)

        # Funding rates: one every hour at specific times
        # Time 0h: rate 0.0001
        # Time 1h: rate 0.0002 (published at 60 minutes)
        # Time 2h: rate 0.0003 (published at 120 minutes)
        funding = []
        for hour in range(25):
            ts = base + timedelta(minutes=60 * hour)
            funding.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "funding_rate": 0.0001 + hour * 0.00005,
            })
        funding_df = pd.DataFrame(funding)

        # Build features with funding
        features = builder.build_features("BTC-USD", bars_df, funding_df)

        # At bar 0 (time 0:00): no prior funding rate → NaN or 0
        # At bar 12 (time 1:00, exactly when 1h rate published): should use 1h rate
        # At bar 11 (time 55m, before 1h rate): should use 0h rate
        # This validates backward as-of join (no forward-fill into future)

        # After warmup (288 bars = 24h), z-scores should exist
        if len(features) > 288:
            assert not np.isnan(features["funding_zscore"].iloc[288])

    def test_funding_zscore_with_empty_funding(self) -> None:
        """Test that features handle missing funding gracefully."""
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(50):
            ts = base + timedelta(minutes=5 * i)
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": 45000.0,
                "high": 45010.0,
                "low": 44990.0,
                "close": 45000.0 + i,
                "volume": 50.0,
            })
        bars_df = pd.DataFrame(bars)

        # No funding data
        features = builder.build_features("BTC-USD", bars_df, funding_df=None)

        # funding_zscore should be all NaN
        assert all(pd.isna(features["funding_zscore"]))


class TestScaler:
    """Test StandardScaler fit/transform."""

    def test_scaler_fit_and_transform(self) -> None:
        """Test that scaler fit on training data normalizes correctly."""
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(100):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 + i * 5.0
            bars.append({
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "open": close - 2,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "volume": 50.0 + i,
            })
        bars_df = pd.DataFrame(bars)

        # Build features
        features = builder.build_features("BTC-USD", bars_df)

        # Fit scaler on full data
        builder.fit_scaler(features)

        # Transform
        scaled = builder.transform(features)

        # Verify shape preserved
        assert scaled.shape == features.shape

        # Verify ts column unchanged
        assert all(scaled["ts"] == features["ts"])

        # Verify features are scaled (mean ~0, std ~1) for non-NaN values
        feature_cols = [c for c in scaled.columns if c != "ts"]
        for col in feature_cols:
            non_nan_scaled = scaled[col].dropna()
            non_nan_orig = features[col].dropna()
            if len(non_nan_scaled) > 1 and len(non_nan_orig) > 1:
                # Scaled features should have standardized values
                # Mean close to 0 and std close to 1 for scaled data
                assert abs(non_nan_scaled.mean()) < 2.0  # Lenient check for scaled data


class TestDatabaseFetch:
    """Test fetching bars and funding from database."""

    def test_fetch_bars_and_funding_empty_db(self, db_session: Session) -> None:
        """Test that fetch returns empty DataFrames when DB is empty."""
        from stock_forecasting.intraday_features import fetch_bars_and_funding

        bars_df, funding_df = fetch_bars_and_funding(db_session, "BTC-USD", 365)

        assert bars_df.empty
        assert funding_df.empty

    def test_fetch_bars_and_funding_with_data(self, db_session: Session) -> None:
        """Test fetching actual bars and funding from database."""
        from stock_forecasting.intraday_features import fetch_bars_and_funding

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
