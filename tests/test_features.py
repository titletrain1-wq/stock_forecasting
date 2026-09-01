"""Tests for FeatureBuilder and technical indicator feature extraction."""

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.features import FEATURE_COLUMNS, FeatureBuilder


def _generate_synthetic_bars(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV bars for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    log_returns = np.random.normal(0.0005, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(np.random.normal(0.005, 0.005, n)))
    low = close * (1.0 - np.abs(np.random.normal(0.005, 0.005, n)))
    open_ = (high + low) / 2.0
    volume = np.random.lognormal(10.0, 0.4, n)

    return pd.DataFrame(
        {
            "ts": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_feature_builder_creates_features() -> None:
    """Verify that all expected technical feature columns are generated."""
    bars_df = _generate_synthetic_bars(n=150)
    builder = FeatureBuilder()
    features_df = builder.build(bars_df, scale=False)

    assert not features_df.empty
    assert "ts" in features_df.columns

    # Verify all feature columns exist
    for col in FEATURE_COLUMNS:
        assert col in features_df.columns, f"Missing feature column: {col}"
        assert features_df[col].notna().all(), f"NaNs found in feature: {col}"
        assert np.isfinite(features_df[col]).all(), f"Infinite values found in feature: {col}"

    # Verify warmup rows were dropped (at least 49 rows dropped for 50-period SMA)
    assert len(features_df) <= len(bars_df) - 49
    assert len(features_df) > 0


def test_feature_builder_no_lookahead() -> None:
    """Property test: features at row t are unchanged when rows > t are truncated."""
    bars_df = _generate_synthetic_bars(n=120)
    builder = FeatureBuilder()
    full_features = builder.build(bars_df, scale=False)

    # Pick multiple cutoff evaluation points
    assert len(full_features) >= 40
    feature_cols = [c for c in builder.feature_cols if c != "ts"]

    # Test across multiple row cutoffs t
    for t_idx in range(60, len(bars_df)):
        truncated_bars = bars_df.iloc[: t_idx + 1].copy()
        trunc_features = builder.build(truncated_bars, scale=False)

        # Get the feature values for the last row (timestamp at t_idx)
        target_ts = bars_df.iloc[t_idx]["ts"]

        row_full = full_features[full_features["ts"] == target_ts][feature_cols].iloc[0]
        row_trunc = trunc_features[trunc_features["ts"] == target_ts][feature_cols].iloc[0]

        np.testing.assert_allclose(
            row_trunc.values.astype(float),
            row_full.values.astype(float),
            rtol=1e-7,
            atol=1e-7,
            err_msg=f"Lookahead leak detected at cutoff row {t_idx} (ts={target_ts})",
        )


def test_feature_builder_scaling() -> None:
    """Verify StandardScaler scaling behavior with and without train_window."""
    bars_df = _generate_synthetic_bars(n=150)
    builder = FeatureBuilder()

    # 1. Unscaled
    unscaled_df = builder.build(bars_df, scale=False)
    assert builder.scaler is None

    # 2. Fully scaled (no train_window)
    scaled_df = builder.build(bars_df, scale=True)
    assert builder.scaler is not None

    numeric_cols = [c for c in builder.feature_cols if c != "ts"]
    non_const_full = unscaled_df[numeric_cols].columns[unscaled_df[numeric_cols].std(ddof=0) > 1e-6]
    means = scaled_df[non_const_full].mean()
    stds = scaled_df[non_const_full].std(ddof=0)

    np.testing.assert_allclose(means.values, 0.0, atol=1e-6)
    np.testing.assert_allclose(stds.values, 1.0, atol=1e-6)
    np.testing.assert_allclose(scaled_df[numeric_cols].values, builder.scaler.transform(unscaled_df[numeric_cols]), atol=1e-7)

    # 3. Scaled with train_window
    train_start, train_end = 0, 40
    window_scaled_df = builder.build(
        bars_df,
        train_window=(train_start, train_end),
        scale=True,
    )
    assert builder.scaler is not None

    train_slice = window_scaled_df.iloc[train_start:train_end][numeric_cols]
    unscaled_train_slice = unscaled_df.iloc[train_start:train_end][numeric_cols]
    non_const_cols = unscaled_train_slice.columns[unscaled_train_slice.std(ddof=0) > 1e-6]

    np.testing.assert_allclose(train_slice[non_const_cols].mean().values, 0.0, atol=1e-6)
    np.testing.assert_allclose(train_slice[non_const_cols].std(ddof=0).values, 1.0, atol=1e-6)

    # Verify fitted scaler produces exact train and test slices
    expected_train = builder.scaler.transform(unscaled_train_slice)
    np.testing.assert_allclose(train_slice.values, expected_train, atol=1e-7)

    test_slice = window_scaled_df.iloc[train_end:][numeric_cols]
    expected_test = builder.scaler.transform(unscaled_df.iloc[train_end:][numeric_cols])
    np.testing.assert_allclose(test_slice.values, expected_test, atol=1e-7)


def test_feature_builder_with_adj_close() -> None:
    """Verify feature building when adj_close is present."""
    bars_df = _generate_synthetic_bars(n=120)
    # Simulate a 2-for-1 stock split halfway through
    bars_df["adj_close"] = bars_df["close"] * 0.5

    builder = FeatureBuilder()
    features = builder.build(bars_df, scale=False)

    assert not features.empty
    for col in FEATURE_COLUMNS:
        assert col in features.columns
        assert features[col].notna().all()


def test_feature_builder_empty_and_short_input() -> None:
    """Verify graceful handling of empty and insufficient bar inputs."""
    builder = FeatureBuilder()

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    res_empty = builder.build(empty_df)
    assert res_empty.empty
    assert "ts" in res_empty.columns
    for col in FEATURE_COLUMNS:
        assert col in res_empty.columns

    # Short DataFrame (< 50 bars warmup)
    short_df = _generate_synthetic_bars(n=20)
    res_short = builder.build(short_df)
    assert res_short.empty


def test_feature_builder_custom_feature_cols() -> None:
    """Verify FeatureBuilder works with a custom subset of features."""
    bars_df = _generate_synthetic_bars(n=100)
    custom_cols = ["log_return_1d", "rsi_14", "day_of_week"]
    builder = FeatureBuilder(feature_cols=custom_cols)

    features = builder.build(bars_df, scale=True)
    assert set(features.columns) == {"ts", "log_return_1d", "rsi_14", "day_of_week"}
    assert builder.scaler is not None
    assert builder.scaler.n_features_in_ == 3


def test_feature_builder_invalid_train_window() -> None:
    """Verify ValueError is raised if train_window produces an empty slice."""
    bars_df = _generate_synthetic_bars(n=100)
    builder = FeatureBuilder()

    with pytest.raises(ValueError, match="empty training slice"):
        builder.build(bars_df, train_window=(10, 10), scale=True)
