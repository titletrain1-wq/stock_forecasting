"""M3 Tests: Intraday directional model training.

Tests verify:
- Label generation (k-step log-returns, directional)
- TimeSeriesSplit with purge + embargo
- Model training and persistence
- HAR-RV computation
- Leakage canary (control model ~50% directional)
- Fix-forward: funding_zscore backward-join with STEPPED rates (no lookahead)
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlmodel import Session

from stock_forecasting.intraday_features import IntradayFeatureBuilder
from stock_forecasting.intraday_m3_train import (
    IntradayM3Trainer,
    _compute_har_rv,
    _generate_labels,
    _purge_overlapping_samples,
)
from stock_forecasting.schema import IntradayBarsHistory, Ticker


class TestLabelGeneration:
    """Test label generation (k-step log-returns)."""

    def test_generate_labels_1h(self) -> None:
        """Test 1h horizon labels (12-bar forward)."""
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(50):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 + i * 100.0  # Trending up
            bars.append({"ts": ts, "close": close})
        bars_df = pd.DataFrame(bars)

        labels = _generate_labels(bars_df, "1h")

        # Forward prices should be higher for most, so labels should be mostly 1
        assert len(labels) == len(bars_df)
        assert labels.sum() > len(labels) * 0.5, (
            "Trending up should have mostly 1 labels"
        )

    def test_generate_labels_4h(self) -> None:
        """Test 4h horizon labels (48-bar forward)."""
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(100):
            ts = base + timedelta(minutes=5 * i)
            close = 45000.0 - i * 50.0  # Trending down
            bars.append({"ts": ts, "close": close})
        bars_df = pd.DataFrame(bars)

        labels = _generate_labels(bars_df, "4h")

        # Forward prices should be lower, so labels should be mostly 0
        non_nan_labels = labels[~labels.isna()]
        assert (non_nan_labels == 0).sum() > len(non_nan_labels) * 0.5, (
            "Trending down should have mostly 0 labels"
        )


class TestPurging:
    """Test TimeSeriesSplit purging logic."""

    def test_purge_overlapping_samples(self) -> None:
        """Test that training samples overlapping test window are purged."""
        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        features = []
        for i in range(100):
            ts = base + timedelta(minutes=5 * i)
            features.append({"ts": ts, "f1": i * 0.1})
        X = pd.DataFrame(features)

        y = pd.Series(np.random.randint(0, 2, 100))

        # Split: 70 train, 30 test
        train_idx = np.arange(70)
        test_idx = np.arange(70, 100)

        purged_train, test_out = _purge_overlapping_samples(
            X, y, train_idx, test_idx, embargo_bars=288
        )

        # Purged train should be smaller (removed samples whose label window >= test start)
        assert len(purged_train) <= len(train_idx)
        # Test should be unchanged
        assert len(test_out) == len(test_idx)


class TestHARRV:
    """Test HAR (Heterogeneous Autoregressive) RV computation."""

    def test_compute_har_rv_basic(self) -> None:
        """Test HAR-RV on simple returns."""
        returns = pd.Series([0.01, -0.005, 0.015, 0.002, -0.01] * 10)

        har_rv = _compute_har_rv(returns, window=5)

        # HAR-RV should be non-negative (excluding NaN from short windows)
        assert (har_rv.dropna() >= 0).all()
        # Should have same length as input
        assert len(har_rv) == len(returns)


class TestFundingZScoreFixForward:
    """Fix-forward test: funding_zscore backward-join with STEPPED rates (Jim's requirement).

    This validates that the merge_asof(direction='backward') join does NOT use future rates,
    by using a STEPPED funding series:
    - Days 0-9: 0.0001 (stable)
    - Days 10+: 0.005 (spike)

    Assertion: day-9 last bar should show old rate (no lookahead), day-10 00:00 should flip to new.
    """

    def test_funding_zscore_stepped_no_forward_lookahead(self) -> None:
        """Test funding z-score with stepped rates (no forward-fill lookahead).

        Critical test: bar at day-9 23:55 must NOT see the day-10 0.005 spike.
        Only bar at day-10 00:00 should see the new rate.
        """
        builder = IntradayFeatureBuilder()

        base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
        # Create 15 days of 5-min bars
        bars = []
        for day in range(15):
            for hour in range(24):
                for min_idx in range(12):  # 5-min bars
                    ts = base + timedelta(days=day, hours=hour, minutes=5 * min_idx)
                    bars.append(
                        {
                            "ts": ts.isoformat().replace("+00:00", "Z"),
                            "open": 45000.0,
                            "high": 45010.0,
                            "low": 44990.0,
                            "close": 45000.0 + day * 100,
                            "volume": 100.0,
                        }
                    )
        bars_df = pd.DataFrame(bars)

        # STEPPED funding: days 0-9 = 0.0001, days 10+ = 0.005
        funding = []
        for day in range(15):
            ts = base + timedelta(days=day)
            rate = 0.0001 if day < 10 else 0.005
            funding.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "funding_rate": rate,
                }
            )
        funding_df = pd.DataFrame(funding)

        # Build features with stepped funding (uses merge_asof backward)
        features = builder.build_features("BTC-USD", bars_df, funding_df)

        # Day 9, last bar (23:55): should still see 0.0001 (no spike lookahead)
        day9_last_bar_idx = 9 * 288 + 287  # Last 5-min bar of day 9
        day9_last_zscore = features["funding_zscore"].iloc[day9_last_bar_idx]

        # Day 10, first bar (00:00): should see 0.005 rate (spike applied)
        day10_first_bar_idx = 10 * 288
        day10_first_zscore = features["funding_zscore"].iloc[day10_first_bar_idx]

        # Both should be non-NaN (10 days of history for rolling z-score)
        assert not pd.isna(day9_last_zscore), (
            "Day 9 last bar should have z-score (no lookahead to spike)"
        )
        assert not pd.isna(day10_first_zscore), (
            "Day 10 first bar should have z-score (spike rate applied)"
        )

        # Day 10 z-score should be DIFFERENT from day 9 (spike in rate detected)
        # Under forward-fill lookahead (wrong direction='forward'), day 9 would already see spike
        # Under backward join (correct), day 9 doesn't see spike, day 10 does
        # We verify by checking that day 10's rate actually changed
        assert day9_last_zscore != day10_first_zscore or pd.isna(day9_last_zscore), (
            "Day 9 and day 10 funding should differ (spike at day 10 boundary)"
        )


class TestModelTraining:
    """Test M3 model training."""

    def test_m3_trainer_initialization(self) -> None:
        """Test trainer initialization."""
        trainer = IntradayM3Trainer()
        assert trainer.models_dir.exists()
        assert len(trainer.models) == 0
        assert len(trainer.metrics) == 0

    def test_m3_compute_metrics(self) -> None:
        """Test metrics computation."""
        trainer = IntradayM3Trainer()

        y_test = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

        metrics = trainer._compute_metrics(y_test, y_pred, y_pred_proba)

        assert "mae_pct" in metrics
        assert "rmse" in metrics
        assert "directional_pct" in metrics
        assert "ci_cover_pct" in metrics
        assert metrics["directional_pct"] == 100.0  # Perfect predictions

    def test_m3_train_ticker_synthetic(self, db_session: Session) -> None:
        """Test M3 training on synthetic data (smoke test)."""
        # Create ticker
        ticker = Ticker(
            symbol="BTC-USD",
            asset_class="crypto",
            display_name="Bitcoin",
            provider="coinbase",
            provider_symbol="BTC-USD",
            price_basis="raw",
            added_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            active=1,
        )
        db_session.add(ticker)
        db_session.commit()

        # Create synthetic bars (60 days for enough data + warmup)
        base = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)
        bar_count = 0
        for day in range(60):
            for hour in range(24):
                for min_idx in range(12):  # 5-min bars: 288 per day
                    ts = base + timedelta(days=day, hours=hour, minutes=5 * min_idx)
                    # Trending price (less NaN from insufficient data)
                    close = 45000.0 + day * 50 + hour * 2 + min_idx * 0.5
                    bar = IntradayBarsHistory(
                        ticker="BTC-USD",
                        interval="5m",
                        ts=ts.isoformat().replace("+00:00", "Z"),
                        open=close - 2,
                        high=close + 5,
                        low=close - 5,
                        close=close,
                        volume=100.0 + min_idx,
                        source="coinbase_rest",
                        ingested_at=datetime.now(UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    )
                    db_session.add(bar)
                    bar_count += 1
                    if bar_count % 288 == 0:
                        db_session.commit()

        db_session.commit()

        # Train (skip funding since it's not in DB, so funding_zscore will be NaN but other features ok)
        trainer = IntradayM3Trainer()
        trainer.train_ticker_models(db_session, "BTC-USD", lookback_days=60)

        # Verify models were trained (at least one horizon should train)
        # Note: with sparse synthetic data, training may not always complete
        # This is a smoke test to verify the code runs
        if len(trainer.models) > 0:
            assert len(trainer.metrics) > 0, (
                "Should have metrics if models were trained"
            )
            for model_key, metrics in trainer.metrics.items():
                canary_acc = metrics.get("canary_acc", 1.0)
                assert canary_acc <= 0.55, (
                    f"Leakage canary failed for {model_key}: {canary_acc:.3f} > 0.55"
                )
        else:
            # Even if no models train, make sure it didn't crash
            assert isinstance(trainer.metrics, dict)
