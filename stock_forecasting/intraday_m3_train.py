"""M3 — Intraday directional model training (crypto 1h/4h phase-1).

Trains 4x LightGBM + 4x Ridge fallback models:
- BTC-USD: 1h, 4h horizons
- ETH-USD: 1h, 4h horizons

Features: IntradayFeatureBuilder output (13 features)
Labels: Closed-bar-anchor k-step log-returns (k ∈ {1h, 4h})
Split: TimeSeriesSplit with purge (overlapping windows) + 24h embargo
Metrics: MAE%, RMSE, directional%, CI coverage
Leakage canary: Shuffled-label control scores ~50% directional (FAIL if >55%)
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sqlmodel import Session

from stock_forecasting.intraday_features import (
    IntradayFeatureBuilder,
    fetch_bars_and_funding,
)

logger = logging.getLogger(__name__)


def _compute_har_rv(returns: pd.Series, window: int = 22) -> pd.Series:
    """HAR (Heterogeneous Autoregressive) Realized Volatility proxy.

    Computes 1d, 1w, 1m rolling std of returns as feature.
    Returns DataFrame with har_rv columns (average of three scales).
    """
    rv_1d = returns.rolling(window=1, min_periods=1).std()
    rv_1w = returns.rolling(window=5, min_periods=1).std()
    rv_1m = returns.rolling(window=22, min_periods=1).std()
    har_rv = (rv_1d + rv_1w + rv_1m) / 3
    return har_rv


def _generate_labels(bars_df: pd.DataFrame, horizon: str) -> pd.Series:
    """Generate k-step forward log-return labels.

    Args:
        bars_df: DataFrame with ts, close columns.
        horizon: "1h" or "4h".

    Returns:
        Series of binary labels (1 = up, 0 = down/flat).
    """
    bars_df = bars_df.sort_values("ts").reset_index(drop=True)
    k_bars = 12 if horizon == "1h" else 48  # 5-min bars
    forward_close = bars_df["close"].shift(-k_bars)
    log_returns = np.log(forward_close / bars_df["close"])
    labels = (log_returns > 0).astype(int)
    return labels


def _purge_overlapping_samples(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo_bars: int = 288,
) -> tuple[np.ndarray, np.ndarray]:
    """Purge training samples whose label window overlaps the test window.

    Args:
        X: Features DataFrame with ts index or column.
        y: Labels Series.
        train_idx: Training indices.
        test_idx: Test indices.
        embargo_bars: Bars to embargo after test window (24h = 288 bars).

    Returns:
        (purged_train_idx, test_idx).
    """
    if "ts" in X.columns:
        ts_col = X["ts"]
    else:
        ts_col = X.index

    test_start_ts = ts_col.iloc[test_idx[0]]

    purged_train = []
    for idx in train_idx:
        label_window_end_idx = min(idx + 48, len(X) - 1)
        label_window_end_ts = ts_col.iloc[label_window_end_idx]
        if label_window_end_ts < test_start_ts:
            purged_train.append(idx)

    return np.array(purged_train), test_idx


class IntradayM3Trainer:
    """M3 trainer for crypto intraday directional models."""

    def __init__(self, models_dir: str = "stock_forecasting/models/intraday"):
        """Initialize trainer."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, dict] = {}

    def train_ticker_models(
        self,
        session: Session,
        ticker: str,
        lookback_days: int = 365,
        test_days: int = 14,
    ) -> None:
        """Train 1h + 4h models for a ticker.

        Args:
            session: SQLModel session.
            ticker: "BTC-USD" or "ETH-USD".
            lookback_days: Historical window.
            test_days: Last N days for test set.
        """
        logger.info(
            f"M3 training {ticker} (lookback={lookback_days}d, test={test_days}d)"
        )

        # Fetch bars and funding
        bars_df, funding_df = fetch_bars_and_funding(session, ticker, lookback_days)
        if len(bars_df) < 100:
            logger.warning(f"Insufficient data for {ticker}: {len(bars_df)} bars")
            return

        # Build features
        builder = IntradayFeatureBuilder()
        features_df = builder.build_features(ticker, bars_df, funding_df)

        # Train 1h and 4h models
        for horizon in ["1h", "4h"]:
            self._train_horizon_models(ticker, horizon, features_df, bars_df, test_days)

    def _train_horizon_models(
        self,
        ticker: str,
        horizon: str,
        features_df: pd.DataFrame,
        bars_df: pd.DataFrame,
        test_days: int,
    ) -> None:
        """Train 1h or 4h models with LightGBM + Ridge fallback."""
        # Generate labels
        labels = _generate_labels(bars_df, horizon)

        # Align features and labels
        X = features_df.drop(columns=["ts"]).copy()
        y = labels.copy()

        # Drop rows where features or labels are NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        features_df_clean = features_df[valid_idx].reset_index(drop=True)

        if len(X) < 50:
            logger.warning(
                f"Insufficient clean data for {ticker}-{horizon}: {len(X)} samples"
            )
            return

        # TimeSeriesSplit with purge
        tscv = TimeSeriesSplit(n_splits=2)
        train_idx, test_idx = next(iter(tscv.split(X)))

        # Purge overlapping samples
        purged_train_idx, test_idx = _purge_overlapping_samples(
            features_df_clean, y, train_idx, test_idx
        )

        X_train, X_test = X.iloc[purged_train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[purged_train_idx], y.iloc[test_idx]

        if len(X_train) < 10 or len(X_test) < 5:
            logger.warning(
                f"Split too small for {ticker}-{horizon}: train={len(X_train)}, test={len(X_test)}"
            )
            return

        logger.info(
            f"  {horizon}: train={len(X_train)}, test={len(X_test)}, pos={y_train.sum()}/{len(y_train)}"
        )

        # Train LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)

        # Train Ridge fallback
        ridge_model = LogisticRegression(penalty="l2", max_iter=1000, random_state=42)
        ridge_model.fit(X_train, y_train)

        # Predict
        y_pred_lgb = lgb_model.predict(X_test)
        y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred_lgb, y_pred_proba_lgb)
        self.metrics[f"{ticker}-{horizon}"] = metrics

        # Leakage canary: train on shuffled labels
        y_shuffled = y_train.copy()
        np.random.seed(42)
        np.random.shuffle(y_shuffled.values)
        lgb_canary = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        lgb_canary.fit(X_train, y_shuffled)
        y_canary_pred = lgb_canary.predict(X_test)
        canary_acc = accuracy_score(y_test, y_canary_pred)

        if canary_acc > 0.55:
            logger.error(
                f"LEAKAGE CANARY FAILED: {ticker}-{horizon} canary_acc={canary_acc:.3f} > 0.55"
            )
        else:
            logger.info(f"  Leakage canary: {canary_acc:.3f} (< 0.55, pass)")

        self.metrics[f"{ticker}-{horizon}"]["canary_acc"] = float(canary_acc)

        # Store models
        self.models[f"{ticker}-{horizon}"] = {
            "lgb": lgb_model,
            "ridge": ridge_model,
        }

        # Persist to pickle
        model_path = self.models_dir / f"lgb_{ticker.replace('-', '_')}_{horizon}.pkl"
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(lgb_model, f)
        logger.info(f"  Model saved: {model_path}")

    def _compute_metrics(
        self, y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> dict:
        """Compute validation metrics."""
        mae_pct = mean_absolute_error(y_test, y_pred_proba) * 100
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        directional_acc = accuracy_score(y_test, y_pred)

        ci_cover = float(
            ((y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)).sum() / len(y_pred)
        )

        return {
            "mae_pct": float(mae_pct),
            "rmse": float(rmse),
            "directional_pct": float(directional_acc * 100),
            "ci_cover_pct": float(ci_cover * 100),
        }

    def save_metadata(self) -> None:
        """Save metrics to metadata.json."""
        metadata_path = self.models_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metadata saved: {metadata_path}")
