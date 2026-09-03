"""Intraday trainer: backfill orchestration (data pipeline) and model training (M3)."""

import json
import logging
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from sqlmodel import Session, select

from stock_forecasting.config import get_settings
from stock_forecasting.database import get_session
from stock_forecasting.intraday_features import (
    IntradayFeatureBuilder,
    fetch_bars_and_funding,
)
from stock_forecasting.intraday_pipeline import (
    as_of_join_funding,
    fetch_funding_rates_from_db,
    fetch_intraday_bars_5m,
    filter_closed_bar_anchors,
)
from stock_forecasting.providers.coinbase import CoinbaseProvider
from stock_forecasting.schema import Ticker

logger = logging.getLogger(__name__)


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
                session.add(
                    Ticker(
                        symbol=ticker_str,
                        asset_class="crypto",
                        display_name=ticker_str,
                        provider="coinbase",
                        provider_symbol=ticker_str,
                        price_basis="raw",
                        added_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        active=1,
                    )
                )
                session.commit()

        # Initialize providers
        coinbase = CoinbaseProvider()

        for ticker_str in tickers:
            logger.info(
                f"Backfilling {ticker_str} intraday bars: {start_date} to {end_date}"
            )

            # Fetch 5-minute bars
            bars_list = fetch_intraday_bars_5m(
                coinbase, ticker_str, start_date, end_date
            )
            logger.info(f"  Fetched {len(bars_list)} 5m bars for {ticker_str}")

            if not bars_list:
                logger.warning(f"No bars fetched for {ticker_str}")
                continue

            # Convert to DataFrame
            import pandas as pd

            bars_df = pd.DataFrame(bars_list)
            bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True)

            # Fetch funding rates from crypto_derivatives table (dYdX ingestion)
            funding_df = fetch_funding_rates_from_db(
                session, ticker_str, start_date, end_date
            )
            logger.info(
                f"  Fetched {len(funding_df)} daily funding rates for {ticker_str}"
            )

            # Perform as-of join (validates no lookahead per F9)
            _, joined_df = as_of_join_funding(bars_df, funding_df)
            logger.info(f"  After as-of join: {len(joined_df)} bars")

            # Filter to closed-bar anchors
            anchored_df = filter_closed_bar_anchors(joined_df)
            logger.info(f"  After anchor filter: {len(anchored_df)} bars")

            # Write to database with INSERT OR IGNORE (dedup via unique constraint uq_intraday_bars_history)
            ingested_now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

            insert_sql = """
                INSERT OR IGNORE INTO intraday_bars_history (ticker, interval, ts, open, high, low, close, volume, source, ingested_at)
                VALUES (:ticker, :interval, :ts, :open, :high, :low, :close, :volume, :source, :ingested_at)
            """
            rows_to_insert = []
            for _, row in anchored_df.iterrows():
                ts_str = row["ts"].isoformat().replace("+00:00", "Z")
                rows_to_insert.append(
                    {
                        "ticker": ticker_str,
                        "interval": "5m",
                        "ts": ts_str,
                        "open": float(row["o"]),
                        "high": float(row["h"]),
                        "low": float(row["l"]),
                        "close": float(row["c"]),
                        "volume": float(row["v"]),
                        "source": "coinbase_rest",
                        "ingested_at": ingested_now,
                    }
                )

            # Execute batch insert with INSERT OR IGNORE
            connection = session.connection()
            connection.execute(text(insert_sql), rows_to_insert)
            session.commit()
            logger.info(
                f"  Wrote {len(anchored_df)} anchored bars to intraday_bars_history (INSERT OR IGNORE)"
            )

    except Exception:
        logger.exception("Backfill failed")
        session.rollback()
        raise


def train_intraday_models(
    session: Session | None = None,
    tickers: list[str] | None = None,
    lookback_days: int = 365,
    test_days: int = 14,
) -> None:
    """M3: Train intraday directional regressor models (crypto 1h/4h).

    Uses LGBMRegressor + Ridge Regressor for log-return prediction,
    with EWMA volatility bands fit on train split only.
    """
    if tickers is None:
        tickers = ["BTC-USD", "ETH-USD"]

    session = session or get_session().__enter__()
    models_dir = Path("stock_forecasting/models/intraday")
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}

    for ticker in tickers:
        logger.info(f"M3 training {ticker}")

        bars_df, funding_df = fetch_bars_and_funding(session, ticker, lookback_days)
        if len(bars_df) < 200:
            logger.warning(f"Insufficient data for {ticker}: {len(bars_df)} bars")
            continue

        bars_df = bars_df.sort_values("ts").reset_index(drop=True)
        bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True)

        # Build features
        builder = IntradayFeatureBuilder()
        features_df = builder.build_features(ticker, bars_df, funding_df)

        for horizon in ["1h", "4h"]:
            _train_regressor_model(
                ticker, horizon, features_df, bars_df, test_days, metadata, models_dir
            )

    # Write metadata
    metadata_path = models_dir / "metadata_crypto.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {metadata_path}")


def _train_regressor_model(
    ticker: str,
    horizon: str,
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    test_days: int,
    metadata: dict,
    models_dir: Path,
) -> None:
    """Train regressor + volatility model for one horizon."""
    k_bars = 12 if horizon == "1h" else 48

    # Filter to closed-bar anchors BEFORE label generation
    bars_anchored = filter_closed_bar_anchors(bars_df.copy())
    if len(bars_anchored) < 100:
        logger.warning(f"Insufficient anchors for {ticker}-{horizon}")
        return

    # Generate continuous log-return labels (regressor target)
    anchor_indices = bars_anchored.index.tolist()
    labels = []
    valid_anchor_idx = []

    for idx in anchor_indices:
        if idx + k_bars < len(bars_df):
            forward_close = bars_df["close"].iloc[idx + k_bars]
            anchor_close = bars_df["close"].iloc[idx]
            log_ret = np.log(forward_close / anchor_close)
            labels.append(log_ret)
            valid_anchor_idx.append(idx)

    if len(valid_anchor_idx) < 50:
        logger.warning(f"Insufficient valid anchors for {ticker}-{horizon}: {len(valid_anchor_idx)}")
        return

    # Align features to valid anchors
    X = features_df.iloc[valid_anchor_idx].drop(columns=["ts"]).copy()
    y = pd.Series(labels, index=range(len(labels)))

    # Clean NaN rows
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    assert len(features_df) == len(bars_df), "Feature-label alignment check"
    assert len(X) > 0, f"No clean training data for {ticker}-{horizon}"

    # 14-day holdout split with purge
    end_date = bars_df["ts"].max()
    test_start_date = end_date - timedelta(days=test_days)

    test_mask = (bars_df["ts"].iloc[valid_anchor_idx] >= test_start_date).values
    test_idx = np.where(test_mask)[0]
    train_idx_full = np.where(~test_mask)[0]

    # Purge training samples whose label window overlaps test window
    test_start_idx = test_idx[0] if len(test_idx) > 0 else len(X)
    purged_train_idx = []
    for idx in train_idx_full:
        label_window_end = idx + k_bars
        if label_window_end < test_start_idx:
            purged_train_idx.append(idx)

    if len(purged_train_idx) < 10 or len(test_idx) < 5:
        logger.warning(
            f"Split too small for {ticker}-{horizon}: train={len(purged_train_idx)}, test={len(test_idx)}"
        )
        return

    X_train = X.iloc[purged_train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[purged_train_idx]
    y_test = y.iloc[test_idx]

    logger.info(
        f"  {horizon}: train={len(X_train)}, test={len(X_test)}"
    )

    # Train regressor models
    lgb = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    lgb.fit(X_train, y_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Predictions on test set
    y_pred_lgb = lgb.predict(X_test)

    # Train EWMA volatility model on training set only
    train_residuals = y_train - lgb.predict(X_train)
    vol_model = _fit_ewma_volatility(train_residuals)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred_lgb)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    direction_acc = np.mean(np.sign(y_pred_lgb) == np.sign(y_test))

    # Volatility bands on test
    test_residuals = y_test - y_pred_lgb
    ci_lower, ci_upper = vol_model(np.abs(test_residuals))
    ci_cover = np.mean((y_test >= ci_lower) & (y_test <= ci_upper))

    # Save models
    ticker_clean = ticker.replace("-", "_").lower()
    lgb.booster_.save_model(str(models_dir / f"intraday_lgb_{ticker_clean}_{horizon}.pkl"))
    with open(models_dir / f"intraday_ridge_{ticker_clean}_fallback_{horizon}.pkl", "wb") as f:
        pickle.dump(ridge, f)
    with open(models_dir / f"intraday_har_rv_{ticker_clean}.pkl", "wb") as f:
        pickle.dump(vol_model, f)

    metadata[f"{ticker}-{horizon}"] = {
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_pct": float(direction_acc * 100),
        "ci_cover_pct": float(ci_cover * 100),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def _fit_ewma_volatility(residuals: pd.Series) -> callable:
    """Fit EWMA volatility model on residuals."""
    ewma_var = residuals.ewm(span=12, adjust=False).var()
    vol_scale = np.sqrt(ewma_var.mean())

    def vol_bounds(residual_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ci_width = 1.96 * vol_scale
        return -ci_width, ci_width

    return vol_bounds


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    backfill_intraday_bars()
    sys.exit(0)
