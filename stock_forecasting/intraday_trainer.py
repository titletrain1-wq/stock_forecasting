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
from sklearn.preprocessing import StandardScaler
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


CODE_VERSION = "m3-1"

_HORIZON_BARS = {"1h": 12, "4h": 48}  # 5-minute bars per horizon
_EMBARGO_BARS = 288  # 24h of 5-minute bars, applied after the test window


class ResidualVolModel:
    """Empirical 95% predictive-interval model for intraday log-return forecasts.

    Fit on the TRAIN split only: stores the 2.5%/97.5% quantiles of the model's
    training residuals plus a reference realised-vol level. At inference the band
    is ``pred + [q_lo, q_hi] * scale`` where ``scale`` tracks how the current
    realised vol compares to the training median (clipped to keep it sane), so the
    interval actually widens/narrows with the market rather than being constant.
    Picklable (plain attributes, no closure).
    """

    def __init__(
        self, q_lo: float, q_hi: float, ref_vol: float, code_version: str
    ) -> None:
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)
        self.ref_vol = float(ref_vol)
        self.code_version = code_version

    def _scale(self, cond_vol: np.ndarray | float) -> np.ndarray | float:
        if self.ref_vol <= 0:
            return 1.0
        return np.clip(np.asarray(cond_vol, dtype=float) / self.ref_vol, 0.5, 2.5)

    def interval(
        self, pred_return: np.ndarray | float, cond_vol: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ci_lower_return, ci_upper_return) around ``pred_return``."""
        scale = self._scale(cond_vol)
        pred = np.asarray(pred_return, dtype=float)
        return pred + self.q_lo * scale, pred + self.q_hi * scale


def _conditional_vol(bars_df: pd.DataFrame, window: int = 12) -> pd.Series:
    """Trailing realised vol: rolling std of 5m log returns (no lookahead)."""
    log_ret = np.log(bars_df["close"] / bars_df["close"].shift(1))
    return log_ret.rolling(window, min_periods=window // 2).std()


def _horizon_anchor_mask(ts: pd.Series, horizon: str) -> pd.Series:
    """Closed-bar anchor mask for a single horizon (no cross-horizon mixing)."""
    minute = ts.dt.minute
    if horizon == "1h":
        return minute == 0
    if horizon == "4h":
        return (ts.dt.hour % 4 == 0) & (minute == 0)
    raise ValueError(f"unknown horizon {horizon}")


def train_intraday_models(
    session: Session | None = None,
    tickers: list[str] | None = None,
    lookback_days: int = 365,
    test_days: int = 14,
    models_dir: str | Path | None = None,
) -> dict:
    """M3: train the crypto intraday log-return regressors (BTC/ETH x 1h/4h).

    Per (ticker, horizon): LGBMRegressor primary + Ridge fallback on the
    continuous log-return target, a StandardScaler fit on train, and a
    :class:`ResidualVolModel` for the CI band. All artefacts are pickled to
    ``stock_forecasting/models/intraday/`` and real validation metrics are
    written to ``metadata_crypto.json``. Returns the metadata dict.
    """
    if tickers is None:
        tickers = ["BTC-USD", "ETH-USD"]

    session = session or get_session().__enter__()
    models_dir = Path(models_dir or "stock_forecasting/models/intraday")
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict = {}

    for ticker in tickers:
        logger.info("M3 training %s", ticker)

        bars_df, funding_df = fetch_bars_and_funding(session, ticker, lookback_days)
        if len(bars_df) < 200:
            logger.warning("Insufficient data for %s: %d bars", ticker, len(bars_df))
            continue

        bars_df = bars_df.copy()
        bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True)
        bars_df = bars_df.sort_values("ts").reset_index(drop=True)

        builder = IntradayFeatureBuilder()
        features_df = builder.build_features(ticker, bars_df, funding_df)
        # M2 builds one feature row per input bar, positionally aligned.
        assert len(features_df) == len(bars_df), "feature/bar row-count mismatch"

        cond_vol = _conditional_vol(bars_df)

        for horizon in ("1h", "4h"):
            entry = _train_one(
                ticker, horizon, bars_df, features_df, cond_vol, test_days, models_dir
            )
            if entry is not None:
                metadata[f"{ticker}-{horizon}"] = entry

    metadata_path = models_dir / "metadata_crypto.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: %s (%d models)", metadata_path, len(metadata))
    return metadata


def _train_one(
    ticker: str,
    horizon: str,
    bars_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cond_vol: pd.Series,
    test_days: int,
    models_dir: Path,
) -> dict | None:
    """Train + persist one (ticker, horizon) model. Returns its metadata entry."""
    k_bars = _HORIZON_BARS[horizon]
    feature_cols = [c for c in features_df.columns if c != "ts"]

    # 1. Closed-bar anchors for THIS horizon only (non-overlapping labels).
    anchor_mask = _horizon_anchor_mask(bars_df["ts"], horizon).to_numpy()
    anchor_pos = np.flatnonzero(anchor_mask)
    anchor_pos = anchor_pos[anchor_pos + k_bars < len(bars_df)]  # need a matured target
    if len(anchor_pos) < 60:
        logger.warning(
            "%s-%s: only %d anchors, skipping", ticker, horizon, len(anchor_pos)
        )
        return None

    # 2. One tidy frame in a single index space: features + label + anchor_ts + cond_vol.
    close = bars_df["close"].to_numpy()
    frame = features_df.iloc[anchor_pos][feature_cols].reset_index(drop=True)
    frame["_y"] = np.log(close[anchor_pos + k_bars] / close[anchor_pos])
    frame["_anchor_ts"] = bars_df["ts"].to_numpy()[anchor_pos]
    frame["_cond_vol"] = cond_vol.to_numpy()[anchor_pos]
    frame["_bar_idx"] = anchor_pos
    frame = frame.dropna().reset_index(drop=True)
    if len(frame) < 60:
        logger.warning(
            "%s-%s: only %d clean rows, skipping", ticker, horizon, len(frame)
        )
        return None

    # 3. Time-ordered split: last `test_days` = test; purge label-window overlap; 24h embargo.
    anchor_ts = frame["_anchor_ts"]
    test_start = anchor_ts.max() - timedelta(days=test_days)
    is_test = (anchor_ts >= test_start).to_numpy()
    test_pos = np.flatnonzero(is_test)
    if len(test_pos) < 10 or test_pos[0] == 0:
        logger.warning(
            "%s-%s: test window too small (%d)", ticker, horizon, len(test_pos)
        )
        return None

    first_test = test_pos[0]
    anchor_bar_idx = frame[
        "_bar_idx"
    ].to_numpy()  # original bar index per surviving row
    # bar index where the test window starts, minus label window and embargo
    test_start_bar = anchor_bar_idx[first_test]
    keep_train = anchor_bar_idx < (test_start_bar - k_bars - _EMBARGO_BARS)
    train_pos = np.flatnonzero(keep_train & ~is_test)
    if len(train_pos) < 40:
        logger.warning(
            "%s-%s: train too small after purge (%d)", ticker, horizon, len(train_pos)
        )
        return None

    X = frame[feature_cols]
    y = frame["_y"]
    X_train, y_train = X.iloc[train_pos], y.iloc[train_pos]
    X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]
    vol_train = frame["_cond_vol"].to_numpy()[train_pos]
    vol_test = frame["_cond_vol"].to_numpy()[test_pos]
    logger.info(
        "  %s-%s: train=%d test=%d", ticker, horizon, len(train_pos), len(test_pos)
    )

    # 4. Scale (fit on train only), then fit both models. Frame is already NaN-free.
    scaler = StandardScaler().fit(X_train.to_numpy())
    Xs_train = scaler.transform(X_train.to_numpy())
    Xs_test = scaler.transform(X_test.to_numpy())

    lgb = LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgb.fit(Xs_train, y_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xs_train, y_train)

    # 5. Volatility model: empirical residual quantiles on TRAIN, vol-conditioned.
    train_resid = y_train.to_numpy() - lgb.predict(Xs_train)
    q_lo, q_hi = np.quantile(train_resid, [0.025, 0.975])
    ref_vol = float(np.nanmedian(vol_train)) if np.isfinite(vol_train).any() else 0.0
    vol_model = ResidualVolModel(q_lo, q_hi, ref_vol, CODE_VERSION)

    # 6. Honest test metrics.
    y_pred = lgb.predict(Xs_test)
    ci_lo, ci_hi = vol_model.interval(y_pred, vol_test)
    yt = y_test.to_numpy()
    mae = float(mean_absolute_error(yt, y_pred))
    rmse = float(np.sqrt(mean_squared_error(yt, y_pred)))
    direction_pct = float(np.mean(np.sign(y_pred) == np.sign(yt)) * 100)
    ci_cover_pct = float(np.mean((yt >= ci_lo) & (yt <= ci_hi)) * 100)

    # 7. Persist all artefacts (consistent pickle for every model).
    tc = ticker.replace("-", "_").lower()
    _pickle(models_dir / f"intraday_lgb_{tc}_{horizon}.pkl", lgb)
    _pickle(models_dir / f"intraday_ridge_{tc}_fallback_{horizon}.pkl", ridge)
    _pickle(models_dir / f"intraday_scaler_{tc}_{horizon}.pkl", scaler)
    _pickle(models_dir / f"intraday_har_rv_{tc}_{horizon}.pkl", vol_model)

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_pct": direction_pct,
        "ci_cover_pct": ci_cover_pct,
        "train_samples": len(train_pos),
        "test_samples": len(test_pos),
        "train_start": anchor_ts.iloc[train_pos[0]].isoformat(),
        "train_end": anchor_ts.iloc[train_pos[-1]].isoformat(),
        "test_start": anchor_ts.iloc[test_pos[0]].isoformat(),
        "code_version": CODE_VERSION,
    }


def _pickle(path: Path, obj: object) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backfill"
    if cmd == "train":
        train_intraday_models()
    else:
        backfill_intraday_bars()
    sys.exit(0)
