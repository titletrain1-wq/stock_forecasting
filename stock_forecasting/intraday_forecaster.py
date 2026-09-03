"""M4: intraday forecast writer.

The sole writer to ``intraday_prediction_snapshots``. Runs hourly from the
worker (never the render path, per design F1). For each (ticker, horizon) it
loads the M3 artefacts, anchors on the most recent CLOSED bar, predicts the
log-return + CI band, and writes one snapshot row (``INSERT OR IGNORE`` on the
``(ticker, horizon, anchor_ts)`` unique constraint).
"""

from __future__ import annotations

import logging
import pickle
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlmodel import Session, select

from stock_forecasting.intraday_features import IntradayFeatureBuilder
from stock_forecasting.intraday_trainer import (
    CODE_VERSION,
    ResidualVolModel,  # noqa: F401  (needed for unpickling)
    _conditional_vol,
    _horizon_anchor_mask,
)
from stock_forecasting.schema import CryptoDerivative, IntradayBarsHistory

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("stock_forecasting/models/intraday")
CRYPTO_TICKERS = ("BTC-USD", "ETH-USD")
HORIZON_DELTA = {"1h": timedelta(hours=1), "4h": timedelta(hours=4)}
_FEATURE_WINDOW_BARS = 600  # 5m bars pulled to build features for the latest anchor


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _load_models(models_dir: Path, tc: str, horizon: str) -> dict | None:
    paths = {
        "lgb": models_dir / f"intraday_lgb_{tc}_{horizon}.pkl",
        "ridge": models_dir / f"intraday_ridge_{tc}_fallback_{horizon}.pkl",
        "scaler": models_dir / f"intraday_scaler_{tc}_{horizon}.pkl",
        "vol": models_dir / f"intraday_har_rv_{tc}_{horizon}.pkl",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pickle.loads(p.read_bytes()) for k, p in paths.items()}


def _recent_bars(session: Session, ticker: str, limit: int) -> pd.DataFrame:
    rows = session.exec(
        select(IntradayBarsHistory)
        .where(
            (IntradayBarsHistory.ticker == ticker)
            & (IntradayBarsHistory.interval == "5m")
        )
        .order_by(IntradayBarsHistory.ts.desc())
        .limit(limit)
    ).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "ts": r.ts,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def _recent_funding(session: Session, ticker: str, since: datetime) -> pd.DataFrame:
    rows = session.exec(
        select(CryptoDerivative).where(
            (CryptoDerivative.ticker == ticker)
            & (CryptoDerivative.funding_rate.isnot(None))
            & (CryptoDerivative.ts >= since.isoformat().replace("+00:00", "Z"))
        )
    ).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{"ts": r.ts, "funding_rate": r.funding_rate} for r in rows])


def write_intraday_forecasts(
    session: Session,
    tickers: tuple[str, ...] = CRYPTO_TICKERS,
    horizons: tuple[str, ...] = ("1h", "4h"),
    models_dir: str | Path | None = None,
) -> list[dict]:
    """Write one forecast snapshot per (ticker, horizon) for the latest closed bar.

    Returns the list of snapshot dicts it attempted to write (for logging/tests).
    """
    models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
    written: list[dict] = []
    made_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    sha = _git_sha()

    for ticker in tickers:
        tc = ticker.replace("-", "_").lower()
        bars = _recent_bars(session, ticker, _FEATURE_WINDOW_BARS)
        if len(bars) < 60:
            logger.warning("intraday forecast: %s has only %d bars", ticker, len(bars))
            continue
        funding = _recent_funding(
            session, ticker, bars["ts"].min().to_pydatetime() - timedelta(days=20)
        )
        feats = IntradayFeatureBuilder().build_features(ticker, bars, funding)
        cond_vol = _conditional_vol(bars).to_numpy()
        feature_cols = [c for c in feats.columns if c != "ts"]

        for horizon in horizons:
            models = _load_models(models_dir, tc, horizon)
            if models is None:
                logger.warning(
                    "intraday forecast: no models for %s-%s", ticker, horizon
                )
                continue

            anchor_mask = _horizon_anchor_mask(bars["ts"], horizon).to_numpy()
            row_ok = anchor_mask & ~np.isnan(feats[feature_cols].to_numpy()).any(axis=1)
            anchor_pos = np.flatnonzero(row_ok)
            if len(anchor_pos) == 0:
                logger.warning(
                    "intraday forecast: no clean anchor for %s-%s", ticker, horizon
                )
                continue
            i = int(anchor_pos[-1])  # most recent closed, feature-complete anchor

            x = feats.iloc[[i]][feature_cols].to_numpy()
            xs = models["scaler"].transform(x)
            pred_return = float(models["lgb"].predict(xs)[0])
            cv = cond_vol[i] if np.isfinite(cond_vol[i]) else models["vol"].ref_vol
            lo_r, hi_r = models["vol"].interval(pred_return, cv)
            lo_r, hi_r = float(np.ravel(lo_r)[0]), float(np.ravel(hi_r)[0])

            anchor_price = float(bars["close"].iloc[i])
            anchor_ts = bars["ts"].iloc[i]
            snap = {
                "ticker": ticker,
                "horizon": horizon,
                "made_at": made_at,
                "anchor_ts": anchor_ts.isoformat().replace("+00:00", "Z"),
                "anchor_price": anchor_price,
                "predicted_return": pred_return,
                "predicted_price": anchor_price * float(np.exp(pred_return)),
                "ci_lower_return": lo_r,
                "ci_upper_return": hi_r,
                "ci_lower_price": anchor_price * float(np.exp(lo_r)),
                "ci_upper_price": anchor_price * float(np.exp(hi_r)),
                "target_ts": (anchor_ts + HORIZON_DELTA[horizon])
                .isoformat()
                .replace("+00:00", "Z"),
                "model_version": CODE_VERSION,
                "model_sha": sha,
            }
            session.connection().execute(
                text(
                    "INSERT OR IGNORE INTO intraday_prediction_snapshots "
                    "(ticker, horizon, made_at, anchor_ts, anchor_price, predicted_return, "
                    " predicted_price, ci_lower_return, ci_upper_return, ci_lower_price, "
                    " ci_upper_price, target_ts, model_version, model_sha) VALUES "
                    "(:ticker, :horizon, :made_at, :anchor_ts, :anchor_price, :predicted_return, "
                    " :predicted_price, :ci_lower_return, :ci_upper_return, :ci_lower_price, "
                    " :ci_upper_price, :target_ts, :model_version, :model_sha)"
                ),
                snap,
            )
            written.append(snap)

    session.commit()
    logger.info("intraday forecast writer: %d snapshots processed", len(written))
    return written
