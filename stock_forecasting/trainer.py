"""Model training pipeline with walk-forward validation and artifact persistence."""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.features import FeatureBuilder
from stock_forecasting.schema import CryptoDerivative, ModelRun, Ticker


def _get_git_sha() -> str:
    """Retrieve the current Git commit SHA, or 'unknown' if not in a git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


@dataclass
class ModelArtifact:
    """Container for trained model outputs, walk-forward validation metrics, and metadata."""

    ticker: str
    horizon: str
    model_type: str
    model_version: str
    code_git_sha: str
    wf_mae: float
    wf_rmse: float
    wf_dir_acc: float
    wf_ci_cov: float
    residual_std: float
    feature_list: list[str]
    hyperparams: dict[str, Any]
    random_seed: int
    artifact_path: str
    model_run_id: int | None = None
    model: Any = None
    scaler: Any = None
    metrics: dict[str, float] = field(default_factory=dict)


class Trainer:
    """Trains walk-forward predictive models for asset return forecasting."""

    HORIZON_DAYS: ClassVar[dict[str, int]] = {"1d": 1, "5d": 5, "30d": 30}

    def __init__(
        self,
        session: Session,
        model_dir: str | Path = "./model_store",
        bar_repo: BarRepository | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            session: Active SQLModel/SQLAlchemy session.
            model_dir: Directory path for persisting .joblib model artifacts.
            bar_repo: Optional BarRepository instance (defaults to new instance with session).
            feature_builder: Optional FeatureBuilder instance (defaults to new FeatureBuilder).
        """
        self.session = session
        self.model_dir = Path(model_dir)
        self.bar_repo = bar_repo or BarRepository(session)
        self.feature_builder = feature_builder or FeatureBuilder()

    def train(
        self,
        ticker: str,
        horizon: str,
        model_type: str,
        model_version: str = "1.0.0",
        random_seed: int = 42,
    ) -> ModelArtifact:
        """Train a walk-forward forecasting model and persist artifact + DB record.

        Args:
            ticker: Asset ticker symbol (e.g. 'AAPL').
            horizon: Forecast horizon key ('1d', '5d', '30d').
            model_type: Model architecture ('ridge' or 'random_forest').
            model_version: Semantic version string for the model artifact.
            random_seed: Random seed for reproducibility.

        Returns:
            ModelArtifact containing model metadata, out-of-sample metrics, and artifact path.

        Raises:
            ValueError: If horizon or model_type is unsupported, or if historical data is insufficient.
        """
        if horizon not in self.HORIZON_DAYS:
            raise ValueError(
                f"Unsupported horizon: '{horizon}'. Must be one of {list(self.HORIZON_DAYS.keys())}"
            )

        norm_model_type = model_type.lower()
        if norm_model_type not in ("ridge", "random_forest"):
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. Must be 'ridge' or 'random_forest'"
            )

        # 1. Fetch historical bars
        bars = self.bar_repo.get_range(ticker, "0000", "9999")
        if not bars or len(bars) < 60:
            raise ValueError(
                f"Insufficient historical bars for ticker '{ticker}': found {len(bars) if bars else 0} bars "
                "(minimum 60 required for indicator warmup and walk-forward split)."
            )

        bars_df = (
            pd.DataFrame(
                [
                    {
                        "ts": b.ts,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "adj_close": b.adj_close,
                        "volume": b.volume,
                    }
                    for b in bars
                ]
            )
            .sort_values("ts")
            .reset_index(drop=True)
        )

        # 2. Build features
        ticker_row = self.session.exec(
            select(Ticker).where(Ticker.symbol == ticker)
        ).first()
        asset_class = ticker_row.asset_class if ticker_row else "equity"
        derivatives_df = None
        if asset_class == "crypto":
            deriv_rows = self.session.exec(
                select(CryptoDerivative).where(CryptoDerivative.ticker == ticker)
            ).all()
            if deriv_rows:
                derivatives_df = pd.DataFrame(
                    [
                        {
                            "ts": r.ts,
                            "funding_rate": r.funding_rate,
                            "open_interest": r.open_interest,
                        }
                        for r in deriv_rows
                    ]
                )

        features_df = self.feature_builder.build(
            bars_df, scale=False, asset_class=asset_class, derivatives_df=derivatives_df
        )

        if features_df.empty:
            raise ValueError(
                f"Insufficient data: feature extraction produced 0 rows for ticker '{ticker}'."
            )

        # 3. Compute log return target: log(close_{t+h} / close_t)
        h = self.HORIZON_DAYS[horizon]
        if "adj_close" in bars_df.columns and bars_df["adj_close"].notna().any():
            close_series = bars_df["adj_close"].fillna(bars_df["close"]).astype(float)
        else:
            close_series = bars_df["close"].astype(float)

        target_series = np.log(close_series.shift(-h) / close_series)
        target_col = f"target_{horizon}"
        features_df[target_col] = target_series.loc[features_df.index]

        valid_df = features_df.dropna(subset=[target_col]).copy()
        min_required_samples = 10
        if len(valid_df) < min_required_samples:
            raise ValueError(
                f"Insufficient valid samples ({len(valid_df)}) for ticker '{ticker}' and horizon '{horizon}' "
                f"(minimum {min_required_samples} required after feature warmup and target shift)."
            )

        feature_cols = [
            c
            for c in self.feature_builder.feature_cols
            if c in valid_df.columns and c != "ts"
        ]

        X_full = valid_df[feature_cols].values
        y_full = valid_df[target_col].values

        n_splits = 5
        if len(valid_df) < n_splits * 2:
            n_splits = max(2, len(valid_df) // 2)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        y_pred_all = []
        y_test_all = []

        if norm_model_type == "ridge":
            hyperparams: dict[str, Any] = {"alpha": 1.0}
        else:
            hyperparams = {"n_estimators": 100, "random_state": random_seed}

        for train_index, test_index in tscv.split(X_full):
            X_train, X_test = X_full[train_index], X_full[test_index]
            y_train, y_test = y_full[train_index], y_full[test_index]

            wf_scaler = StandardScaler()
            X_train_scaled = wf_scaler.fit_transform(X_train)
            X_test_scaled = wf_scaler.transform(X_test)

            if norm_model_type == "ridge":
                wf_model = Ridge(alpha=1.0)
            else:
                wf_model = RandomForestRegressor(
                    n_estimators=100, random_state=random_seed
                )

            wf_model.fit(X_train_scaled, y_train)
            y_pred = wf_model.predict(X_test_scaled)

            y_pred_all.extend(y_pred)
            y_test_all.extend(y_test)

        y_pred_all = np.array(y_pred_all)
        y_test_all = np.array(y_test_all)

        wf_mae = float(np.mean(np.abs(y_test_all - y_pred_all)))
        wf_rmse = float(np.sqrt(np.mean((y_test_all - y_pred_all) ** 2)))
        wf_dir_acc = float(np.mean(np.sign(y_pred_all) == np.sign(y_test_all)))

        residuals = y_test_all - y_pred_all
        residual_std = (
            float(np.std(residuals, ddof=1))
            if len(residuals) > 1
            else float(np.std(residuals))
        )

        # residual_std is already a h-horizon quantity: the target is the h-day
        # cumulative log return (see target_series above), so residuals are h-day
        # errors. Do NOT scale by sqrt(h) again -- that double-counts time and
        # inflates the interval until wf_ci_cov pins at 1.0.
        scaled_std = residual_std

        if scaled_std > 1e-12:
            ci_lower = y_pred_all - 1.96 * scaled_std
            ci_upper = y_pred_all + 1.96 * scaled_std
            wf_ci_cov = float(
                np.mean((y_test_all >= ci_lower) & (y_test_all <= ci_upper))
            )
        else:
            wf_ci_cov = 1.0 if len(y_test_all) > 0 else 0.0

        # Fit production model on entire valid dataset with full scaler
        prod_scaler = StandardScaler()
        X_full = valid_df[feature_cols].values
        y_full = valid_df[target_col].values
        X_full_scaled = prod_scaler.fit_transform(X_full)

        if norm_model_type == "ridge":
            prod_model = Ridge(alpha=1.0)
        else:
            prod_model = RandomForestRegressor(
                n_estimators=100, random_state=random_seed
            )
        prod_model.fit(X_full_scaled, y_full)

        # 7. Dump .joblib model artifact
        dest_dir = self.model_dir / ticker / horizon / norm_model_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = dest_dir / f"{model_version}.joblib"

        metrics_dict = {
            "wf_mae": wf_mae,
            "wf_rmse": wf_rmse,
            "wf_dir_acc": wf_dir_acc,
            "wf_ci_cov": wf_ci_cov,
            "residual_std": residual_std,
        }

        train_start = str(bars_df["ts"].iloc[0])
        train_end = str(bars_df["ts"].iloc[-1])

        artifact_payload = {
            "model": prod_model,
            "scaler": prod_scaler,
            "feature_list": feature_cols,
            "ticker": ticker,
            "horizon": horizon,
            "model_type": norm_model_type,
            "model_version": model_version,
            "residual_std": residual_std,
            "hyperparams": hyperparams,
            "random_seed": random_seed,
            "metrics": metrics_dict,
            "train_start": train_start,
            "train_end": train_end,
        }
        joblib.dump(artifact_payload, artifact_path)

        # 8. Deactivate previous active ModelRun rows and insert new ModelRun
        deactivate_stmt = select(ModelRun).where(
            ModelRun.ticker == ticker,
            ModelRun.horizon == horizon,
            ModelRun.model_type == norm_model_type,
            ModelRun.is_active == 1,
        )
        previous_runs = self.session.exec(deactivate_stmt).all()
        for prev_run in previous_runs:
            prev_run.is_active = 0
            self.session.add(prev_run)

        now_iso = datetime.now(UTC).isoformat()
        git_sha = _get_git_sha()

        model_run = ModelRun(
            ticker=ticker,
            horizon=horizon,
            model_type=norm_model_type,
            model_version=model_version,
            code_git_sha=git_sha,
            trained_at=now_iso,
            train_start=train_start,
            train_end=train_end,
            hyperparams_json=json.dumps(hyperparams),
            feature_list_json=json.dumps(feature_cols),
            random_seed=random_seed,
            wf_mae=wf_mae,
            wf_rmse=wf_rmse,
            wf_dir_acc=wf_dir_acc,
            wf_ci_cov=wf_ci_cov,
            residual_std=residual_std,
            artifact_path=str(artifact_path),
            is_active=1,
        )
        self.session.add(model_run)
        self.session.commit()
        self.session.refresh(model_run)

        return ModelArtifact(
            ticker=ticker,
            horizon=horizon,
            model_type=norm_model_type,
            model_version=model_version,
            code_git_sha=git_sha,
            wf_mae=wf_mae,
            wf_rmse=wf_rmse,
            wf_dir_acc=wf_dir_acc,
            wf_ci_cov=wf_ci_cov,
            residual_std=residual_std,
            feature_list=feature_cols,
            hyperparams=hyperparams,
            random_seed=random_seed,
            artifact_path=str(artifact_path),
            model_run_id=model_run.id,
            model=prod_model,
            scaler=prod_scaler,
            metrics=metrics_dict,
        )
