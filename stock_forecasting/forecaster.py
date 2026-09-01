"""Forecaster module for generating multi-horizon asset predictions and persisting snapshots."""

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.features import FeatureBuilder
from stock_forecasting.schema import (
    CryptoDerivative,
    ModelRun,
    PredictionSnapshot,
    Ticker,
)


@dataclass
class ForecastResult:
    """Container for forecast predictions, confidence intervals, and explainability metrics."""

    ticker: str
    horizon: str
    predicted_return: float
    predicted_price: float
    lower_bound: float
    upper_bound: float
    model_type: str
    explain: dict[str, float] = field(default_factory=dict)
    prediction_id: str | None = None
    target_ts: str | None = None
    made_at: str | None = None
    made_from_ts: str | None = None
    model_version: str | None = None
    anchor_price: float | None = None


class ForecastService:
    """Service to generate and transactionally persist multi-horizon asset forecasts."""

    HORIZON_DAYS: ClassVar[dict[str, int]] = {"1d": 1, "5d": 5, "30d": 30}

    def __init__(
        self,
        session: Session,
        model_dir: str | Path = "./model_store",
        bar_repo: BarRepository | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        """Initialize ForecastService.

        Args:
            session: Active SQLModel database session.
            model_dir: Directory where trained model artifacts (.joblib) are stored.
            bar_repo: Optional BarRepository instance.
            feature_builder: Optional FeatureBuilder instance.
        """
        self.session = session
        self.model_dir = Path(model_dir)
        self.bar_repo = bar_repo or BarRepository(session)
        self.feature_builder = feature_builder or FeatureBuilder()

    def generate_and_persist(
        self,
        ticker: str,
        horizons: Sequence[str] = ("1d", "5d", "30d"),
        model_types: Sequence[str] = ("ridge",),
    ) -> dict[str, ForecastResult]:
        """Generate forecasts for given horizons and model types and persist snapshots in a single transaction.

        Args:
            ticker: Asset ticker symbol.
            horizons: Forecast horizons to generate predictions for.
            model_types: Model types to generate predictions with.

        Returns:
            Dictionary mapping forecast keys (e.g. '1d_ridge', '1d') to ForecastResult instances.

        Raises:
            ValueError: If historical bars or active models are missing or insufficient.
            FileNotFoundError: If model artifact file is missing from disk.
        """
        # 1. Fetch latest 100 bars from repository
        bars = self.bar_repo.get_latest(ticker, limit=100)
        if not bars:
            raise ValueError(f"No bars found for ticker '{ticker}'.")
        if len(bars) < 50:
            raise ValueError(
                f"Insufficient historical bars for ticker '{ticker}': found {len(bars)}, "
                "minimum 50 required for technical indicator warmup."
            )

        # Sort bars ascending by timestamp
        bars_sorted = sorted(bars, key=lambda b: str(b.ts))

        bars_df = pd.DataFrame(
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
                for b in bars_sorted
            ]
        ).reset_index(drop=True)

        # 2. Build unscaled technical indicator features
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
                f"Feature extraction produced 0 rows for ticker '{ticker}' (warmup period not satisfied)."
            )

        # 3. Extract anchor price and timestamp from the latest bar
        last_bar = bars_sorted[-1]
        made_from_ts = str(last_bar.ts)
        if last_bar.adj_close is not None and not np.isnan(last_bar.adj_close):
            anchor_price = float(last_bar.adj_close)
        else:
            anchor_price = float(last_bar.close)

        latest_feature_row = features_df.iloc[-1]
        now_iso = datetime.now(UTC).isoformat()

        results: dict[str, ForecastResult] = {}

        snapshots: list[PredictionSnapshot] = []

        # 4. Generate forecasts for each horizon and model type
        for horizon in horizons:
            if horizon not in self.HORIZON_DAYS:
                raise ValueError(
                    f"Unsupported horizon '{horizon}'. Must be one of {list(self.HORIZON_DAYS.keys())}"
                )
            h_days = self.HORIZON_DAYS[horizon]
            target_ts = self._compute_target_ts(ticker, made_from_ts, horizon)

            for model_type in model_types:
                norm_model_type = model_type.lower()

                # Query active ModelRun
                stmt = (
                    select(ModelRun)
                    .where(
                        ModelRun.ticker == ticker,
                        ModelRun.horizon == horizon,
                        ModelRun.model_type == norm_model_type,
                        ModelRun.is_active == 1,
                    )
                    .order_by(ModelRun.id.desc())
                )
                model_run = self.session.exec(stmt).first()
                if model_run is None:
                    raise ValueError(
                        f"No active ModelRun found for ticker '{ticker}', horizon '{horizon}', model_type '{norm_model_type}'."
                    )

                # Load artifact
                artifact_path = Path(model_run.artifact_path)
                if not artifact_path.exists():
                    alt_path = self.model_dir / artifact_path.name
                    if alt_path.exists():
                        artifact_path = alt_path
                    else:
                        raise FileNotFoundError(
                            f"Model artifact file not found at '{artifact_path}'."
                        )

                artifact = joblib.load(artifact_path)
                model = artifact["model"]
                scaler = artifact.get("scaler")
                feature_cols = artifact.get(
                    "feature_list",
                    [c for c in features_df.columns if c != "ts"],
                )
                residual_std = float(
                    artifact.get("residual_std", model_run.residual_std)
                )
                model_version = str(
                    artifact.get("model_version", model_run.model_version)
                )

                # Check missing features
                missing_cols = [c for c in feature_cols if c not in features_df.columns]
                if missing_cols:
                    raise ValueError(
                        f"Missing required feature columns in input: {missing_cols}"
                    )

                # Extract and scale features for the latest bar
                X = latest_feature_row[feature_cols].values.reshape(1, -1)
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X

                # Predict return and compute predicted price
                predicted_return = float(model.predict(X_scaled)[0])
                predicted_price = float(anchor_price * np.exp(predicted_return))

                # Confidence intervals (95% CI scaled by sqrt(h))
                scaled_std = float(residual_std * np.sqrt(h_days))
                lower_bound = float(predicted_price * np.exp(-1.96 * scaled_std))
                upper_bound = float(predicted_price * np.exp(1.96 * scaled_std))

                # Compute feature explainability
                if hasattr(model, "coef_"):
                    coefs = model.coef_
                    if hasattr(coefs, "ravel"):
                        coefs = coefs.ravel()
                    contributions = coefs * X_scaled[0]
                    explain = {
                        str(feat): float(val)
                        for feat, val in zip(feature_cols, contributions, strict=False)
                    }
                elif hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    explain = {
                        str(feat): float(val)
                        for feat, val in zip(feature_cols, importances, strict=False)
                    }
                else:
                    explain = {}

                # Create PredictionSnapshot
                prediction_id = str(uuid.uuid4())
                snapshot = PredictionSnapshot(
                    prediction_id=prediction_id,
                    ticker=ticker,
                    made_at=now_iso,
                    made_from_ts=made_from_ts,
                    anchor_price=anchor_price,
                    horizon=horizon,
                    target_ts=target_ts,
                    predicted_return=predicted_return,
                    predicted_price=predicted_price,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    model_type=norm_model_type,
                    model_version=model_version,
                    model_run_id=model_run.id,
                    explain_json=json.dumps(explain),
                    input_is_stale=0,
                )
                snapshots.append(snapshot)

                result = ForecastResult(
                    ticker=ticker,
                    horizon=horizon,
                    predicted_return=predicted_return,
                    predicted_price=predicted_price,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    model_type=norm_model_type,
                    explain=explain,
                    prediction_id=prediction_id,
                    target_ts=target_ts,
                    made_at=now_iso,
                    made_from_ts=made_from_ts,
                    model_version=model_version,
                    anchor_price=anchor_price,
                )
                results[f"{horizon}_{norm_model_type}"] = result
                if horizon not in results:
                    results[horizon] = result

        # 5. Persist all snapshots in ONE atomic transaction
        for s in snapshots:
            self.session.add(s)
        self.session.commit()
        for s in snapshots:
            self.session.refresh(s)

        return results

    def predict(
        self,
        ticker: str,
        horizon: str = "1d",
        model_type: str = "ridge",
    ) -> ForecastResult:
        """Generate a single forecast and persist its snapshot.

        Args:
            ticker: Asset ticker symbol.
            horizon: Forecast horizon ('1d', '5d', '30d').
            model_type: Model type ('ridge', 'random_forest').

        Returns:
            ForecastResult instance.
        """
        results = self.generate_and_persist(
            ticker=ticker,
            horizons=[horizon],
            model_types=[model_type],
        )
        return results[f"{horizon}_{model_type.lower()}"]

    def _compute_target_ts(self, ticker: str, made_from_ts: str, horizon: str) -> str:
        from stock_forecasting.schema import Ticker

        h_days = self.HORIZON_DAYS[horizon]
        made_dt = pd.to_datetime(made_from_ts, utc=True)

        ticker_obj = self.session.get(Ticker, ticker)
        asset_class = ticker_obj.asset_class if ticker_obj else "equity"

        if asset_class == "equity":
            # Add business days
            target_dt = made_dt + pd.offsets.BDay(h_days)
            # if made_dt was already on a weekend, BDay(h_days) might not behave exactly intuitively,
            # but it is a standard approach.
        else:
            target_dt = made_dt + pd.Timedelta(days=h_days)

        return target_dt.isoformat()


Forecaster = ForecastService
