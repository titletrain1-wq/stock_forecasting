"""Tests for Trainer and ModelArtifact walk-forward training pipeline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.providers.base import Bar
from stock_forecasting.schema import ModelRun
from stock_forecasting.trainer import ModelArtifact, Trainer


def _insert_sample_bars(
    session: Session, ticker: str = "TEST", n: int = 100, seed: int = 42
) -> None:
    """Helper to insert deterministic synthetic bars for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    log_returns = np.random.normal(0.0005, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(np.random.normal(0.005, 0.005, n)))
    low = close * (1.0 - np.abs(np.random.normal(0.005, 0.005, n)))
    open_ = (high + low) / 2.0
    volume = np.random.lognormal(10.0, 0.4, n)

    bars = [
        Bar(
            ts=dates[i].isoformat(),
            open=float(open_[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            adj_close=float(close[i]),
            volume=float(volume[i]),
        )
        for i in range(n)
    ]
    repo = BarRepository(session)
    repo.upsert_bars(ticker=ticker, bars=bars, source="test")


def test_trainer_ridge_1d(db_session: Session, tmp_path: Path) -> None:
    """Verify training a Ridge model on 1d horizon produces valid artifact and DB record."""
    _insert_sample_bars(db_session, ticker="TEST", n=100)

    trainer = Trainer(session=db_session, model_dir=tmp_path)
    artifact = trainer.train(ticker="TEST", horizon="1d", model_type="ridge")

    # Verify ModelArtifact fields
    assert isinstance(artifact, ModelArtifact)
    assert artifact.ticker == "TEST"
    assert artifact.horizon == "1d"
    assert artifact.model_type == "ridge"
    assert artifact.model_version == "1.0.0"
    assert artifact.code_git_sha is not None
    assert artifact.wf_mae >= 0.0
    assert artifact.wf_rmse >= 0.0
    assert 0.0 <= artifact.wf_dir_acc <= 1.0
    assert 0.0 <= artifact.wf_ci_cov <= 1.0
    assert artifact.residual_std >= 0.0
    assert len(artifact.feature_list) > 0
    assert artifact.hyperparams == {"alpha": 1.0}
    assert artifact.random_seed == 42
    assert artifact.model_run_id is not None
    assert Path(artifact.artifact_path).exists()
    assert artifact.model is not None
    assert artifact.scaler is not None

    # Verify ModelRun in database
    run = db_session.get(ModelRun, artifact.model_run_id)
    assert run is not None
    assert run.ticker == "TEST"
    assert run.horizon == "1d"
    assert run.model_type == "ridge"
    assert run.is_active == 1
    assert json.loads(run.feature_list_json) == artifact.feature_list
    assert json.loads(run.hyperparams_json) == {"alpha": 1.0}

    # Retrain: verify old run is deactivated (is_active=0) and new run is active
    artifact_v2 = trainer.train(
        ticker="TEST",
        horizon="1d",
        model_type="ridge",
        model_version="1.1.0",
    )
    db_session.refresh(run)
    assert run.is_active == 0

    run_v2 = db_session.get(ModelRun, artifact_v2.model_run_id)
    assert run_v2 is not None
    assert run_v2.is_active == 1
    assert run_v2.model_version == "1.1.0"


def test_trainer_random_forest_5d(db_session: Session, tmp_path: Path) -> None:
    """Verify training a Random Forest model on 5d horizon."""
    _insert_sample_bars(db_session, ticker="AAPL", n=120)

    trainer = Trainer(session=db_session, model_dir=tmp_path)
    artifact = trainer.train(ticker="AAPL", horizon="5d", model_type="random_forest")

    assert artifact.ticker == "AAPL"
    assert artifact.horizon == "5d"
    assert artifact.model_type == "random_forest"
    assert artifact.hyperparams == {"n_estimators": 100, "random_state": 42}
    assert Path(artifact.artifact_path).exists()

    # Verify joblib payload integrity
    payload = joblib.load(artifact.artifact_path)
    assert "model" in payload
    assert "scaler" in payload
    assert payload["ticker"] == "AAPL"
    assert payload["horizon"] == "5d"
    assert payload["model_type"] == "random_forest"
    assert "metrics" in payload
    assert payload["metrics"]["wf_mae"] == artifact.wf_mae


def test_trainer_insufficient_data(db_session: Session, tmp_path: Path) -> None:
    """Verify that insufficient bar counts raise descriptive ValueError."""
    trainer = Trainer(session=db_session, model_dir=tmp_path)

    # Case 1: 0 bars
    with pytest.raises(ValueError, match="Insufficient historical bars"):
        trainer.train(ticker="EMPTY", horizon="1d", model_type="ridge")

    # Case 2: 10 bars (too few for SMA50 warmup)
    _insert_sample_bars(db_session, ticker="FEW", n=10)
    with pytest.raises(ValueError, match="Insufficient historical bars"):
        trainer.train(ticker="FEW", horizon="1d", model_type="ridge")


def test_trainer_unsupported_params(db_session: Session, tmp_path: Path) -> None:
    """Verify error handling on unsupported horizons or model types."""
    _insert_sample_bars(db_session, ticker="TEST", n=100)
    trainer = Trainer(session=db_session, model_dir=tmp_path)

    with pytest.raises(ValueError, match="Unsupported horizon"):
        trainer.train(ticker="TEST", horizon="10d", model_type="ridge")

    with pytest.raises(ValueError, match="Unsupported model_type"):
        trainer.train(ticker="TEST", horizon="1d", model_type="neural_net")
