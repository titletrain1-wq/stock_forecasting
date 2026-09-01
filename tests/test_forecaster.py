"""Unit and integration tests for ForecastService and Forecaster."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session, select

from stock_forecasting.bar_store import BarRepository
from stock_forecasting.forecaster import Forecaster, ForecastResult, ForecastService
from stock_forecasting.providers.base import Bar
from stock_forecasting.schema import PredictionSnapshot
from stock_forecasting.trainer import Trainer


def _insert_sample_bars(
    session: Session,
    ticker: str = "AAPL",
    n: int = 100,
    seed: int = 42,
) -> None:
    """Helper to insert deterministic synthetic bars for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    log_returns = np.random.normal(0.0005, 0.015, n)
    close = 150.0 * np.exp(np.cumsum(log_returns))
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


def test_forecaster_input_is_stale_reflects_anchor_freshness(
    db_session: Session, tmp_path: Path
) -> None:
    """input_is_stale is computed from the anchor bar's schedule, not hardcoded 0."""
    ticker = "AAPL"
    _insert_sample_bars(db_session, ticker=ticker, n=120)
    trainer = Trainer(session=db_session, model_dir=tmp_path)
    trainer.train(ticker=ticker, horizon="1d", model_type="ridge")
    service = ForecastService(session=db_session, model_dir=tmp_path)

    last_ts = db_session.exec(
        select(PredictionSnapshot)
    ).first()  # none yet; get anchor from bars instead
    assert last_ts is None

    # Evaluate "now" right at the last bar's timestamp -> fresh.
    anchor_dt = pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(days=119)
    fresh = service.generate_and_persist(
        ticker=ticker,
        horizons=["1d"],
        model_types=["ridge"],
        now=anchor_dt.to_pydatetime(),
    )
    fresh_snap = db_session.exec(
        select(PredictionSnapshot).where(
            PredictionSnapshot.prediction_id == fresh["1d_ridge"].prediction_id
        )
    ).first()
    assert fresh_snap.input_is_stale == 0

    # Evaluate a week later with no new bar -> stale.
    stale = service.generate_and_persist(
        ticker=ticker,
        horizons=["1d"],
        model_types=["ridge"],
        now=(anchor_dt + pd.Timedelta(days=8)).to_pydatetime(),
    )
    stale_snap = db_session.exec(
        select(PredictionSnapshot).where(
            PredictionSnapshot.prediction_id == stale["1d_ridge"].prediction_id
        )
    ).first()
    assert stale_snap.input_is_stale == 1


def test_forecaster_predict(db_session: Session, tmp_path: Path) -> None:
    """Verify forecasting with a trained Ridge model generates valid predictions, CI, and persisted snapshot."""
    ticker = "AAPL"
    _insert_sample_bars(db_session, ticker=ticker, n=100)

    # 1. Train model
    trainer = Trainer(session=db_session, model_dir=tmp_path)
    artifact = trainer.train(ticker=ticker, horizon="1d", model_type="ridge")
    assert artifact is not None

    # 2. Generate and persist forecast
    service = ForecastService(session=db_session, model_dir=tmp_path)
    results = service.generate_and_persist(
        ticker=ticker, horizons=["1d"], model_types=["ridge"]
    )

    assert "1d_ridge" in results
    assert "1d" in results

    res = results["1d_ridge"]
    assert isinstance(res, ForecastResult)
    assert res.ticker == ticker
    assert res.horizon == "1d"
    assert res.model_type == "ridge"
    assert np.isfinite(res.predicted_return)
    assert res.predicted_price > 0
    assert res.lower_bound < res.predicted_price < res.upper_bound
    assert isinstance(res.explain, dict)
    assert len(res.explain) > 0
    for feat, val in res.explain.items():
        assert isinstance(feat, str)
        assert np.isfinite(val)

    # 3. Verify PredictionSnapshot row in database
    stmt = select(PredictionSnapshot).where(
        PredictionSnapshot.ticker == ticker,
        PredictionSnapshot.horizon == "1d",
    )
    snapshots = list(db_session.exec(stmt).all())
    assert len(snapshots) == 1

    snap = snapshots[0]
    assert snap.prediction_id == res.prediction_id
    assert snap.ticker == ticker
    assert snap.horizon == "1d"
    assert snap.model_type == "ridge"
    assert snap.model_run_id == artifact.model_run_id
    assert snap.predicted_price == pytest.approx(res.predicted_price)
    assert snap.predicted_return == pytest.approx(res.predicted_return)
    assert snap.lower_bound == pytest.approx(res.lower_bound)
    assert snap.upper_bound == pytest.approx(res.upper_bound)
    # Synthetic bars end in 2023; evaluated at real "now" the anchor is years
    # behind its expected schedule -> stale input flagged.
    assert snap.input_is_stale == 1
    assert snap.realized_price is None

    # Verify explain_json matches explain dict
    explain_data = json.loads(snap.explain_json)
    assert explain_data == res.explain


def test_forecaster_missing_model(db_session: Session, tmp_path: Path) -> None:
    """Verify appropriate errors are raised when bars or models are missing."""
    service = ForecastService(session=db_session, model_dir=tmp_path)

    # Missing bars
    with pytest.raises(ValueError, match="No bars found"):
        service.generate_and_persist(
            ticker="NONEXISTENT", horizons=["1d"], model_types=["ridge"]
        )

    # Seed bars, but no trained model
    _insert_sample_bars(db_session, ticker="AAPL", n=100)
    with pytest.raises(ValueError, match="No active ModelRun found"):
        service.generate_and_persist(
            ticker="AAPL", horizons=["1d"], model_types=["ridge"]
        )


def test_forecaster_target_ts_calculation(db_session: Session, tmp_path: Path) -> None:
    """Verify target_ts is correctly offset by horizon days from made_from_ts."""
    ticker = "AAPL"
    _insert_sample_bars(db_session, ticker=ticker, n=100)

    trainer = Trainer(session=db_session, model_dir=tmp_path)
    for h in ["1d", "5d", "30d"]:
        trainer.train(ticker=ticker, horizon=h, model_type="ridge")

    service = ForecastService(session=db_session, model_dir=tmp_path)
    results = service.generate_and_persist(
        ticker=ticker,
        horizons=["1d", "5d", "30d"],
        model_types=["ridge"],
    )

    for h, days in [("1d", 1), ("5d", 5), ("30d", 30)]:
        res = results[f"{h}_ridge"]
        made_dt = pd.to_datetime(res.made_from_ts, utc=True)
        target_dt = pd.to_datetime(res.target_ts, utc=True)
        # For equity, it uses BDay
        expected_target_dt = made_dt + pd.offsets.BDay(days)
        assert target_dt == expected_target_dt


def test_forecaster_target_ts_crypto(db_session: Session, tmp_path: Path) -> None:
    """Verify target_ts is correctly offset by horizon days for crypto."""
    ticker = "BTC"
    from stock_forecasting.schema import Ticker

    db_session.add(
        Ticker(
            symbol=ticker,
            asset_class="crypto",
            display_name="Bitcoin",
            provider="ccxt",
            provider_symbol="BTC/USD",
            price_basis="raw",
            added_at="2026-09-01T00:00:00Z",
        )
    )
    db_session.commit()
    _insert_sample_bars(db_session, ticker=ticker, n=100)

    trainer = Trainer(session=db_session, model_dir=tmp_path)
    trainer.train(ticker=ticker, horizon="5d", model_type="ridge")

    service = ForecastService(session=db_session, model_dir=tmp_path)
    results = service.generate_and_persist(
        ticker=ticker,
        horizons=["5d"],
        model_types=["ridge"],
    )

    res = results["5d_ridge"]
    made_dt = pd.to_datetime(res.made_from_ts, utc=True)
    target_dt = pd.to_datetime(res.target_ts, utc=True)
    diff_days = (target_dt - made_dt).total_seconds() / 86400
    assert diff_days == pytest.approx(5)


def test_forecaster_random_forest(db_session: Session, tmp_path: Path) -> None:
    """Verify forecasting and feature importances with RandomForest model."""
    ticker = "MSFT"
    _insert_sample_bars(db_session, ticker=ticker, n=100)

    trainer = Trainer(session=db_session, model_dir=tmp_path)
    trainer.train(ticker=ticker, horizon="1d", model_type="random_forest")

    service = Forecaster(session=db_session, model_dir=tmp_path)
    res = service.predict(ticker=ticker, horizon="1d", model_type="random_forest")

    assert res.ticker == ticker
    assert res.model_type == "random_forest"
    assert res.lower_bound < res.predicted_price < res.upper_bound
    assert len(res.explain) > 0
    for imp in res.explain.values():
        assert imp >= 0.0


def test_forecaster_transaction_atomic(db_session: Session, tmp_path: Path) -> None:
    """Verify multiple snapshots are persisted in a single transaction."""
    ticker = "AAPL"
    _insert_sample_bars(db_session, ticker=ticker, n=100)

    trainer = Trainer(session=db_session, model_dir=tmp_path)
    trainer.train(ticker=ticker, horizon="1d", model_type="ridge")
    trainer.train(ticker=ticker, horizon="5d", model_type="ridge")

    service = ForecastService(session=db_session, model_dir=tmp_path)
    results = service.generate_and_persist(
        ticker=ticker,
        horizons=["1d", "5d"],
        model_types=["ridge"],
    )

    assert len(results) >= 2
    stmt = select(PredictionSnapshot).where(PredictionSnapshot.ticker == ticker)
    snapshots = list(db_session.exec(stmt).all())
    assert len(snapshots) == 2
