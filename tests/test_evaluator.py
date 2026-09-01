"""Unit and integration tests for EvaluatorService."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from sqlmodel import Session, select

from stock_forecasting.evaluator import EvaluatorService
from stock_forecasting.schema import (
    AccuracyRecord,
    ModelRun,
    OhlcvBar,
    PredictionSnapshot,
)


def _create_sample_model_run(session: Session, ticker: str = "AAPL") -> ModelRun:
    """Helper to create and persist a dummy ModelRun."""
    model_run = ModelRun(
        ticker=ticker,
        horizon="1d",
        model_type="ridge",
        model_version="1.0.0",
        code_git_sha="abcdef1",
        trained_at=datetime.now(UTC).isoformat(),
        train_start="2022-01-01T00:00:00+00:00",
        train_end="2022-12-31T00:00:00+00:00",
        hyperparams_json=json.dumps({"alpha": 1.0}),
        feature_list_json=json.dumps(["log_return_1d"]),
        random_seed=42,
        wf_mae=0.01,
        wf_rmse=0.015,
        wf_dir_acc=0.6,
        wf_ci_cov=0.95,
        residual_std=0.01,
        artifact_path="/tmp/model.joblib",
        is_active=1,
    )
    session.add(model_run)
    session.commit()
    session.refresh(model_run)
    return model_run


def test_evaluator_grades_matured_snapshot(db_session: Session) -> None:
    """Test that a matured prediction snapshot is correctly evaluated against realized price bar."""
    model_run = _create_sample_model_run(db_session, ticker="AAPL")

    # 1. Seed prediction snapshot
    snap = PredictionSnapshot(
        prediction_id="snap-100",
        ticker="AAPL",
        made_at="2023-01-01T00:00:00+00:00",
        made_from_ts="2023-01-01T00:00:00+00:00",
        anchor_price=100.0,
        horizon="1d",
        target_ts="2023-01-02T00:00:00+00:00",
        predicted_return=0.05,
        predicted_price=105.127,
        lower_bound=98.0,
        upper_bound=112.0,
        model_type="ridge",
        model_version="1.0.0",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        explain_json="{}",
        input_is_stale=0,
    )
    db_session.add(snap)

    # 2. Seed realized bar at target_ts
    realized_bar = OhlcvBar(
        ticker="AAPL",
        interval="1d",
        ts="2023-01-02T00:00:00+00:00",
        open=100.0,
        high=106.0,
        low=99.0,
        close=105.0,
        adj_close=105.0,
        volume=10000.0,
        source="test",
        ingested_at="2023-01-02T01:00:00+00:00",
    )
    db_session.add(realized_bar)
    db_session.commit()

    # 3. Run evaluator as of after target_ts
    evaluator = EvaluatorService()
    evaluated = evaluator.run(session=db_session, as_of="2023-01-03T00:00:00+00:00")

    assert len(evaluated) == 1
    db_session.refresh(snap)

    expected_return = np.log(105.0 / 100.0)
    assert snap.evaluated_at is not None
    assert snap.realized_price == 105.0
    assert snap.realized_return == pytest.approx(expected_return)
    assert snap.error_abs == pytest.approx(abs(0.05 - expected_return))
    assert snap.error_signed == pytest.approx(0.05 - expected_return)
    assert snap.is_direction_hit == 1
    assert snap.is_within_ci == 1

    # Verify accuracy record was automatically generated
    accuracy_records = list(
        db_session.exec(
            select(AccuracyRecord).where(AccuracyRecord.ticker == "AAPL")
        ).all()
    )
    assert len(accuracy_records) == 1
    rec = accuracy_records[0]
    assert rec.n == 1
    assert rec.mae == pytest.approx(snap.error_abs)
    assert rec.dir_acc == 1.0


def test_evaluator_unmatured_snapshot_ignored(db_session: Session) -> None:
    """Test that a future prediction snapshot whose target_ts > as_of is not evaluated."""
    model_run = _create_sample_model_run(db_session, ticker="AAPL")

    snap = PredictionSnapshot(
        prediction_id="snap-unmatured",
        ticker="AAPL",
        made_at="2023-01-01T00:00:00+00:00",
        made_from_ts="2023-01-01T00:00:00+00:00",
        anchor_price=100.0,
        horizon="5d",
        target_ts="2023-01-06T00:00:00+00:00",
        predicted_return=0.02,
        predicted_price=102.0,
        lower_bound=95.0,
        upper_bound=110.0,
        model_type="ridge",
        model_version="1.0.0",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        explain_json="{}",
        input_is_stale=0,
    )
    db_session.add(snap)
    db_session.commit()

    evaluator = EvaluatorService(db_session)
    evaluated = evaluator.run(as_of="2023-01-03T00:00:00+00:00")

    assert len(evaluated) == 0
    db_session.refresh(snap)
    assert snap.evaluated_at is None
    assert snap.realized_price is None


def test_evaluator_missing_realized_bar_increments_attempts(
    db_session: Session,
) -> None:
    """Test that if target_ts has matured but no matching bar is present, eval_attempts increments."""
    model_run = _create_sample_model_run(db_session, ticker="MSFT")

    snap = PredictionSnapshot(
        prediction_id="snap-missing-bar",
        ticker="MSFT",
        made_at="2023-01-01T00:00:00+00:00",
        made_from_ts="2023-01-01T00:00:00+00:00",
        anchor_price=200.0,
        horizon="1d",
        target_ts="2023-01-02T00:00:00+00:00",
        predicted_return=0.01,
        predicted_price=202.0,
        lower_bound=190.0,
        upper_bound=215.0,
        model_type="ridge",
        model_version="1.0.0",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        explain_json="{}",
        input_is_stale=0,
        eval_attempts=0,
    )
    db_session.add(snap)
    db_session.commit()

    evaluator = EvaluatorService(db_session)
    # Run once
    evaluated = evaluator.run(as_of="2023-01-03T00:00:00+00:00")
    assert len(evaluated) == 0
    db_session.refresh(snap)
    assert snap.evaluated_at is None
    assert snap.eval_attempts == 1

    # Run second time
    evaluator.run(as_of="2023-01-03T00:00:00+00:00")
    db_session.refresh(snap)
    assert snap.eval_attempts == 2


def test_evaluator_direction_miss_and_out_of_ci(db_session: Session) -> None:
    """Test metric computations when forecast misses direction and falls outside confidence interval."""
    model_run = _create_sample_model_run(db_session, ticker="TSLA")

    snap = PredictionSnapshot(
        prediction_id="snap-miss",
        ticker="TSLA",
        made_at="2023-01-01T00:00:00+00:00",
        made_from_ts="2023-01-01T00:00:00+00:00",
        anchor_price=200.0,
        horizon="1d",
        target_ts="2023-01-02T00:00:00+00:00",
        predicted_return=0.05,  # positive forecast
        predicted_price=210.34,
        lower_bound=195.0,
        upper_bound=225.0,
        model_type="ridge",
        model_version="1.0.0",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        explain_json="{}",
        input_is_stale=0,
    )
    db_session.add(snap)

    # Realized bar drops sharply to 180.0 (negative return and below lower_bound 195.0)
    realized_bar = OhlcvBar(
        ticker="TSLA",
        interval="1d",
        ts="2023-01-02T00:00:00+00:00",
        open=200.0,
        high=201.0,
        low=178.0,
        close=180.0,
        adj_close=180.0,
        volume=50000.0,
        source="test",
        ingested_at="2023-01-02T01:00:00+00:00",
    )
    db_session.add(realized_bar)
    db_session.commit()

    evaluator = EvaluatorService(db_session)
    evaluator.run(as_of="2023-01-03T00:00:00+00:00")

    db_session.refresh(snap)
    assert snap.evaluated_at is not None
    assert snap.realized_price == 180.0
    assert snap.is_direction_hit == 0
    assert snap.is_within_ci == 0
