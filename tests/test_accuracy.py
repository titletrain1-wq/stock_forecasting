"""Unit and integration tests for AccuracyService and trust verdict calculation."""

import json
from datetime import UTC, datetime

import numpy as np
from sqlmodel import Session, select

from stock_forecasting.accuracy import AccuracyService
from stock_forecasting.schema import AccuracyRecord, ModelRun, PredictionSnapshot


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


def _seed_evaluated_snapshots(
    session: Session,
    ticker: str,
    horizon: str,
    model_type: str,
    model_run_id: int,
    n: int,
    dir_hit_ratio: float = 0.60,
    ci_hit_ratio: float = 0.90,
) -> list[PredictionSnapshot]:
    """Helper to seed n evaluated prediction snapshots."""
    snapshots: list[PredictionSnapshot] = []
    now_iso = datetime.now(UTC).isoformat()

    for i in range(n):
        # Determine if direction hit
        is_dir_hit = 1 if (i / n) < dir_hit_ratio else 0
        is_ci_hit = 1 if (i / n) < ci_hit_ratio else 0

        pred_ret = 0.02
        realized_ret = 0.02 if is_dir_hit else -0.02

        anchor_price = 100.0
        pred_price = anchor_price * np.exp(pred_ret)
        realized_price = anchor_price * np.exp(realized_ret)

        err_abs = float(abs(pred_ret - realized_ret))
        err_signed = float(pred_ret - realized_ret)

        snap = PredictionSnapshot(
            prediction_id=f"snap-{ticker}-{horizon}-{model_type}-{i}",
            ticker=ticker,
            made_at="2023-01-01T00:00:00+00:00",
            made_from_ts="2023-01-01T00:00:00+00:00",
            anchor_price=anchor_price,
            horizon=horizon,
            target_ts="2023-01-02T00:00:00+00:00",
            predicted_return=pred_ret,
            predicted_price=pred_price,
            lower_bound=95.0 if is_ci_hit else 110.0,
            upper_bound=105.0 if is_ci_hit else 115.0,
            model_type=model_type,
            model_version="1.0.0",
            model_run_id=model_run_id,
            explain_json="{}",
            input_is_stale=0,
            realized_price=realized_price,
            realized_return=realized_ret,
            evaluated_at=now_iso,
            error_abs=err_abs,
            error_signed=err_signed,
            is_direction_hit=is_dir_hit,
            is_within_ci=is_ci_hit,
            eval_attempts=0,
        )
        session.add(snap)
        snapshots.append(snap)

    session.commit()
    return snapshots


def test_accuracy_rebuild_and_verdict(db_session: Session) -> None:
    """Test AccuracyService computes aggregates and evaluates is_trustworthy threshold (dir_acc >= 0.55 and n >= 30)."""
    model_run = _create_sample_model_run(db_session, ticker="AAPL")

    # Group 1: AAPL (n=35, dir_acc=60%) -> Should be trustworthy (n >= 30 and dir_acc >= 0.55)
    _seed_evaluated_snapshots(
        db_session,
        ticker="AAPL",
        horizon="1d",
        model_type="ridge",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        n=35,
        dir_hit_ratio=0.60,
    )

    # Group 2: MSFT (n=10, dir_acc=80%) -> Should NOT be trustworthy (n < 30)
    _seed_evaluated_snapshots(
        db_session,
        ticker="MSFT",
        horizon="1d",
        model_type="ridge",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        n=10,
        dir_hit_ratio=0.80,
    )

    # Group 3: NVDA (n=40, dir_acc=45%) -> Should NOT be trustworthy (dir_acc < 0.55)
    _seed_evaluated_snapshots(
        db_session,
        ticker="NVDA",
        horizon="1d",
        model_type="ridge",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        n=40,
        dir_hit_ratio=0.45,
    )

    records = AccuracyService.rebuild_aggregates(db_session)
    assert len(records) > 0

    # 1. Check AAPL ticker record
    aapl_rec = db_session.exec(
        select(AccuracyRecord).where(
            AccuracyRecord.scope == "ticker",
            AccuracyRecord.ticker == "AAPL",
            AccuracyRecord.horizon == "1d",
            AccuracyRecord.model_type == "ridge",
        )
    ).first()
    assert aapl_rec is not None
    assert aapl_rec.n == 35
    assert aapl_rec.dir_acc >= 0.55
    assert aapl_rec.is_trustworthy == 1
    assert aapl_rec.mae > 0
    assert aapl_rec.rmse > 0
    assert 0.0 <= aapl_rec.ci_coverage <= 1.0
    assert aapl_rec.mae_price_pct >= 0

    # 2. Check MSFT ticker record (n=10 < 30)
    msft_rec = db_session.exec(
        select(AccuracyRecord).where(
            AccuracyRecord.scope == "ticker",
            AccuracyRecord.ticker == "MSFT",
            AccuracyRecord.horizon == "1d",
            AccuracyRecord.model_type == "ridge",
        )
    ).first()
    assert msft_rec is not None
    assert msft_rec.n == 10
    assert msft_rec.dir_acc == 0.80
    assert msft_rec.is_trustworthy == 0  # not enough samples

    # 3. Check NVDA ticker record (dir_acc < 0.55)
    nvda_rec = db_session.exec(
        select(AccuracyRecord).where(
            AccuracyRecord.scope == "ticker",
            AccuracyRecord.ticker == "NVDA",
            AccuracyRecord.horizon == "1d",
            AccuracyRecord.model_type == "ridge",
        )
    ).first()
    assert nvda_rec is not None
    assert nvda_rec.n == 40
    assert nvda_rec.dir_acc == 0.45
    assert nvda_rec.is_trustworthy == 0  # accuracy below threshold

    # 4. Check global record (scope="global", ticker=None)
    global_rec = db_session.exec(
        select(AccuracyRecord).where(
            AccuracyRecord.scope == "global",
            AccuracyRecord.ticker.is_(None),
            AccuracyRecord.horizon == "1d",
            AccuracyRecord.model_type == "ridge",
        )
    ).first()
    assert global_rec is not None
    assert global_rec.n == 35 + 10 + 40  # 85 total
    assert global_rec.ticker is None
    assert global_rec.scope == "global"


def test_accuracy_rebuild_empty_snapshots(db_session: Session) -> None:
    """Test AccuracyService gracefully handles empty snapshots table."""
    records = AccuracyService.rebuild_aggregates(db_session)
    assert records == []


def test_accuracy_rebuild_upsert_idempotence(db_session: Session) -> None:
    """Test that multiple rebuilds update records in-place without duplicating rows."""
    model_run = _create_sample_model_run(db_session, ticker="GOOG")

    _seed_evaluated_snapshots(
        db_session,
        ticker="GOOG",
        horizon="1d",
        model_type="ridge",
        model_run_id=model_run.id,  # type: ignore[arg-type]
        n=5,
    )

    recs1 = AccuracyService.rebuild_aggregates(db_session)
    count1 = len(db_session.exec(select(AccuracyRecord)).all())

    # Rebuild again
    recs2 = AccuracyService.rebuild_aggregates(db_session)
    count2 = len(db_session.exec(select(AccuracyRecord)).all())

    assert count1 == count2
    assert len(recs1) == len(recs2)
