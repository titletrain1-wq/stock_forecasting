"""M5 tests: intraday forecast evaluator (grading against realized closed bars)."""

import math
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, SQLModel, create_engine, select

from stock_forecasting.intraday_evaluator import grade_intraday_forecasts
from stock_forecasting.schema import (
    IntradayAccuracyRecord,
    IntradayBarsHistory,
    IntradayPredictionSnapshot,
)

TICKER = "BTC-USD"


def _engine():
    e = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(e)
    return e


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _snapshot(session, *, anchor_price, predicted_return, target_ts, ci=(-0.05, 0.05)):
    snap = IntradayPredictionSnapshot(
        ticker=TICKER,
        horizon="1h",
        made_at=_iso(datetime.now(UTC)),
        anchor_ts=_iso(datetime.now(UTC) - timedelta(hours=1)),
        anchor_price=anchor_price,
        predicted_return=predicted_return,
        predicted_price=anchor_price * math.exp(predicted_return),
        ci_lower_return=ci[0],
        ci_upper_return=ci[1],
        ci_lower_price=anchor_price * math.exp(ci[0]),
        ci_upper_price=anchor_price * math.exp(ci[1]),
        target_ts=target_ts,
        model_version="m3-1",
        model_sha="abc",
    )
    session.add(snap)
    session.commit()
    session.refresh(snap)
    return snap


def _bar(session, ts: str, close: float):
    session.add(
        IntradayBarsHistory(
            ticker=TICKER,
            interval="5m",
            ts=ts,
            open=close,
            high=close,
            low=close,
            close=close,
            source="coinbase_rest",
            ingested_at=_iso(datetime.now(UTC)),
        )
    )
    session.commit()


def test_matured_forecast_is_graded_with_correct_fields():
    e = _engine()
    now = datetime.now(UTC)
    target = _iso(now - timedelta(minutes=30))
    with Session(e) as s:
        snap = _snapshot(s, anchor_price=100.0, predicted_return=0.02, target_ts=target)
        _bar(s, target, close=103.0)  # realized up +3%

        summary = grade_intraday_forecasts(s, now=now)
        assert summary["graded"] == 1

        rec = s.exec(
            select(IntradayAccuracyRecord).where(
                IntradayAccuracyRecord.prediction_id == snap.id
            )
        ).one()
        assert rec.realized_price == 103.0
        assert rec.realized_return == math.log(103.0 / 100.0)
        assert rec.signed_error == rec.realized_return - 0.02
        assert rec.abs_error_pct == abs(rec.signed_error) * 100
        assert rec.direction_hit == 1  # both positive
        assert rec.ci_cover == 1  # ln(1.03) ~ 0.0296 within [-0.05, 0.05]


def test_direction_miss_and_ci_breach_recorded():
    e = _engine()
    now = datetime.now(UTC)
    target = _iso(now - timedelta(minutes=10))
    with Session(e) as s:
        snap = _snapshot(
            s,
            anchor_price=100.0,
            predicted_return=0.01,
            target_ts=target,
            ci=(-0.005, 0.005),
        )
        _bar(s, target, close=94.0)  # realized down -6%, outside CI, wrong direction
        grade_intraday_forecasts(s, now=now)
        rec = s.exec(
            select(IntradayAccuracyRecord).where(
                IntradayAccuracyRecord.prediction_id == snap.id
            )
        ).one()
        assert rec.direction_hit == 0
        assert rec.ci_cover == 0


def test_immature_forecast_is_deferred_then_graded_later():
    e = _engine()
    now = datetime.now(UTC)
    target = _iso(now - timedelta(minutes=5))
    with Session(e) as s:
        snap = _snapshot(s, anchor_price=100.0, predicted_return=0.0, target_ts=target)
        # no bar at target yet
        summary = grade_intraday_forecasts(s, now=now)
        assert summary == {"graded": 0, "deferred": 1, "checked": 1}
        rec = s.exec(
            select(IntradayAccuracyRecord).where(
                IntradayAccuracyRecord.prediction_id == snap.id
            )
        ).one()
        assert rec.grading_attempts == 1
        assert rec.realized_return is None

        _bar(s, target, close=101.0)  # bar lands
        grade_intraday_forecasts(s, now=now)
        s.refresh(rec)
        assert rec.realized_return is not None
        assert rec.grading_attempts == 2


def test_future_forecast_not_touched():
    e = _engine()
    now = datetime.now(UTC)
    with Session(e) as s:
        _snapshot(
            s,
            anchor_price=100.0,
            predicted_return=0.0,
            target_ts=_iso(now + timedelta(hours=1)),
        )
        summary = grade_intraday_forecasts(s, now=now)
        assert summary["checked"] == 0
        assert s.exec(select(IntradayAccuracyRecord)).all() == []


def test_grading_is_idempotent():
    e = _engine()
    now = datetime.now(UTC)
    target = _iso(now - timedelta(minutes=30))
    with Session(e) as s:
        _snapshot(s, anchor_price=100.0, predicted_return=0.01, target_ts=target)
        _bar(s, target, close=101.0)
        grade_intraday_forecasts(s, now=now)
        grade_intraday_forecasts(s, now=now)  # rerun
        recs = s.exec(select(IntradayAccuracyRecord)).all()
        assert len(recs) == 1
        assert recs[0].grading_attempts == 1  # not re-incremented once graded
