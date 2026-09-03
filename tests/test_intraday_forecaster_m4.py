"""M4 tests: intraday forecast writer (sole writer to intraday_prediction_snapshots)."""

import numpy as np
from sqlmodel import Session, SQLModel, create_engine, select

from stock_forecasting.intraday_forecaster import write_intraday_forecasts
from stock_forecasting.intraday_trainer import train_intraday_models
from stock_forecasting.schema import IntradayPredictionSnapshot
from tests.test_intraday_trainer_m3 import TICKER, _seed


def _prepared(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    _seed(session, days=120)
    train_intraday_models(
        session=session,
        tickers=[TICKER],
        lookback_days=120,
        test_days=14,
        models_dir=tmp_path,
    )
    return session, tmp_path


def test_writer_creates_one_snapshot_per_horizon(tmp_path):
    session, mdir = _prepared(tmp_path)
    written = write_intraday_forecasts(session, tickers=(TICKER,), models_dir=mdir)

    assert {w["horizon"] for w in written} == {"15m", "1h", "4h"}
    rows = session.exec(
        select(IntradayPredictionSnapshot).where(
            IntradayPredictionSnapshot.ticker == TICKER
        )
    ).all()
    assert len(rows) == 3

    for r in rows:
        assert np.isfinite(r.predicted_return) and abs(r.predicted_return) < 0.5
        assert r.ci_lower_return < r.predicted_return < r.ci_upper_return
        assert r.ci_lower_price < r.predicted_price < r.ci_upper_price
        assert (
            r.predicted_price
            == np.float64(r.anchor_price * np.exp(r.predicted_return)).item()
            or abs(r.predicted_price - r.anchor_price * np.exp(r.predicted_return))
            < 1e-6
        )
        assert r.target_ts > r.anchor_ts
        assert r.model_version


def test_writer_is_idempotent_on_same_anchor(tmp_path):
    session, mdir = _prepared(tmp_path)
    write_intraday_forecasts(session, tickers=(TICKER,), models_dir=mdir)
    write_intraday_forecasts(session, tickers=(TICKER,), models_dir=mdir)  # rerun

    rows = session.exec(
        select(IntradayPredictionSnapshot).where(
            IntradayPredictionSnapshot.ticker == TICKER
        )
    ).all()
    assert len(rows) == 3  # INSERT OR IGNORE dedup on (ticker, horizon, anchor_ts)


def test_writer_anchor_is_a_closed_bar_boundary(tmp_path):
    session, mdir = _prepared(tmp_path)
    written = write_intraday_forecasts(session, tickers=(TICKER,), models_dir=mdir)
    for w in written:
        assert w["anchor_ts"].endswith(":00:00Z") or "T" in w["anchor_ts"]
        # 4h anchors sit on a 4-hour boundary
        if w["horizon"] == "4h":
            hour = int(w["anchor_ts"][11:13])
            assert hour % 4 == 0


def test_writer_skips_when_models_absent(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _seed(session, days=30)
        written = write_intraday_forecasts(
            session, tickers=(TICKER,), models_dir=tmp_path / "empty"
        )
    assert written == []
