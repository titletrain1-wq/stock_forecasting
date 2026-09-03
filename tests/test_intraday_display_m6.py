"""M6 tests: intraday forecast display (viz figure + app loader)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlmodel import Session, SQLModel, create_engine

from stock_forecasting.app import load_intraday_forecasts
from stock_forecasting.schema import IntradayPredictionSnapshot
from stock_forecasting.viz import build_intraday_forecast_figure


def _fc(horizon, anchor_price, pred, lo, hi, target):
    return SimpleNamespace(
        horizon=horizon,
        anchor_ts="2026-09-04T12:00:00Z",
        anchor_price=anchor_price,
        predicted_price=pred,
        ci_lower_price=lo,
        ci_upper_price=hi,
        target_ts=target,
    )


def test_figure_draws_price_and_each_forecast():
    bars = [
        SimpleNamespace(ts=f"2026-09-04T{h:02d}:00:00Z", close=45000 + h * 10)
        for h in range(10)
    ]
    forecasts = [
        _fc("1h", 45100, 45250, 45050, 45450, "2026-09-04T13:00:00Z"),
        _fc("4h", 45100, 45500, 45000, 46000, "2026-09-04T16:00:00Z"),
    ]
    fig = build_intraday_forecast_figure(bars, forecasts)
    names = [t.name for t in fig.data if t.name]
    assert "price (5m)" in names
    assert "1h forecast" in names and "4h forecast" in names

    fc_1h = next(t for t in fig.data if t.name == "1h forecast")
    assert list(fc_1h.y) == [45100, 45250]  # anchor -> predicted
    # CI band: two zero-width boundary traces per horizon with a fill
    assert any(getattr(t, "fill", None) == "tonexty" for t in fig.data)


def test_figure_handles_empty():
    fig = build_intraday_forecast_figure([], [])
    assert fig.layout.annotations  # shows the "no forecast yet" note


def test_load_intraday_forecasts_returns_latest_per_horizon():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    now = datetime.now(UTC)
    with Session(engine) as session:
        for age_min, pred in (
            (90, 100.0),
            (10, 200.0),
        ):  # older then newer, same horizon
            made = (now - timedelta(minutes=age_min)).isoformat().replace("+00:00", "Z")
            session.add(
                IntradayPredictionSnapshot(
                    ticker="BTC-USD",
                    horizon="1h",
                    made_at=made,
                    anchor_ts=made,
                    anchor_price=100.0,
                    predicted_return=0.01,
                    predicted_price=pred,
                    ci_lower_return=-0.01,
                    ci_upper_return=0.03,
                    ci_lower_price=pred * 0.99,
                    ci_upper_price=pred * 1.03,
                    target_ts=(now + timedelta(hours=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    model_version="m3-1",
                    model_sha="abc",
                )
            )
        session.commit()

    latest = load_intraday_forecasts(engine, "BTC-USD")
    assert len(latest) == 1
    assert latest[0].predicted_price == 200.0  # the newer row wins


def test_end_to_end_train_write_display(tmp_path):
    """M3 -> M4 -> M6: real models -> real snapshots -> a renderable figure."""
    from stock_forecasting.intraday_forecaster import write_intraday_forecasts
    from stock_forecasting.intraday_trainer import train_intraday_models
    from tests.test_intraday_trainer_m3 import TICKER, _make_engine, _seed

    engine = _make_engine()
    session = Session(engine)
    _seed(session, days=120)
    train_intraday_models(
        session=session,
        tickers=[TICKER],
        lookback_days=120,
        test_days=14,
        models_dir=tmp_path,
    )
    write_intraday_forecasts(session, tickers=(TICKER,), models_dir=tmp_path)

    forecasts = load_intraday_forecasts(engine, TICKER)
    assert {f.horizon for f in forecasts} == {"15m", "1h", "4h"}
    fig = build_intraday_forecast_figure([], forecasts)
    assert any(t.name == "1h forecast" for t in fig.data)
    session.close()
