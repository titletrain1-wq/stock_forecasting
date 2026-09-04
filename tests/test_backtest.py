"""Tests for the walk-forward backtest (T-019).

These exercise the real forecast+grade path against a trained model, and pin the
three properties that make the backtest trustworthy:
  1. it produces graded forecasts (n > 0), not an empty smoke result;
  2. no lookahead — a forecast made as-of date T only consumes bars with ts <= T;
  3. isolation — it never writes to prediction_snapshots / accuracy_records.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from stock_forecasting.backtest import BacktestResult, BacktestService
from stock_forecasting.bar_store import BarRepository
from stock_forecasting.providers.base import Bar
from stock_forecasting.schema import AccuracyRecord, PredictionSnapshot, Ticker
from stock_forecasting.trainer import Trainer


def _seed_bars(session: Session, ticker: str, n: int = 420, seed: int = 7) -> None:
    """Insert deterministic synthetic daily bars (geometric random walk)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    log_returns = rng.normal(0.0004, 0.014, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(rng.normal(0.004, 0.003, n)))
    low = close * (1.0 - np.abs(rng.normal(0.004, 0.003, n)))
    open_ = (high + low) / 2.0
    volume = rng.lognormal(10.0, 0.3, n)
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
    BarRepository(session).upsert_bars(ticker=ticker, bars=bars, source="test")


def _seed_ticker(session: Session, symbol: str) -> None:
    if not session.exec(select(Ticker).where(Ticker.symbol == symbol)).first():
        session.add(
            Ticker(
                symbol=symbol,
                asset_class="equity",
                display_name=symbol,
                provider="test",
                provider_symbol=symbol,
                price_basis="adjusted",
                added_at="2023-01-01T00:00:00+00:00",
            )
        )
        session.commit()


def _trained_service(
    session: Session, tmp_path: Path, ticker: str, horizons: tuple[str, ...]
) -> BacktestService:
    _seed_ticker(session, ticker)
    _seed_bars(session, ticker)
    trainer = Trainer(session=session, model_dir=tmp_path)
    for h in horizons:
        trainer.train(ticker=ticker, horizon=h, model_type="ridge")
    return BacktestService(session, model_dir=str(tmp_path))


def test_backtest_produces_graded_forecasts(db_session, tmp_path):
    """A real model + 400 bars -> many graded forecasts per horizon, valid ranges."""
    svc = _trained_service(db_session, tmp_path, "BTQ", ("1d", "5d"))
    results = svc.run_backtest("BTQ", horizons=("1d", "5d"), lookback_days=90)

    for h in ("1d", "5d"):
        r = results[h]
        assert isinstance(r, BacktestResult)
        assert r.n >= 20, f"{h}: expected many graded forecasts, got n={r.n}"
        assert r.mae > 0.0
        assert 0.0 <= r.dir_acc <= 1.0
        assert 0.0 <= r.ci_coverage <= 1.0
        assert r.mae_price_pct > 0.0


def test_backtest_no_lookahead(db_session, tmp_path, monkeypatch):
    """Every bar the forecaster sees during the backtest has ts <= the as-of cutoff."""
    svc = _trained_service(db_session, tmp_path, "NLQ", ("1d",))
    real_get_up_to = BarRepository.get_up_to
    seen: list[tuple[str, str]] = []

    def _spy(self, ticker, cutoff_ts, limit=None, interval="1d"):
        rows = real_get_up_to(self, ticker, cutoff_ts, limit=limit, interval=interval)
        for b in rows:
            seen.append((b.ts, cutoff_ts))
        return rows

    monkeypatch.setattr(BarRepository, "get_up_to", _spy)
    svc.run_backtest("NLQ", horizons=("1d",), lookback_days=60)

    assert seen, "backtest never fetched bars via get_up_to"
    assert all(bar_ts <= cutoff for bar_ts, cutoff in seen), (
        "lookahead: a bar with ts > cutoff was fed to the forecaster"
    )


def test_backtest_is_isolated_from_live_tables(db_session, tmp_path):
    """Backtest writes nothing to prediction_snapshots or accuracy_records."""
    svc = _trained_service(db_session, tmp_path, "ISQ", ("1d",))
    before_snaps = len(db_session.exec(select(PredictionSnapshot)).all())
    before_acc = len(db_session.exec(select(AccuracyRecord)).all())

    svc.run_backtest("ISQ", horizons=("1d",), lookback_days=60)

    assert len(db_session.exec(select(PredictionSnapshot)).all()) == before_snaps
    assert len(db_session.exec(select(AccuracyRecord)).all()) == before_acc


def test_backtest_deterministic(db_session, tmp_path):
    """Same data + same model -> identical metrics across two runs."""
    svc = _trained_service(db_session, tmp_path, "DTQ", ("1d",))
    a = svc.run_backtest("DTQ", horizons=("1d",), lookback_days=60)["1d"]
    b = svc.run_backtest("DTQ", horizons=("1d",), lookback_days=60)["1d"]
    assert (a.n, a.mae, a.dir_acc, a.ci_coverage) == (
        b.n,
        b.mae,
        b.dir_acc,
        b.ci_coverage,
    )


def test_backtest_no_bars_returns_zeroed_results(db_session):
    """No bars at all -> a full result dict, every horizon n=0 (no crash)."""
    results = BacktestService(db_session).run_backtest(
        "NOPE", horizons=("1d", "5d", "30d"), lookback_days=180
    )
    assert set(results) == {"1d", "5d", "30d"}
    for h in ("1d", "5d", "30d"):
        assert results[h].n == 0
        assert results[h].mae == 0.0


def test_backtest_lookback_widens_sample(db_session, tmp_path):
    """A longer lookback grades at least as many forecasts as a shorter one."""
    svc = _trained_service(db_session, tmp_path, "LBQ", ("1d",))
    short = svc.run_backtest("LBQ", horizons=("1d",), lookback_days=30)["1d"]
    long = svc.run_backtest("LBQ", horizons=("1d",), lookback_days=90)["1d"]
    assert long.n >= short.n > 0
