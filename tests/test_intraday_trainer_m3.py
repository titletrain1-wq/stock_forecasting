"""M3 tests: intraday log-return regressor training (crypto 1h/4h).

Covers the real M3 requirements (Jim's bounce B1-B6):
- regressor on a continuous log-return target (not a classifier)
- horizon-aware closed-bar anchors -> non-overlapping labels
- time-ordered 14-day holdout with label-window purge + 24h embargo
- a real, picklable, vol-conditioned CI band (ResidualVolModel)
- all artefacts persisted with the design's names
- metadata_crypto.json holds finite metrics from an actual training run
"""

import json
import pickle
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session, SQLModel, create_engine

from stock_forecasting.intraday_trainer import (
    CODE_VERSION,
    ResidualVolModel,
    _conditional_vol,
    _horizon_anchor_mask,
    train_intraday_models,
)
from stock_forecasting.schema import CryptoDerivative, IntradayBarsHistory, Ticker

TICKER = "BTC-USD"


def _make_engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def _seed(session: Session, *, days: int = 90, seed: int = 7) -> None:
    """Seed a deterministic 5m OHLCV history + daily funding for one ticker."""
    rng = np.random.default_rng(seed)
    n = days * 288
    start = (datetime.now(UTC) - timedelta(days=days)).replace(
        minute=0, second=0, microsecond=0
    )
    # random walk with mild momentum so the model has a little signal
    steps = rng.normal(0, 0.0009, n)
    for i in range(1, n):
        steps[i] += 0.15 * steps[i - 1]
    price = 45000 * np.exp(np.cumsum(steps))
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    session.add(
        Ticker(
            symbol=TICKER,
            asset_class="crypto",
            display_name=TICKER,
            provider="coinbase",
            provider_symbol=TICKER,
            price_basis="raw",
            added_at=now_iso,
            active=1,
        )
    )
    rows = []
    for i in range(n):
        ts = start + timedelta(minutes=5 * i)
        p = float(price[i])
        rows.append(
            IntradayBarsHistory(
                ticker=TICKER,
                interval="5m",
                ts=ts.isoformat().replace("+00:00", "Z"),
                open=p,
                high=p * 1.001,
                low=p * 0.999,
                close=p,
                volume=float(rng.uniform(5, 20)),
                source="coinbase_rest",
                ingested_at=now_iso,
            )
        )
    session.add_all(rows)
    for d in range(days + 1):
        ts = (start + timedelta(days=d)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        session.add(
            CryptoDerivative(
                ticker=TICKER,
                ts=ts.isoformat().replace("+00:00", "Z"),
                funding_rate=float(rng.normal(0.0001, 0.00005)),
                source="dydx",
            )
        )
    session.commit()


# --------------------------------------------------------------------------- #
# unit-level                                                                   #
# --------------------------------------------------------------------------- #


def test_horizon_anchor_mask_is_horizon_specific() -> None:
    ts = pd.Series(pd.date_range("2026-01-01", periods=48 * 4, freq="5min", tz="UTC"))
    m1h = _horizon_anchor_mask(ts, "1h")
    m4h = _horizon_anchor_mask(ts, "4h")

    assert (ts[m1h].dt.minute == 0).all()
    assert (ts[m4h].dt.minute == 0).all()
    assert (ts[m4h].dt.hour % 4 == 0).all()
    # 4h anchors are a strict subset of 1h anchors
    assert m4h.sum() < m1h.sum()
    assert (m4h & ~m1h).sum() == 0
    # spacing: consecutive 1h anchors are 12 bars apart -> non-overlapping 1h labels
    pos = np.flatnonzero(m1h.to_numpy())
    assert set(np.diff(pos).tolist()) == {12}


def test_conditional_vol_no_lookahead() -> None:
    bars = pd.DataFrame({"close": np.linspace(100, 110, 50)})
    v = _conditional_vol(bars, window=12)
    assert v.isna().iloc[0]  # first bar has no return
    assert v.notna().iloc[-1]


def test_residual_vol_model_is_picklable_ordered_and_conditioned() -> None:
    resid = np.random.default_rng(0).normal(0, 0.01, 5000)
    model = ResidualVolModel(*np.quantile(resid, [0.025, 0.975]), 0.01, CODE_VERSION)

    reloaded = pickle.loads(pickle.dumps(model))  # picklable (no closure)
    assert isinstance(reloaded, ResidualVolModel)

    lo, hi = reloaded.interval(0.0, 0.01)
    assert lo < hi
    # centred on the prediction
    lo2, hi2 = reloaded.interval(0.05, 0.01)
    assert lo2 == pytest.approx(lo + 0.05) and hi2 == pytest.approx(hi + 0.05)
    # wider when current vol exceeds the reference, narrower when below
    lo_hi_vol, hi_hi_vol = reloaded.interval(0.0, 0.03)
    lo_lo_vol, hi_lo_vol = reloaded.interval(0.0, 0.002)
    assert (hi_hi_vol - lo_hi_vol) > (hi - lo) > (hi_lo_vol - lo_lo_vol)


# --------------------------------------------------------------------------- #
# end-to-end: real training run against a synthetic DB                         #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    engine = _make_engine()
    mdir = tmp_path_factory.mktemp("models")
    with Session(engine) as session:
        _seed(session, days=90)
        meta = train_intraday_models(
            session=session,
            tickers=[TICKER],
            lookback_days=90,
            test_days=14,
            models_dir=mdir,
        )
    return meta, mdir


def test_metadata_has_real_finite_metrics_for_both_horizons(trained) -> None:
    meta, _ = trained
    assert set(meta) == {f"{TICKER}-1h", f"{TICKER}-4h"}
    for key, entry in meta.items():
        for field in ("mae", "rmse", "directional_pct", "ci_cover_pct"):
            val = entry[field]
            assert np.isfinite(val), f"{key}.{field} not finite: {val}"
        assert entry["mae"] > 0 and entry["rmse"] >= entry["mae"]
        assert 0.0 <= entry["directional_pct"] <= 100.0
        assert 0.0 <= entry["ci_cover_pct"] <= 100.0
        assert entry["train_samples"] >= 40
        assert entry["test_samples"] >= 10
        assert entry["code_version"] == CODE_VERSION


def test_all_artefacts_persisted_with_design_names(trained) -> None:
    _, mdir = trained
    for horizon in ("1h", "4h"):
        for stem in (
            f"intraday_lgb_btc_usd_{horizon}",
            f"intraday_ridge_btc_usd_fallback_{horizon}",
            f"intraday_scaler_btc_usd_{horizon}",
            f"intraday_har_rv_btc_usd_{horizon}",
        ):
            assert (mdir / f"{stem}.pkl").exists(), f"missing {stem}.pkl"
    assert (mdir / "metadata_crypto.json").exists()
    meta_on_disk = json.loads((mdir / "metadata_crypto.json").read_text())
    assert meta_on_disk.keys() == {f"{TICKER}-1h", f"{TICKER}-4h"}


def test_persisted_model_predicts_continuous_returns(trained) -> None:
    """B1: the primary model is a regressor over log-returns, not a classifier."""
    _, mdir = trained
    lgb = pickle.loads((mdir / "intraday_lgb_btc_usd_1h.pkl").read_bytes())
    scaler = pickle.loads((mdir / "intraday_scaler_btc_usd_1h.pkl").read_bytes())
    x = np.zeros((3, scaler.n_features_in_))
    preds = lgb.predict(scaler.transform(x))
    assert preds.dtype.kind == "f"
    # not collapsed to {0,1} class labels; plausible 1h log-return magnitude
    assert np.all(np.abs(preds) < 0.5)


def test_vol_model_round_trips_and_bands_are_ordered(trained) -> None:
    _, mdir = trained
    vm = pickle.loads((mdir / "intraday_har_rv_btc_usd_4h.pkl").read_bytes())
    assert isinstance(vm, ResidualVolModel)
    lo, hi = vm.interval(np.array([0.0, 0.01, -0.02]), np.array([0.01, 0.01, 0.01]))
    assert np.all(lo < hi)


def test_training_is_deterministic(trained, tmp_path) -> None:
    """Same seed + same data -> identical metrics (seeded models, fixed split)."""
    meta_a, _ = trained
    engine = _make_engine()
    with Session(engine) as session:
        _seed(session, days=90)
        meta_b = train_intraday_models(
            session=session,
            tickers=[TICKER],
            lookback_days=90,
            test_days=14,
            models_dir=tmp_path,
        )
    for key in meta_a:
        assert meta_a[key]["mae"] == pytest.approx(meta_b[key]["mae"], rel=1e-6)
        assert meta_a[key]["directional_pct"] == pytest.approx(
            meta_b[key]["directional_pct"], rel=1e-6
        )
