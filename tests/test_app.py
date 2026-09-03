"""Smoke tests for the Streamlit app module.

Streamlit is stubbed so the module imports without a script-run context.
Plotly is a real dependency and imports normally.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

_st = MagicMock()
_st.columns.side_effect = lambda spec: [
    MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
]
_st.sidebar.columns.side_effect = _st.columns.side_effect
sys.modules["streamlit"] = _st


def test_app_imports_cleanly() -> None:
    from stock_forecasting import app

    assert hasattr(app, "main")
    for fn in (
        "render_chart_panel",
        "render_accuracy_panel",
        "render_explain_panel",
        "render_health_panel",
    ):
        assert hasattr(app, fn)
    # streamlit-lightweight-charts must be gone
    assert "streamlit_lightweight_charts" not in sys.modules


def test_load_helpers_query_without_error(temp_db) -> None:
    from stock_forecasting import app

    assert app.load_tickers(temp_db) == []
    assert app.load_bars(temp_db, "AAPL", 90) == []
    assert app.load_snapshots(temp_db, "AAPL") == []


def test_add_ticker_inserts_once(temp_db) -> None:
    from stock_forecasting import app

    app.add_ticker(temp_db, "aapl")
    app.add_ticker(temp_db, "AAPL")
    tickers = app.load_tickers(temp_db)
    assert [t.symbol for t in tickers] == ["AAPL"]


def test_add_ticker_detects_crypto_and_equity(temp_db) -> None:
    from stock_forecasting import app

    app.add_ticker(temp_db, "SOL-USD")
    app.add_ticker(temp_db, "AAPL")
    tickers = {t.symbol: t for t in app.load_tickers(temp_db)}

    assert tickers["SOL-USD"].asset_class == "crypto"
    assert tickers["SOL-USD"].provider == "coinbase"
    assert tickers["SOL-USD"].price_basis == "raw"

    assert tickers["AAPL"].asset_class == "equity"
    assert tickers["AAPL"].provider == "yfinance"
    assert tickers["AAPL"].price_basis == "adjusted"


def test_render_price_header_handles_empty_and_populated() -> None:
    from stock_forecasting import app

    app.render_price_header("AAPL", [])
    bars = [
        SimpleNamespace(ts="2026-02-01T00:00:00Z", close=100.0),
        SimpleNamespace(ts="2026-02-02T00:00:00Z", close=110.0),
    ]
    app.render_price_header("AAPL", bars)

    # A live quote overrides the last close in the price tile.
    captured: list = []
    cols = [
        SimpleNamespace(metric=lambda *a, **k: captured.append(a)) for _ in range(3)
    ]
    _st.columns.side_effect = lambda spec: cols
    quotes = [SimpleNamespace(ts="2026-02-02T15:30:00Z", price=123.45)]
    app.render_price_header("AAPL", bars, quotes)
    _st.columns.side_effect = lambda spec: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    assert any("$123.45" in a[1] for a in captured)


def test_load_accuracy_records_scoped(temp_db) -> None:
    from sqlmodel import Session

    from stock_forecasting import app
    from stock_forecasting.schema import AccuracyRecord

    with Session(temp_db) as s:
        s.add(
            AccuracyRecord(
                scope="ticker",
                ticker="AAPL",
                horizon="1d",
                model_type="ridge",
                n=140,
                mae=1.0,
                rmse=2.0,
                dir_acc=0.58,
                ci_coverage=0.95,
                mae_price_pct=0.02,
                window="all",
                is_trustworthy=1,
                updated_at="2026-02-01T00:00:00Z",
            )
        )
        s.add(
            AccuracyRecord(
                scope="global",
                ticker=None,
                horizon="1d",
                model_type="ridge",
                n=500,
                mae=1.5,
                rmse=2.5,
                dir_acc=0.53,
                ci_coverage=0.9,
                mae_price_pct=0.03,
                window="all",
                is_trustworthy=0,
                updated_at="2026-02-01T00:00:00Z",
            )
        )
        s.commit()

    t = app.load_accuracy_records(temp_db, "AAPL", scope="ticker")
    assert [r.n for r in t] == [140]
    g = app.load_accuracy_records(temp_db, "AAPL", scope="global")
    assert [r.n for r in g] == [500]


def test_render_panels_no_crash_on_empty_db(temp_db) -> None:
    from stock_forecasting import app

    app.render_accuracy_panel(temp_db, "AAPL")
    app.render_explain_panel(temp_db, "AAPL")
    app.render_health_panel(temp_db)


def test_refresh_for_cadence_per_asset_class() -> None:
    from stock_forecasting import app

    assert app._refresh_for("crypto") == 2
    assert app._refresh_for("equity") == 15


def test_delayed_badge_only_for_equity() -> None:
    from stock_forecasting import app

    assert "15-min delayed" in app._delayed_badge("equity")
    assert app._delayed_badge("crypto") == ""


def test_load_live_returns_empty_on_empty_db(temp_db) -> None:
    from stock_forecasting import app

    quotes, intraday = app.load_live(temp_db, "BTC-USD", "crypto")
    assert quotes == []
    assert intraday == []
