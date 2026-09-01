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
    assert hasattr(app, "render_chart_panel")
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


def test_render_price_header_handles_empty_and_populated() -> None:
    from stock_forecasting import app

    app.render_price_header("AAPL", [])
    bars = [
        SimpleNamespace(ts="2026-02-01T00:00:00Z", close=100.0),
        SimpleNamespace(ts="2026-02-02T00:00:00Z", close=110.0),
    ]
    app.render_price_header("AAPL", bars)
