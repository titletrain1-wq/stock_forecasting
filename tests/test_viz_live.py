"""Tests for the M5 live-streaming chart additions in stock_forecasting.viz."""

from types import SimpleNamespace

import plotly.graph_objects as go

from stock_forecasting.viz import add_live_price_line, build_price_figure


def _ibar(ts: str, close: float, provisional: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        ts=ts,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1.0,
        is_provisional=provisional,
    )


def _quote(price: float, ts: str = "2026-09-01T14:00:30Z") -> SimpleNamespace:
    return SimpleNamespace(price=price, ts=ts, source="coinbase_ws")


def test_add_live_price_line_appends_live_trace_ending_at_quote() -> None:
    bars = [
        _ibar("2026-09-01T13:57:00Z", 100.0),
        _ibar("2026-09-01T13:59:00Z", 102.0),
        _ibar("2026-09-01T13:58:00Z", 101.0),
    ]
    out = add_live_price_line(go.Figure(), [_quote(103.5)], bars)
    live = next(t for t in out.data if t.name == "live")
    assert list(live.x) == [
        "2026-09-01T13:57:00Z",
        "2026-09-01T13:58:00Z",
        "2026-09-01T13:59:00Z",
        "2026-09-01T14:00:30Z",
    ]
    assert live.y[-1] == 103.5


def test_forming_candle_trace_is_provisional_and_faded() -> None:
    bars = [_ibar("2026-09-01T13:59:00Z", 102.0, provisional=1)]
    out = add_live_price_line(go.Figure(), [_quote(103.0)], bars)
    forming = next(t for t in out.data if t.name == "forming")
    assert forming.opacity < 1
    assert "provisional" in forming.hovertext.lower()
    # latest tick moves the forming close
    assert forming.close[0] == 103.0


def test_add_live_price_line_is_noop_without_any_data() -> None:
    out = add_live_price_line(go.Figure(), [], [])
    assert not any(t.name in ("live", "forming") for t in out.data)


def test_build_price_figure_sets_uirevision_for_fragment_refreshes() -> None:
    fig = build_price_figure([], [])
    assert fig.layout.uirevision is True
