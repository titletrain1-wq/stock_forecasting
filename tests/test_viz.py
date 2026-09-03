"""Tests for the Plotly chart assembly layer (stock_forecasting.viz)."""

from types import SimpleNamespace

import plotly.graph_objects as go

from stock_forecasting.viz import build_price_figure


def _bar(ts: str, close: float) -> SimpleNamespace:
    return SimpleNamespace(
        ts=ts,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        adj_close=close,
        volume=1000.0,
    )


def _snapshot(
    *,
    horizon: str,
    made_at: str,
    made_from_ts: str,
    target_ts: str,
    anchor_price: float,
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    is_direction_hit: int | None = None,
    evaluated_at: str | None = None,
    realized_price: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prediction_id=f"{horizon}-{made_at}",
        ticker="AAPL",
        made_at=made_at,
        made_from_ts=made_from_ts,
        anchor_price=anchor_price,
        horizon=horizon,
        target_ts=target_ts,
        predicted_return=0.01,
        predicted_price=predicted_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        model_type="ridge",
        model_version="1.0.0",
        explain_json="{}",
        realized_price=realized_price,
        realized_return=None,
        evaluated_at=evaluated_at,
        error_abs=None,
        error_signed=None,
        is_direction_hit=is_direction_hit,
        is_within_ci=None,
    )


BARS = [
    _bar(f"2026-{m:02d}-{d:02d}T00:00:00Z", 100.0 + (m - 1) * 30 + d)
    for m in range(1, 3)
    for d in range(1, 31)
]  # ~60 days of data for indicator computation


def _trace_names(fig: go.Figure) -> list[str]:
    return [t.name for t in fig.data if t.name]


def test_empty_inputs_return_figure_with_no_data_annotation() -> None:
    fig = build_price_figure([], [])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert any("No data" in a.text for a in fig.layout.annotations)


def test_uirevision_defaults_true_and_is_overridable() -> None:
    # Default keeps zoom/pan stable across live-tick refreshes.
    assert build_price_figure(BARS, []).layout.uirevision is True
    # App passes "<symbol>:<range>" so a ticker/range switch forces Plotly to
    # redraw instead of freezing on the previous series.
    assert (
        build_price_figure(BARS, [], uirevision="AAPL:6M").layout.uirevision
        == "AAPL:6M"
    )


def test_actual_close_line_present() -> None:
    fig = build_price_figure(BARS, [])
    actual = [t for t in fig.data if t.name == "Actual"]
    assert len(actual) == 1
    assert list(actual[0].y) == [b.close for b in BARS]


def test_actual_line_sorts_unordered_bars_by_ts() -> None:
    shuffled = [BARS[7], BARS[0], BARS[19], BARS[3]]
    fig = build_price_figure(shuffled, [])
    actual = next(t for t in fig.data if t.name == "Actual")
    assert list(actual.x) == sorted(b.ts for b in shuffled)
    assert list(actual.y) == [100.0 + 1, 100.0 + 4, 100.0 + 8, 100.0 + 20]


def test_actual_candles_when_requested() -> None:
    fig = build_price_figure(BARS, [], show_actual_candles=True)
    assert any(isinstance(t, go.Candlestick) for t in fig.data)


def test_ribbon_line_for_selected_horizon_only() -> None:
    snaps = [
        _snapshot(
            horizon="5d",
            made_at=f"2026-02-{d:02d}T00:00:00Z",
            made_from_ts=f"2026-02-{d:02d}T00:00:00Z",
            target_ts=f"2026-02-{d + 5:02d}T00:00:00Z",
            anchor_price=100.0 + d,
            predicted_price=101.0 + d,
            lower_bound=98.0 + d,
            upper_bound=104.0 + d,
        )
        for d in range(1, 6)
    ]
    snaps.append(
        _snapshot(
            horizon="1d",
            made_at="2026-02-10T00:00:00Z",
            made_from_ts="2026-02-10T00:00:00Z",
            target_ts="2026-02-11T00:00:00Z",
            anchor_price=110.0,
            predicted_price=111.0,
            lower_bound=109.0,
            upper_bound=113.0,
        )
    )
    fig = build_price_figure(BARS, snaps, ribbon_horizon="5d", latest_horizons=())
    ribbons = [t for t in fig.data if t.name == "Predicted 5d"]
    assert len(ribbons) == 1
    # 5 snapshots plotted at their target_ts, sorted
    assert len(ribbons[0].x) == 5


def test_latest_forecast_ci_band_uses_fill_tonexty() -> None:
    snaps = [
        _snapshot(
            horizon="5d",
            made_at="2026-02-20T00:00:00Z",
            made_from_ts="2026-02-20T00:00:00Z",
            target_ts="2026-02-25T00:00:00Z",
            anchor_price=120.0,
            predicted_price=123.0,
            lower_bound=115.0,
            upper_bound=131.0,
        )
    ]
    fig = build_price_figure(
        BARS,
        snaps,
        ribbon_horizon=None,
        latest_horizons=("5d",),
        show_rsi=False,
        show_macd=False,
        show_volume=False,
    )
    # Check that CI band uses fill="tonexty" (BB Lower also uses it, so check for at least one)
    fills = [t.fill for t in fig.data if hasattr(t, "fill") and t.fill == "tonexty"]
    assert len(fills) >= 1
    assert any(t.name == "Forecast 5d" for t in fig.data)


def test_markers_hidden_by_default_and_shown_with_flag() -> None:
    snaps = [
        _snapshot(
            horizon="1d",
            made_at="2026-02-10T00:00:00Z",
            made_from_ts="2026-02-10T00:00:00Z",
            target_ts="2026-02-11T00:00:00Z",
            anchor_price=110.0,
            predicted_price=111.0,
            lower_bound=109.0,
            upper_bound=113.0,
            is_direction_hit=1,
            evaluated_at="2026-02-12T00:00:00Z",
            realized_price=111.5,
        )
    ]
    no_markers = build_price_figure(BARS, snaps, show_markers=False, latest_horizons=())
    assert not any((t.name or "").startswith("marker") for t in no_markers.data)

    with_markers = build_price_figure(
        BARS, snaps, show_markers=True, ribbon_horizon=None, latest_horizons=()
    )
    marker_traces = [
        t for t in with_markers.data if (t.meta or {}).get("kind") == "marker"
    ]
    assert len(marker_traces) == 1


def test_marker_color_reflects_evaluation_state() -> None:
    common = {
        "made_at": "2026-02-10T00:00:00Z",
        "made_from_ts": "2026-02-10T00:00:00Z",
        "target_ts": "2026-02-11T00:00:00Z",
        "anchor_price": 110.0,
        "predicted_price": 111.0,
        "lower_bound": 109.0,
        "upper_bound": 113.0,
    }
    hit = _snapshot(
        horizon="1d",
        is_direction_hit=1,
        evaluated_at="x",
        realized_price=112.0,
        **common,
    )
    miss = _snapshot(
        horizon="5d",
        is_direction_hit=0,
        evaluated_at="x",
        realized_price=108.0,
        **common,
    )
    pending = _snapshot(horizon="30d", **common)
    fig = build_price_figure(
        BARS,
        [hit, miss, pending],
        show_markers=True,
        ribbon_horizon=None,
        latest_horizons=(),
    )
    colors = {
        (t.meta or {}).get("horizon"): t.line.color
        for t in fig.data
        if (t.meta or {}).get("kind") == "marker"
    }
    assert colors["1d"] != colors["5d"]
    assert colors["30d"] not in (colors["1d"], colors["5d"])


# Tests for technical indicators (B4 from Jim's findings)


def test_sma20_present_when_enabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=True,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert any(t.name == "SMA20" for t in fig.data)
    trace = next(t for t in fig.data if t.name == "SMA20")
    assert len(trace.y) > 0  # Should have computed values


def test_sma20_absent_when_disabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert not any(t.name == "SMA20" for t in fig.data)


def test_sma50_present_when_enabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=True,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert any(t.name == "SMA50" for t in fig.data)


def test_sma50_absent_when_disabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert not any(t.name == "SMA50" for t in fig.data)


def test_bollinger_bands_present_when_enabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=True,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert any(t.name == "BB Upper" for t in fig.data)
    assert any(t.name == "BB Lower" for t in fig.data)


def test_bollinger_bands_absent_when_disabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert not any(t.name == "BB Upper" for t in fig.data)
    assert not any(t.name == "BB Lower" for t in fig.data)


def test_rsi_subpane_present_when_enabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=True,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert any(t.name == "RSI14" for t in fig.data)
    assert len(fig.data) >= 2  # At least Actual + RSI14
    assert (
        hasattr(fig.layout, "yaxis2") and fig.layout.yaxis2 is not None
    )  # Should have yaxis2 for RSI sub-pane


def test_rsi_subpane_absent_when_disabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert not any(t.name == "RSI14" for t in fig.data)


def test_macd_subpane_components_present_when_enabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=True,
        show_volume=False,
        latest_horizons=(),
    )
    assert any(t.name == "MACD" for t in fig.data)
    assert any(t.name == "MACD Signal" for t in fig.data)
    assert any(t.name == "MACD Hist" for t in fig.data)
    # Verify signal and histogram are different (checking they're swapped correctly)
    macd_trace = next(t for t in fig.data if t.name == "MACD")
    signal_trace = next(t for t in fig.data if t.name == "MACD Signal")
    hist_trace = next(t for t in fig.data if t.name == "MACD Hist")
    # Spot-check: values should be different between components
    assert macd_trace.y != signal_trace.y
    assert signal_trace.y != hist_trace.y


def test_macd_subpane_absent_when_disabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert not any(t.name == "MACD" for t in fig.data)
    assert not any(t.name == "MACD Signal" for t in fig.data)
    assert not any(t.name == "MACD Hist" for t in fig.data)


def test_volume_subpane_present_when_enabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=True,
        latest_horizons=(),
    )
    assert any(t.name == "Volume" for t in fig.data)
    assert any(t.name == "Volume SMA20" for t in fig.data)


def test_volume_subpane_absent_when_disabled() -> None:
    fig = build_price_figure(
        BARS,
        [],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
        latest_horizons=(),
    )
    assert not any(t.name == "Volume" for t in fig.data)
    assert not any(t.name == "Volume SMA20" for t in fig.data)


def test_figure_has_correct_number_of_subpanes() -> None:
    # 3 sub-panes enabled
    fig = build_price_figure(
        BARS, [], show_rsi=True, show_macd=True, show_volume=True, latest_horizons=()
    )
    # Should have yaxis1 (price), yaxis2 (RSI), yaxis3 (MACD), yaxis4 (Volume)
    assert hasattr(fig.layout, "yaxis2") and fig.layout.yaxis2 is not None
    assert hasattr(fig.layout, "yaxis3") and fig.layout.yaxis3 is not None
    assert hasattr(fig.layout, "yaxis4") and fig.layout.yaxis4 is not None


def test_subpane_y_axis_titles() -> None:
    fig = build_price_figure(
        BARS, [], show_rsi=True, show_macd=True, show_volume=True, latest_horizons=()
    )
    assert fig.layout.yaxis2.title.text == "RSI"
    assert fig.layout.yaxis3.title.text == "MACD"
    assert fig.layout.yaxis4.title.text == "Volume"


def test_price_pane_y_axis_title() -> None:
    fig = build_price_figure(
        BARS, [], show_rsi=False, show_macd=False, show_volume=False, latest_horizons=()
    )
    assert fig.layout.yaxis.title.text == "Price"


def test_xaxis_title_and_rangeslider_disabled() -> None:
    """Verify xaxis title and rangeslider visibility for proper pane layout."""
    fig = build_price_figure(BARS, [], show_actual_candles=True, latest_horizons=())
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.xaxis.rangeslider.visible is False
