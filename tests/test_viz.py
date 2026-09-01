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


BARS = [_bar(f"2026-02-{d:02d}T00:00:00Z", 100.0 + d) for d in range(1, 21)]


def _trace_names(fig: go.Figure) -> list[str]:
    return [t.name for t in fig.data if t.name]


def test_empty_inputs_return_figure_with_no_data_annotation() -> None:
    fig = build_price_figure([], [])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert any("No data" in a.text for a in fig.layout.annotations)


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
    fig = build_price_figure(BARS, snaps, ribbon_horizon=None, latest_horizons=("5d",))
    fills = [t.fill for t in fig.data if t.fill == "tonexty"]
    assert len(fills) == 1
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
