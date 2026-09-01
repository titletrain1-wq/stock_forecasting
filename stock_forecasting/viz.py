"""Plotly chart assembly for the forecast view.

Pure functions only — no Streamlit, no DB. The Streamlit layer loads bars and
prediction snapshots and hands them here; this module returns a ``go.Figure``.

Spec §9 "The overlay":
  1. Actual close (line, or candles on request).
  2. Ribbon view (default) — a continuous predicted line for ONE horizon, each
     point plotted at ``target_ts`` with y = ``predicted_price``.
  3. Historical forecast markers (toggle) — a faint segment per past prediction
     ``(made_from_ts, anchor_price) -> (target_ts, predicted_price)``, coloured
     green / red / grey by direction hit / miss / not-yet-matured.
  4. Latest forecast — dashed anchor->forecast line + a widening CI band
     (``lower_bound`` / ``upper_bound``) via ``go.Scatter(fill="tonexty")``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import plotly.graph_objects as go

ACTUAL_COLOR = "#2962FF"
RIBBON_COLOR = "#FF6D00"
LIVE_COLOR = "#00C853"
FORMING_COLOR = "rgba(0, 200, 83, 0.9)"
BAND_FILLCOLOR = "rgba(255, 109, 0, 0.13)"
HIT_COLOR = "#26A69A"
MISS_COLOR = "#EF5350"
PENDING_COLOR = "rgba(120, 120, 120, 0.55)"

HORIZON_DASH: dict[str, str] = {"1d": "dot", "5d": "dash", "30d": "longdash"}
DEFAULT_LATEST_HORIZONS: tuple[str, ...] = ("1d", "5d", "30d")


class _BarLike(Protocol):
    ts: str
    open: float
    high: float
    low: float
    close: float


class _QuoteLike(Protocol):
    price: float
    ts: str


class _IntradayBarLike(Protocol):
    ts: str
    open: float
    high: float
    low: float
    close: float
    is_provisional: int


class _SnapshotLike(Protocol):
    horizon: str
    made_at: str
    made_from_ts: str
    target_ts: str
    anchor_price: float
    predicted_price: float
    lower_bound: float
    upper_bound: float
    model_type: str
    model_version: str
    realized_price: float | None
    evaluated_at: str | None
    is_direction_hit: int | None


def _marker_color(snap: _SnapshotLike) -> str:
    """Green = direction hit, red = miss, grey = not yet evaluated."""
    if snap.evaluated_at is None or snap.is_direction_hit is None:
        return PENDING_COLOR
    return HIT_COLOR if snap.is_direction_hit else MISS_COLOR


def _latest_per_horizon(
    snapshots: Sequence[_SnapshotLike], horizons: Sequence[str]
) -> dict[str, _SnapshotLike]:
    latest: dict[str, _SnapshotLike] = {}
    for snap in snapshots:
        if snap.horizon not in horizons:
            continue
        cur = latest.get(snap.horizon)
        if cur is None or str(snap.made_at) > str(cur.made_at):
            latest[snap.horizon] = snap
    return latest


def build_price_figure(
    bars: Sequence[_BarLike],
    snapshots: Sequence[_SnapshotLike],
    *,
    ribbon_horizon: str | None = "5d",
    show_markers: bool = False,
    show_actual_candles: bool = False,
    latest_horizons: Sequence[str] = DEFAULT_LATEST_HORIZONS,
    title: str = "",
) -> go.Figure:
    """Assemble the actual + forecast-overlay Plotly figure.

    Args:
        bars: OHLCV bars (need ``ts`` + OHLC attributes), any order.
        snapshots: prediction snapshots for this ticker, any order.
        ribbon_horizon: which horizon's continuous predicted line to draw, or
            ``None`` to omit the ribbon.
        show_markers: draw the per-prediction historical segments.
        show_actual_candles: draw actual price as candles instead of a line.
        latest_horizons: horizons to draw a latest-forecast dashed line + CI band for.
        title: chart title.

    Returns:
        A ``plotly.graph_objects.Figure``. Empty inputs yield an annotated blank figure.
    """
    fig = go.Figure()
    horizons = tuple(latest_horizons)

    if bars:
        b_sorted = sorted(bars, key=lambda b: str(b.ts))
        x = [b.ts for b in b_sorted]
        if show_actual_candles:
            fig.add_trace(
                go.Candlestick(
                    x=x,
                    open=[b.open for b in b_sorted],
                    high=[b.high for b in b_sorted],
                    low=[b.low for b in b_sorted],
                    close=[b.close for b in b_sorted],
                    name="Actual",
                    increasing_line_color=HIT_COLOR,
                    decreasing_line_color=MISS_COLOR,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[b.close for b in b_sorted],
                    name="Actual",
                    mode="lines",
                    line={"color": ACTUAL_COLOR, "width": 2},
                    hovertemplate="%{x}<br>close %{y:.2f}<extra></extra>",
                )
            )

    if ribbon_horizon:
        ribbon = sorted(
            (s for s in snapshots if s.horizon == ribbon_horizon),
            key=lambda s: str(s.target_ts),
        )
        if ribbon:
            fig.add_trace(
                go.Scatter(
                    x=[s.target_ts for s in ribbon],
                    y=[s.predicted_price for s in ribbon],
                    name=f"Predicted {ribbon_horizon}",
                    mode="lines+markers",
                    line={"color": RIBBON_COLOR, "width": 2},
                    marker={"size": 4},
                    hovertemplate=(
                        "target %{x}<br>predicted %{y:.2f}"
                        f"<br>{ribbon_horizon} model<extra></extra>"
                    ),
                )
            )

    if show_markers:
        for snap in sorted(snapshots, key=lambda s: str(s.made_at)):
            color = _marker_color(snap)
            realized = (
                f"{snap.realized_price:.2f}"
                if snap.realized_price is not None
                else "pending"
            )
            fig.add_trace(
                go.Scatter(
                    x=[snap.made_from_ts, snap.target_ts],
                    y=[snap.anchor_price, snap.predicted_price],
                    mode="lines",
                    line={"color": color, "width": 1},
                    opacity=0.5,
                    showlegend=False,
                    meta={"kind": "marker", "horizon": snap.horizon},
                    hovertemplate=(
                        f"made {snap.made_at}<br>{snap.horizon} {snap.model_type} "
                        f"v{snap.model_version}"
                        f"<br>predicted {snap.predicted_price:.2f} · realized {realized}"
                        "<extra></extra>"
                    ),
                )
            )

    latest = _latest_per_horizon(snapshots, horizons)
    for horizon in horizons:
        snap = latest.get(horizon)
        if snap is None:
            continue
        dash = HORIZON_DASH.get(horizon, "dash")
        seg_x = [snap.made_from_ts, snap.target_ts]
        # lower bound (invisible line) then upper bound with fill down to it
        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=[snap.anchor_price, snap.lower_bound],
                mode="lines",
                line={"width": 0, "color": BAND_FILLCOLOR},
                showlegend=False,
                hoverinfo="skip",
                meta={"kind": "ci_lower", "horizon": horizon},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=[snap.anchor_price, snap.upper_bound],
                mode="lines",
                line={"width": 0, "color": BAND_FILLCOLOR},
                fill="tonexty",
                fillcolor=BAND_FILLCOLOR,
                name=f"CI {horizon}",
                hoverinfo="skip",
                meta={"kind": "ci_upper", "horizon": horizon},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=[snap.anchor_price, snap.predicted_price],
                mode="lines+markers",
                line={"color": RIBBON_COLOR, "width": 2, "dash": dash},
                marker={"size": 6},
                name=f"Forecast {horizon}",
                meta={"kind": "forecast", "horizon": horizon},
                hovertemplate=(
                    f"{horizon} forecast %{{y:.2f}}<br>"
                    f"CI [{snap.lower_bound:.2f}, {snap.upper_bound:.2f}]<extra></extra>"
                ),
            )
        )

    layout: dict[str, Any] = {
        "title": title,
        "xaxis_title": "Date",
        "yaxis_title": "Price",
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
        "xaxis": {"rangeslider": {"visible": False}},
        "height": 480,
        # uirevision keeps zoom / pan / hover state across @st.fragment tick
        # refreshes (M5 streaming chart) — without it every live update snaps
        # the view back to the default range.
        "uirevision": True,
    }
    fig.update_layout(**layout)

    if not fig.data:
        fig.add_annotation(
            text="No data yet — run the worker to ingest bars and forecasts.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )

    return fig


def add_live_price_line(
    fig: go.Figure,
    quotes: Sequence[_QuoteLike],
    intraday: Sequence[_IntradayBarLike],
) -> go.Figure:
    """Overlay the live price line + a forming-candle trace onto ``fig``.

    ``intraday`` is the recent ``intraday_bars`` cache (any order); ``quotes`` is
    the current ``live_quotes`` point(s) — only the last is used. The forecast
    ribbon and CI band are untouched: they stay anchored to ``P_close`` (ML core
    is frozen), and the live price only moves the on-chart line and the forming
    candle. A no-op when there is nothing live to draw.

    Crypto-EOD caveat (GATE 0 condition 3): crypto trades 24/7 with no real
    00:00 UTC close, so the "daily close" is only the provider's cutoff bar. The
    live line -> ribbon-origin handover can therefore show a small step when the
    provider's daily close differs from the last live tick. Equities (a real
    16:00 ET close) do not have this. See docs/KNOWN_LIMITATIONS.md §0.
    """
    bars = sorted(intraday, key=lambda b: str(b.ts))
    quote = quotes[-1] if quotes else None

    x: list[str] = [b.ts for b in bars]
    y: list[float] = [b.close for b in bars]
    if quote is not None:
        x.append(quote.ts)
        y.append(quote.price)
    if not x:
        return fig

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name="live",
            mode="lines",
            line={"color": LIVE_COLOR, "width": 2},
            hovertemplate="%{x}<br>live %{y:.2f}<extra></extra>",
            meta={"kind": "live"},
        )
    )

    # Forming candle: the latest still-provisional bucket (extended by the newest
    # tick), or a flat candle at the live quote when no provisional bucket exists
    # yet. Rendered faded with an is_provisional marker in the hover text.
    forming = next((b for b in reversed(bars) if int(b.is_provisional) == 1), None)
    if forming is not None:
        o, h, low, c = forming.open, forming.high, forming.low, forming.close
        fx = forming.ts
        if quote is not None:
            h, low, c = max(h, quote.price), min(low, quote.price), quote.price
    elif quote is not None:
        o = h = low = c = quote.price
        fx = quote.ts
    else:
        return fig

    fig.add_trace(
        go.Candlestick(
            x=[fx],
            open=[o],
            high=[h],
            low=[low],
            close=[c],
            name="forming",
            showlegend=False,
            opacity=0.5,
            increasing_line_color=FORMING_COLOR,
            decreasing_line_color=FORMING_COLOR,
            line={"width": 1},
            hovertext="forming bucket — provisional (is_provisional=1)",
            meta={"kind": "forming"},
        )
    )
    return fig
