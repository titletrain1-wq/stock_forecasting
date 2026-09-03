"""M6 — ML overlay integrity fence.

The v2 streaming chart moves a live price line across a *static* forecast band.
These tests guarantee the CI band + ribbon stay anchored to P_close: mutating
``live_quotes.price`` must not move ``lower_bound`` / ``upper_bound`` or any
ribbon point, and the calibration disclaimer must be surfaced.
"""

from pathlib import Path
from types import SimpleNamespace

import plotly.graph_objects as go

from stock_forecasting.viz import CI_DISCLAIMER, add_live_price_line, build_price_figure

_KNOWN_LIMITATIONS = (
    Path(__file__).resolve().parents[1] / "docs" / "KNOWN_LIMITATIONS.md"
)


def _bar(ts: str, close: float) -> SimpleNamespace:
    return SimpleNamespace(
        ts=ts, open=close, high=close + 1, low=close - 1, close=close
    )


def _ibar(ts: str, close: float, provisional: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        ts=ts,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        is_provisional=provisional,
    )


def _quote(price: float, ts: str = "2026-02-20T14:00:00Z") -> SimpleNamespace:
    return SimpleNamespace(price=price, ts=ts, source="coinbase_ws")


def _snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        horizon="5d",
        made_at="2026-02-20T00:00:00Z",
        made_from_ts="2026-02-20T00:00:00Z",
        target_ts="2026-02-25T00:00:00Z",
        anchor_price=120.0,
        predicted_price=123.0,
        lower_bound=115.0,
        upper_bound=131.0,
        model_type="ridge",
        model_version="1.0.0",
        realized_price=None,
        evaluated_at=None,
        is_direction_hit=None,
    )


_BARS = [_bar(f"2026-02-{d:02d}T00:00:00Z", 100.0 + d) for d in range(1, 21)]
_INTRADAY = [_ibar(f"2026-02-20T1{h}:00:00Z", 121.0 + h) for h in range(3)]


def _non_live_signature(fig: go.Figure) -> list[tuple]:
    """Every trace that is NOT the live line / forming candle, as comparable data."""
    out: list[tuple] = []
    for t in fig.data:
        if t.name in ("live", "forming"):
            continue
        out.append((t.name, t.type, tuple(t.x or ()), tuple(t.y or ())))
    return out


def _figure_with_live(quote_price: float) -> go.Figure:
    fig = build_price_figure(
        _BARS,
        [_snapshot()],
        ribbon_horizon="5d",
        latest_horizons=("5d",),
        title="X",
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
    )
    add_live_price_line(fig, [_quote(quote_price)], _INTRADAY)
    return fig


def test_band_invariant_under_live_price() -> None:
    """Mutating the live price by +25% leaves the band + ribbon byte-identical."""
    base = _figure_with_live(121.0)
    bumped = _figure_with_live(121.0 * 1.25)

    assert _non_live_signature(base) == _non_live_signature(bumped)

    # ...and the live trace really did move (guards against a no-op false pass).
    live_base = next(t for t in base.data if t.name == "live")
    live_bumped = next(t for t in bumped.data if t.name == "live")
    assert live_base.y[-1] != live_bumped.y[-1]


def test_ci_band_traces_read_only_snapshot_fields() -> None:
    """The CI band y-values equal the snapshot bounds exactly (P_close-anchored)."""
    snap = _snapshot()
    fig = build_price_figure(
        _BARS,
        [snap],
        ribbon_horizon=None,
        latest_horizons=("5d",),
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
    )
    lower = next(t for t in fig.data if (t.meta or {}).get("kind") == "ci_lower")
    upper = next(t for t in fig.data if (t.meta or {}).get("kind") == "ci_upper")
    assert lower.y[-1] == snap.lower_bound
    assert upper.y[-1] == snap.upper_bound


def test_disclaimer_present_in_figure_and_known_limitations() -> None:
    import unicodedata

    fig = build_price_figure(
        _BARS,
        [_snapshot()],
        show_sma=False,
        show_bollinger=False,
        show_rsi=False,
        show_macd=False,
        show_volume=False,
    )
    assert any(a.text == CI_DISCLAIMER for a in fig.layout.annotations)
    known_limitations_text = _KNOWN_LIMITATIONS.read_text(encoding="utf-8")
    # Normalize both to NFC to handle any cross-platform encoding variants
    disclaimer_nfc = unicodedata.normalize("NFC", CI_DISCLAIMER)
    text_nfc = unicodedata.normalize("NFC", known_limitations_text)
    assert disclaimer_nfc in text_nfc


def test_disclaimer_surfaced_by_app_caption() -> None:
    """app.py renders the verbatim disclaimer as a chart caption."""
    from stock_forecasting import app

    assert app.CI_DISCLAIMER == CI_DISCLAIMER
    assert "st.caption(CI_DISCLAIMER)" in Path(app.__file__).read_text()
