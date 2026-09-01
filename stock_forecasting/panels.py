"""Pure data shaping for the accuracy + explain panels (no Streamlit, no DB).

The Streamlit layer queries ``accuracy_records`` and ``prediction_snapshots`` and
hands rows here; this module returns plain dicts / lists / a ``go.Figure``.

Spec §9:
  - Accuracy panel: rows = horizons; cols = MAE (price %), RMSE, directional %,
    CI coverage, n; one-line verdict per horizon from ``is_trustworthy``
    (``dir_acc >= 0.55 AND n >= 30``).
  - Explain panel: horizontal bar chart of signed feature contributions for the
    latest forecast (ridge: coef*value; RF: global importances).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

import plotly.graph_objects as go

HORIZON_ORDER: tuple[str, ...] = ("1d", "5d", "30d")
POS_COLOR = "#26A69A"
NEG_COLOR = "#EF5350"


class _AccuracyLike(Protocol):
    horizon: str
    model_type: str
    n: int
    mae: float
    rmse: float
    dir_acc: float
    ci_coverage: float
    mae_price_pct: float
    is_trustworthy: int


class _SnapshotLike(Protocol):
    horizon: str
    made_at: str
    model_type: str
    explain_json: str


def verdict_label(is_trustworthy: int | None) -> str:
    """Short trust badge for an accuracy row."""
    return "✅ Trust" if is_trustworthy else "❌ Don't"


def verdict_sentence(rec: _AccuracyLike, ticker: str | None) -> str:
    """One-line human verdict, e.g. 'AAPL 1d ridge: 58% directional (n=140) — trust.'"""
    who = f"{ticker} " if ticker else "global "
    pct = f"{rec.dir_acc * 100:.0f}% directional (n={rec.n})"
    if rec.is_trustworthy:
        tail = "trust."
    elif rec.n < 30:
        tail = f"only {rec.n} samples — not enough to judge."
    else:
        tail = "coin-flip — don't."
    return f"{who}{rec.horizon} {rec.model_type}: {pct} — {tail}"


def accuracy_rows(
    records: Sequence[_AccuracyLike],
    *,
    ticker: str | None = None,
    horizons: Sequence[str] = HORIZON_ORDER,
) -> list[dict[str, Any]]:
    """One display row per horizon present in ``records`` (ordered 1d/5d/30d first)."""
    by_h = {r.horizon: r for r in records}
    ordered = [h for h in horizons if h in by_h]
    ordered += sorted(h for h in by_h if h not in horizons)
    rows: list[dict[str, Any]] = []
    for h in ordered:
        r = by_h[h]
        rows.append(
            {
                "horizon": h,
                "MAE %": round(r.mae_price_pct * 100, 2),
                "RMSE": round(r.rmse, 4),
                "dir %": round(r.dir_acc * 100, 1),
                "CI cov %": round(r.ci_coverage * 100, 1),
                "n": r.n,
                "verdict": verdict_label(r.is_trustworthy),
                "_sentence": verdict_sentence(r, ticker),
            }
        )
    return rows


def latest_snapshot(
    snapshots: Sequence[_SnapshotLike],
    *,
    horizon: str,
    model_type: str | None = None,
) -> _SnapshotLike | None:
    """Most-recent (by ``made_at``) snapshot for a horizon (and optional model)."""
    best: _SnapshotLike | None = None
    for s in snapshots:
        if s.horizon != horizon:
            continue
        if model_type is not None and s.model_type != model_type:
            continue
        if best is None or str(s.made_at) > str(best.made_at):
            best = s
    return best


def explain_contributions(
    explain_json: str | None, *, top_n: int | None = None
) -> list[tuple[str, float]]:
    """Parse ``explain_json`` into (feature, value) pairs sorted by |value| desc."""
    if not explain_json:
        return []
    try:
        raw = json.loads(explain_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(raw, dict):
        return []
    pairs = [(str(k), float(v)) for k, v in raw.items()]
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return pairs[:top_n] if top_n else pairs


def build_explain_figure(
    contributions: Sequence[tuple[str, float]], *, title: str = "Why this forecast?"
) -> go.Figure:
    """Horizontal bar chart of signed feature contributions (largest |value| on top)."""
    fig = go.Figure()
    if not contributions:
        fig.add_annotation(
            text="No explain data for the latest forecast.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(title=title, height=240)
        return fig

    # plot smallest->largest so the biggest bar sits at the top
    ordered = list(reversed(contributions))
    feats = [k for k, _ in ordered]
    vals = [v for _, v in ordered]
    fig.add_trace(
        go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker_color=[POS_COLOR if v >= 0 else NEG_COLOR for v in vals],
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="signed contribution",
        height=max(240, 28 * len(feats) + 90),
        margin={"l": 140, "r": 20, "t": 50, "b": 40},
    )
    return fig
