"""Tests for the accuracy + explain panel data shaping (stock_forecasting.panels)."""

import json
from types import SimpleNamespace

import plotly.graph_objects as go

from stock_forecasting.panels import (
    accuracy_rows,
    build_explain_figure,
    explain_contributions,
    latest_snapshot,
    verdict_label,
    verdict_sentence,
)


def _acc(horizon, *, n=120, dir_acc=0.58, trustworthy=1, model_type="ridge"):
    return SimpleNamespace(
        horizon=horizon,
        model_type=model_type,
        n=n,
        mae=1.2,
        rmse=2.3,
        dir_acc=dir_acc,
        ci_coverage=0.94,
        mae_price_pct=0.021,
        is_trustworthy=trustworthy,
    )


def test_verdict_label_maps_trust_flag() -> None:
    assert verdict_label(1) == "✅ Trust"
    assert verdict_label(0) == "❌ Don't"
    assert verdict_label(None) == "❌ Don't"


def test_verdict_sentence_variants() -> None:
    trust = verdict_sentence(_acc("1d", n=140, dir_acc=0.58, trustworthy=1), "AAPL")
    assert "AAPL 1d ridge" in trust and "trust." in trust

    coinflip = verdict_sentence(_acc("30d", n=90, dir_acc=0.51, trustworthy=0), "AAPL")
    assert "coin-flip" in coinflip

    thin = verdict_sentence(_acc("5d", n=12, dir_acc=0.7, trustworthy=0), "AAPL")
    assert "not enough" in thin


def test_accuracy_rows_ordered_and_verdict_present() -> None:
    recs = [
        _acc("30d", trustworthy=0),
        _acc("1d", trustworthy=1),
        _acc("5d", trustworthy=1),
    ]
    rows = accuracy_rows(recs, ticker="AAPL")
    assert [r["horizon"] for r in rows] == ["1d", "5d", "30d"]
    assert rows[0]["verdict"] == "✅ Trust"
    assert rows[2]["verdict"] == "❌ Don't"
    assert rows[0]["MAE %"] == 2.1
    assert rows[0]["dir %"] == 58.0


def test_latest_snapshot_picks_most_recent_and_filters_model() -> None:
    snaps = [
        SimpleNamespace(
            horizon="1d",
            made_at="2026-02-01T00:00:00Z",
            model_type="ridge",
            explain_json="{}",
        ),
        SimpleNamespace(
            horizon="1d",
            made_at="2026-02-09T00:00:00Z",
            model_type="ridge",
            explain_json="{}",
        ),
        SimpleNamespace(
            horizon="1d",
            made_at="2026-02-20T00:00:00Z",
            model_type="random_forest",
            explain_json="{}",
        ),
    ]
    assert latest_snapshot(snaps, horizon="1d").made_at == "2026-02-20T00:00:00Z"
    assert (
        latest_snapshot(snaps, horizon="1d", model_type="ridge").made_at
        == "2026-02-09T00:00:00Z"
    )
    assert latest_snapshot(snaps, horizon="5d") is None


def test_explain_contributions_sorted_by_abs_value() -> None:
    j = json.dumps({"rsi_14": 0.02, "macd_hist_norm": -0.11, "sma20_stretch": 0.05})
    pairs = explain_contributions(j)
    assert [p[0] for p in pairs] == ["macd_hist_norm", "sma20_stretch", "rsi_14"]
    assert explain_contributions(j, top_n=2) == pairs[:2]


def test_explain_contributions_handles_bad_input() -> None:
    assert explain_contributions(None) == []
    assert explain_contributions("") == []
    assert explain_contributions("not json") == []
    assert explain_contributions("[1,2,3]") == []


def test_build_explain_figure_bar_orientation_and_colors() -> None:
    fig = build_explain_figure([("macd_hist_norm", -0.11), ("rsi_14", 0.02)])
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    assert bar.orientation == "h"
    # largest |value| ends up last in trace (top of chart)
    assert bar.y[-1] == "macd_hist_norm"


def test_build_explain_figure_empty() -> None:
    fig = build_explain_figure([])
    assert len(fig.data) == 0
    assert any("No explain data" in a.text for a in fig.layout.annotations)
