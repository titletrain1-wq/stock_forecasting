"""Streamlit UI for stock_forecasting — watchlist, price + forecast overlay chart.

The chart is built with Plotly (`st.plotly_chart`): `streamlit-lightweight-charts`
has no fill-between / BandSeries API so the CI-band ribbon cannot render there
(ruling carried from god, 2026-09-01). The service layer stays Streamlit-free so a
React+FastAPI path remains open later if ever needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import streamlit as st
from sqlmodel import Session, select

from stock_forecasting.database import get_engine
from stock_forecasting.panels import (
    accuracy_rows,
    build_explain_figure,
    explain_contributions,
    latest_snapshot,
)
from stock_forecasting.schema import (
    AccuracyRecord,
    OhlcvBar,
    PredictionSnapshot,
    Ticker,
)
from stock_forecasting.viz import DEFAULT_LATEST_HORIZONS, build_price_figure

RANGE_DAYS: dict[str, int] = {"1M": 31, "3M": 93, "6M": 186, "1Y": 372}


def load_tickers(engine) -> list[Ticker]:
    """Return all active tickers."""
    with Session(engine) as session:
        return list(session.exec(select(Ticker).where(Ticker.active == 1)).all())


def load_bars(engine, symbol: str, days: int) -> list[OhlcvBar]:
    """Return the selected ticker's daily bars within the last ``days``."""
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with Session(engine) as session:
        return list(
            session.exec(
                select(OhlcvBar)
                .where(
                    OhlcvBar.ticker == symbol,
                    OhlcvBar.interval == "1d",
                    OhlcvBar.ts >= cutoff,
                )
                .order_by(OhlcvBar.ts.asc())
            ).all()
        )


def load_snapshots(engine, symbol: str) -> list[PredictionSnapshot]:
    """Return all prediction snapshots for a ticker (oldest first)."""
    with Session(engine) as session:
        return list(
            session.exec(
                select(PredictionSnapshot)
                .where(PredictionSnapshot.ticker == symbol)
                .order_by(PredictionSnapshot.made_at.asc())
            ).all()
        )


def load_accuracy_records(
    engine, symbol: str, *, scope: str = "ticker"
) -> list[AccuracyRecord]:
    """Return accuracy_records for a ticker (scope='ticker') or all (scope='global')."""
    with Session(engine) as session:
        stmt = select(AccuracyRecord).where(
            AccuracyRecord.scope == scope,
            AccuracyRecord.window == "all",
        )
        if scope == "ticker":
            stmt = stmt.where(AccuracyRecord.ticker == symbol)
        return list(session.exec(stmt).all())


def add_ticker(engine, symbol: str) -> None:
    """Insert a new active equity ticker if it does not already exist."""
    sym = symbol.strip().upper()
    if not sym:
        return
    with Session(engine) as session:
        existing = session.exec(select(Ticker).where(Ticker.symbol == sym)).first()
        if existing:
            return
        session.add(
            Ticker(
                symbol=sym,
                asset_class="equity",
                display_name=sym,
                provider="yfinance",
                provider_symbol=sym,
                price_basis="adjusted",
                added_at=datetime.now(UTC).isoformat(),
                active=1,
            )
        )
        session.commit()


def render_sidebar(engine, tickers: list[Ticker]) -> None:
    """Watchlist list + add-ticker form."""
    st.sidebar.title("Watchlist")
    if not tickers:
        st.sidebar.write("No active tickers.")
    else:
        for t in tickers:
            st.sidebar.write(f"- **{t.symbol}** · {t.display_name} ({t.asset_class})")

    st.sidebar.subheader("Manage")
    with st.sidebar.form("add_ticker"):
        new_symbol = st.text_input("Symbol")
        if st.form_submit_button("Add ticker") and new_symbol:
            add_ticker(engine, new_symbol)
            st.rerun()


def render_price_header(symbol: str, bars: list[OhlcvBar]) -> None:
    """Symbol · latest price · % change from previous close."""
    if not bars:
        st.subheader(f"{symbol} — no bars ingested yet")
        return
    last = bars[-1]
    prev_close = bars[-2].close if len(bars) > 1 else last.close
    pct = ((last.close - prev_close) / prev_close * 100.0) if prev_close else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric(symbol, f"${last.close:,.2f}", f"{pct:+.2f}%")
    col2.metric("Latest bar", last.ts[:10])
    col3.metric("Bars in range", str(len(bars)))


def render_chart_panel(engine, symbol: str, ticker: Ticker) -> None:
    """Range / overlay controls + the Plotly actual+forecast chart."""
    c1, c2, c3 = st.columns([1, 1, 2])
    range_label = c1.radio("Range", list(RANGE_DAYS), index=2, horizontal=True)
    use_candles = c2.checkbox("Candles", value=False)
    show_markers = c2.checkbox("Historical forecasts", value=False)
    ribbon_horizon = c3.selectbox(
        "Ribbon horizon", ["1d", "5d", "30d", "(none)"], index=1
    )
    latest_horizons = c3.multiselect(
        "Latest forecast + CI band",
        list(DEFAULT_LATEST_HORIZONS),
        default=list(DEFAULT_LATEST_HORIZONS),
    )

    bars = load_bars(engine, symbol, RANGE_DAYS[range_label])
    snapshots = load_snapshots(engine, symbol)

    render_price_header(symbol, bars)

    fig = build_price_figure(
        bars,
        snapshots,
        ribbon_horizon=None if ribbon_horizon == "(none)" else ribbon_horizon,
        show_markers=show_markers,
        show_actual_candles=use_candles,
        latest_horizons=tuple(latest_horizons),
        title=f"{symbol} · actual vs forecast",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_accuracy_panel(engine, symbol: str) -> None:
    """Accuracy table (one row per horizon) + one-line trust verdicts. Spec §9."""
    st.subheader("Accuracy")
    a1, a2 = st.columns(2)
    scope = a1.radio(
        "Scope", ["this ticker", "global"], horizontal=True, key="acc_scope"
    )
    scope_key = "ticker" if scope == "this ticker" else "global"
    records = load_accuracy_records(engine, symbol, scope=scope_key)

    models = sorted({r.model_type for r in records})
    model = a2.selectbox("Model", models or ["ridge"], key="acc_model")
    records = [r for r in records if r.model_type == model]

    if not records:
        st.caption(
            "No graded predictions yet — the evaluator populates this once forecasts mature."
        )
        return

    rows = accuracy_rows(records, ticker=symbol if scope_key == "ticker" else None)
    st.dataframe(
        [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows],
        hide_index=True,
        use_container_width=True,
    )
    for row in rows:
        st.caption(row["_sentence"])


def render_explain_panel(engine, symbol: str) -> None:
    """Collapsible 'Why this forecast?' — signed feature contributions. Spec §9."""
    with st.expander("Why this forecast?", expanded=False):
        snapshots = load_snapshots(engine, symbol)
        if not snapshots:
            st.caption("No forecast yet for this ticker.")
            return
        horizons = sorted({s.horizon for s in snapshots}, key=lambda h: (len(h), h))
        models = sorted({s.model_type for s in snapshots})
        e1, e2 = st.columns(2)
        horizon = e1.selectbox("Horizon", horizons, key="explain_horizon")
        model = e2.selectbox("Model", models, key="explain_model")

        snap = latest_snapshot(snapshots, horizon=horizon, model_type=model)
        if snap is None:
            st.caption("No forecast for that horizon/model combination.")
            return
        st.caption(
            f"Latest {horizon} {model} forecast made {snap.made_at[:16]} — "
            f"predicted ${snap.predicted_price:,.2f}"
        )
        contribs = explain_contributions(snap.explain_json)
        st.plotly_chart(
            build_explain_figure(contribs, title=f"{symbol} {horizon} {model}"),
            use_container_width=True,
        )


def main() -> None:
    """Render the single-screen forecast view."""
    st.set_page_config(layout="wide", page_title="Stock Forecasting")
    engine = get_engine()

    tickers = load_tickers(engine)
    render_sidebar(engine, tickers)

    st.title("Stock Forecast View")
    if not tickers:
        st.info("Add a ticker in the sidebar to get started.")
        return

    symbol = st.selectbox("Ticker", [t.symbol for t in tickers])
    ticker = next(t for t in tickers if t.symbol == symbol)
    render_chart_panel(engine, symbol, ticker)
    render_accuracy_panel(engine, symbol)
    render_explain_panel(engine, symbol)


if __name__ == "__main__":
    main()
