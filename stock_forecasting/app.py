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

from stock_forecasting.config import get_settings
from stock_forecasting.database import create_tables, get_engine, seed_watchlist
from stock_forecasting.health_view import build_health_view
from stock_forecasting.intraday_store import IntradayRepository, LiveQuoteRepository
from stock_forecasting.panels import (
    accuracy_rows,
    build_explain_figure,
    explain_contributions,
    latest_snapshot,
)
from stock_forecasting.schema import (
    AccuracyRecord,
    IntradayAccuracyRecord,
    IntradayBarsHistory,
    IntradayPredictionSnapshot,
    OhlcvBar,
    PredictionSnapshot,
    SystemHeartbeat,
    Ticker,
)
from stock_forecasting.viz import (
    CI_DISCLAIMER,
    DEFAULT_LATEST_HORIZONS,
    add_live_price_line,
    build_intraday_forecast_figure,
    build_price_figure,
)

INTRADAY_CRYPTO = {"BTC-USD", "ETH-USD"}

RANGE_DAYS: dict[str, int] = {"1M": 31, "3M": 93, "6M": 186, "1Y": 372}


def _refresh_for(asset_class: str) -> int:
    """Fragment ``run_every`` cadence (seconds) per asset class. Spec §3.2."""
    s = get_settings()
    if asset_class == "crypto":
        return s.live_fragment_refresh_crypto_sec
    return s.live_fragment_refresh_equity_sec


def _delayed_badge(asset_class: str) -> str:
    """Equity intraday data is ~15 min delayed; crypto is live. Spec §3."""
    return "" if asset_class == "crypto" else "🟡 15-min delayed"


def load_live(
    engine, symbol: str, asset_class: str
) -> tuple[list[object], list[object]]:
    """Return ``([latest live_quote] or [], recent intraday_bars)`` for a symbol.

    Reads only the live display tables — never a provider. Crypto uses 1m
    buckets, equities the configured intraday interval.
    """
    interval = (
        "1m" if asset_class == "crypto" else get_settings().intraday_equity_interval
    )
    with Session(engine) as session:
        quote = LiveQuoteRepository(session).get(symbol)
        bars = IntradayRepository(session).get_recent(symbol, interval, limit=180)
    return ([quote] if quote is not None else [], list(bars))


def load_tickers(engine) -> list[Ticker]:
    """Return all active tickers."""
    with Session(engine) as session:
        return list(session.exec(select(Ticker).where(Ticker.active == 1)).all())


def load_bars(engine, symbol: str, days: int) -> list[OhlcvBar]:
    """Return the selected ticker's daily bars within the last ``days``.

    The cutoff is a date-only prefix (``YYYY-MM-DD``) so the lexicographic
    ``OhlcvBar.ts >= cutoff`` filter is independent of the ISO offset spelling
    (``Z`` vs ``+00:00``) and always includes the whole boundary day.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
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
    """Insert a new active ticker (equity or crypto) if it does not already exist."""
    sym = symbol.strip().upper()
    if not sym:
        return
    with Session(engine) as session:
        existing = session.exec(select(Ticker).where(Ticker.symbol == sym)).first()
        if existing:
            return
        is_crypto = sym.endswith("-USD")
        session.add(
            Ticker(
                symbol=sym,
                asset_class="crypto" if is_crypto else "equity",
                display_name=sym,
                provider="coinbase" if is_crypto else "yfinance",
                provider_symbol=sym,
                price_basis="raw" if is_crypto else "adjusted",
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


def render_price_header(symbol: str, bars: list[OhlcvBar], quotes=None) -> None:
    """Symbol · latest price · % change from previous close.

    ``quotes`` (optional) is the live-quote sequence; when present the price tile
    shows the latest live tick instead of the last daily close.
    """
    if not bars:
        st.subheader(f"{symbol} — no bars ingested yet")
        return
    last = bars[-1]
    prev_close = bars[-2].close if len(bars) > 1 else last.close
    quote = quotes[-1] if quotes else None
    price = quote.price if quote is not None else last.close
    pct = ((price - prev_close) / prev_close * 100.0) if prev_close else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric(symbol, f"${price:,.2f}", f"{pct:+.2f}%")
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

    # Indicator toggles
    i1, i2, i3 = st.columns(3)
    show_sma = i1.checkbox("SMA20/50", value=True, key="show_sma")
    show_bollinger = i1.checkbox("Bollinger Bands", value=True, key="show_bollinger")
    show_rsi = i2.checkbox("RSI14", value=True, key="show_rsi")
    show_macd = i2.checkbox("MACD", value=True, key="show_macd")
    show_volume = i3.checkbox("Volume", value=True, key="show_volume")

    bars = load_bars(engine, symbol, RANGE_DAYS[range_label])
    snapshots = load_snapshots(engine, symbol)
    quotes, intraday = load_live(engine, symbol, ticker.asset_class)

    def _live_header() -> None:
        """Live price header — the ONLY thing that re-runs on the fragment tick.

        The actual+forecast chart is deliberately NOT in here: Streamlit re-mounts
        an ``st.plotly_chart`` on every ``run_every`` tick (no in-place
        ``Plotly.react``), which throws away the viewer's zoom / pan and makes the
        chart impossible to study while it refreshes. The chart is rendered once
        per interaction below; only this lightweight header chases the live quote.
        """
        q, _intraday = load_live(engine, symbol, ticker.asset_class)
        render_price_header(symbol, bars, q)
        badge = _delayed_badge(ticker.asset_class)
        if badge:
            st.caption(badge)

    st.fragment(_live_header, run_every=_refresh_for(ticker.asset_class))()

    fig = build_price_figure(
        bars,
        snapshots,
        ribbon_horizon=None if ribbon_horizon == "(none)" else ribbon_horizon,
        show_markers=show_markers,
        show_actual_candles=use_candles,
        latest_horizons=tuple(latest_horizons),
        title=f"{symbol} · actual vs forecast",
        uirevision=f"{symbol}:{range_label}",
        show_sma=show_sma,
        show_bollinger=show_bollinger,
        show_rsi=show_rsi,
        show_macd=show_macd,
        show_volume=show_volume,
    )
    add_live_price_line(fig, quotes, intraday)
    # Stable key so a control change (range / overlay toggle) redraws in place and
    # keeps the current zoom. The live line here is as-of the last full rerun; it
    # advances when you touch a control or reload, not on a timer — a deliberate
    # trade so the chart stays steady enough to read.
    st.plotly_chart(fig, use_container_width=True, key="live_price_chart")
    # M6: calibration disclaimer — the CI band is anchored to P_close, never
    # the live price. Also on the figure itself and in KNOWN_LIMITATIONS.md.
    st.caption(CI_DISCLAIMER)


def load_intraday_history(engine, symbol: str, hours: int = 36) -> list:
    """Recent closed 5m bars from intraday_bars_history for the intraday panel."""
    cutoff = (
        (datetime.now(UTC) - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    )
    with Session(engine) as session:
        return list(
            session.exec(
                select(IntradayBarsHistory)
                .where(
                    IntradayBarsHistory.ticker == symbol,
                    IntradayBarsHistory.interval == "5m",
                    IntradayBarsHistory.ts >= cutoff,
                )
                .order_by(IntradayBarsHistory.ts.asc())
            ).all()
        )


def load_intraday_forecasts(engine, symbol: str) -> list:
    """Latest intraday forecast per horizon for a ticker."""
    with Session(engine) as session:
        rows = list(
            session.exec(
                select(IntradayPredictionSnapshot)
                .where(IntradayPredictionSnapshot.ticker == symbol)
                .order_by(IntradayPredictionSnapshot.made_at.desc())
            ).all()
        )
    latest: dict[str, object] = {}
    for r in rows:
        latest.setdefault(r.horizon, r)
    return list(latest.values())


def load_intraday_accuracy(engine, symbol: str) -> dict:
    """Per-horizon rollup of graded intraday forecasts: n / dir% / MAE% / CI-cover%."""
    with Session(engine) as session:
        recs = list(
            session.exec(
                select(IntradayAccuracyRecord).where(
                    IntradayAccuracyRecord.ticker == symbol,
                    IntradayAccuracyRecord.realized_return.isnot(None),
                )
            ).all()
        )
    out: dict[str, dict] = {}
    for horizon in ("15m", "1h", "4h"):
        h = [r for r in recs if r.horizon == horizon]
        if not h:
            continue
        n = len(h)
        out[horizon] = {
            "n": n,
            "dir_pct": 100 * sum(r.direction_hit or 0 for r in h) / n,
            "mae_pct": sum(r.abs_error_pct or 0.0 for r in h) / n,
            "ci_pct": 100 * sum(r.ci_cover or 0 for r in h) / n,
        }
    return out


def render_intraday_forecast_panel(engine, symbol: str, ticker: Ticker) -> None:
    """Short-horizon (15m/1h/4h) crypto forecast chart + scorecard. Design §7."""
    if symbol not in INTRADAY_CRYPTO:
        return
    forecasts = load_intraday_forecasts(engine, symbol)
    bars = load_intraday_history(engine, symbol)
    st.subheader("Intraday forecast (crypto)")
    if not forecasts:
        st.caption(
            "No intraday forecast yet — the hourly `job_intraday_forecast` "
            "worker job writes one once models are trained and bars are ingested."
        )
        return
    cols = st.columns(len(forecasts))
    for col, fc in zip(
        cols, sorted(forecasts, key=lambda f: f.target_ts), strict=False
    ):
        move = (fc.predicted_price / fc.anchor_price - 1) * 100
        col.metric(
            f"{fc.horizon} → {fc.target_ts[11:16]} UTC",
            f"{fc.predicted_price:,.0f}",
            f"{move:+.2f}%",
        )
        col.caption(f"CI {fc.ci_lower_price:,.0f} – {fc.ci_upper_price:,.0f}")
    fig = build_intraday_forecast_figure(bars, forecasts, title=f"{symbol} · intraday")
    st.plotly_chart(fig, use_container_width=True, key="intraday_forecast_chart")

    scorecard = load_intraday_accuracy(engine, symbol)
    if scorecard:
        st.markdown("**Intraday accuracy** (graded forecasts)")
        st.table(
            {
                "horizon": list(scorecard),
                "n": [s["n"] for s in scorecard.values()],
                "direction %": [f"{s['dir_pct']:.0f}" for s in scorecard.values()],
                "MAE %": [f"{s['mae_pct']:.2f}" for s in scorecard.values()],
                "CI cover %": [f"{s['ci_pct']:.0f}" for s in scorecard.values()],
            }
        )
    else:
        st.caption("Scorecard warming up — grades appear as forecast targets mature.")
    st.caption(
        "Forecast anchored to the last closed 5-minute bar; graded when the "
        "target bar closes. Separate from the daily model above."
    )


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


def render_health_panel(engine) -> None:
    """Data-feed health strip — system badge, providers, watchdog, pending evals. Spec §6."""
    with Session(engine) as session:
        view = build_health_view(session)

    st.subheader("System health")
    h1, h2, h3 = st.columns(3)
    h1.metric("System", view.badge)
    h2.metric("Worker", view.worker_label)
    h3.metric("Data quality", f"{view.data_quality_pct}%")

    if view.warnings:
        for w in view.warnings:
            st.warning(w)

    if view.providers:
        st.dataframe(
            [
                {
                    "provider": c.provider,
                    "RTT p50": f"{c.rtt_p50_ms:.0f}ms" if c.rtt_p50_ms else "—",
                    "err %": f"{c.error_rate * 100:.0f}%",
                    "quota %": f"{c.quota_pct * 100:.0f}%",
                    "breaker": c.breaker_state,
                    "status": c.badge,
                }
                for c in view.providers
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No provider metrics yet — start the worker.")

    watchdog = " · ".join(f"{r.job_type} {r.age_label}" for r in view.watchdog)
    st.caption(
        f"Watchdog: {watchdog or 'no jobs recorded'}    Pending evals: {view.pending_evals}"
    )


def _bootstrap(engine) -> None:
    """Best-effort schema + watchlist bootstrap.

    In production the worker / backfill job owns the schema; the app only
    reads. A DDL hiccup on the hosted DB (e.g. Turso quirks around
    ``CREATE TABLE``) must not take the whole UI down when the tables already
    exist, so this is non-fatal.
    """
    for step in (create_tables, seed_watchlist):
        try:
            step(engine)
        except Exception as exc:  # noqa: BLE001 - startup must not hard-fail
            st.session_state.setdefault("_bootstrap_warnings", []).append(
                f"{step.__name__}: {exc}"
            )


def main() -> None:
    """Render the single-screen forecast view."""
    st.set_page_config(layout="wide", page_title="Stock Forecasting")
    engine = get_engine()
    _bootstrap(engine)

    try:
        tickers = load_tickers(engine)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Database read failed: {exc}")
        st.stop()
        return
    render_sidebar(engine, tickers)

    st.title("Stock Forecast View")
    for warn in st.session_state.get("_bootstrap_warnings", []):
        st.caption(f"startup warning - {warn}")

    with st.expander("diagnostics", expanded=not tickers):
        s = get_settings()
        turso = bool(s.turso_database_url and s.turso_database_url.strip())
        st.write(f"DB target: **{'Turso (remote)' if turso else 'local sqlite'}**")
        st.write(f"TURSO_DATABASE_URL set: {turso}")
        try:
            with Session(engine) as _s:
                nb = _s.exec(select(OhlcvBar)).all()
                nh = _s.exec(select(SystemHeartbeat)).all()
            st.write(f"ohlcv_bars rows: {len(nb)}  ·  system_heartbeat rows: {len(nh)}")
        except Exception as exc:  # noqa: BLE001
            st.write(f"count query failed: {exc}")
    if not tickers:
        st.info("Add a ticker in the sidebar to get started.")
        return

    symbol = st.selectbox("Ticker", [t.symbol for t in tickers])
    ticker = next(t for t in tickers if t.symbol == symbol)
    render_chart_panel(engine, symbol, ticker)
    render_intraday_forecast_panel(engine, symbol, ticker)
    render_accuracy_panel(engine, symbol)
    render_explain_panel(engine, symbol)
    render_health_panel(engine)


if __name__ == "__main__":
    main()
