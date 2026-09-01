from datetime import UTC, datetime

import pandas as pd
import streamlit as st
from sqlmodel import Session, select
from streamlit_lightweight_charts import renderLightweightCharts

from stock_forecasting.database import get_engine
from stock_forecasting.schema import Ticker


def load_tickers(engine):
    with Session(engine) as session:
        return session.exec(select(Ticker).where(Ticker.active == 1)).all()

def main():
    st.set_page_config(layout="wide", page_title="Stock Forecasting")
    engine = get_engine()

    st.sidebar.title("Watchlist")

    tickers = load_tickers(engine)
    if not tickers:
        st.sidebar.write("No active tickers.")
    else:
        for t in tickers:
            st.sidebar.write(f"- {t.symbol} ({t.display_name})")

    st.sidebar.subheader("Manage Tickers")
    with st.sidebar.form("add_ticker"):
        new_symbol = st.text_input("Symbol")
        submit = st.form_submit_button("Add Ticker")
        if submit and new_symbol:
            with Session(engine) as session:
                existing = session.exec(select(Ticker).where(Ticker.symbol == new_symbol.upper())).first()
                if not existing:
                    t = Ticker(
                        symbol=new_symbol.upper(),
                        asset_class="equity",
                        display_name=new_symbol.upper(),
                        provider="yahoo",
                        provider_symbol=new_symbol.upper(),
                        price_basis="adjusted",
                        added_at=datetime.now(UTC).isoformat(),
                        active=1
                    )
                    session.add(t)
                    session.commit()
                    st.rerun()

    st.title("Stock Forecast View")

    if tickers:
        selected = st.selectbox("Select Ticker", [t.symbol for t in tickers])
        st.subheader(f"Ticker: {selected}")
        st.markdown("**Latest Price Badge: $100.00**")  # Dummy for now

        chartOptions = {
            "layout": {
                "textColor": 'black',
                "background": {
                    "type": 'solid',
                    "color": 'white'
                }
            }
        }
        
        base_date = pd.Timestamp("2023-01-01")
        dates = [base_date + pd.Timedelta(days=i) for i in range(100)]
        
        candle_data = []
        for i, d in enumerate(dates):
            candle_data.append({
                "time": d.strftime("%Y-%m-%d"),
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 102 + i
            })
            
        line_data = []
        for i, d in enumerate(dates):
            line_data.append({
                "time": d.strftime("%Y-%m-%d"),
                "value": 100 + i
            })

        upper_bound = []
        lower_bound = []
        for i, d in enumerate(dates[-20:]):
            upper_bound.append({
                "time": d.strftime("%Y-%m-%d"),
                "value": 105 + i + 10
            })
            lower_bound.append({
                "time": d.strftime("%Y-%m-%d"),
                "value": 95 + i - 10
            })
            
        seriesCandlestickChart = [
            {
                "type": "Candlestick",
                "data": candle_data,
                "options": {
                    "upColor": "#26a69a",
                    "downColor": "#ef5350",
                    "borderVisible": False,
                    "wickUpColor": "#26a69a",
                    "wickDownColor": "#ef5350"
                }
            },
            {
                "type": "Line",
                "data": line_data,
                "options": {
                    "color": "blue",
                    "lineWidth": 2
                }
            },
            {
                "type": "Area",
                "data": upper_bound,
                "options": {
                    "lineColor": "rgba(0,0,0,0)",
                    "topColor": "rgba(21, 146, 230, 0.4)",
                    "bottomColor": "rgba(21, 146, 230, 0.0)" 
                }
            },
            {
                "type": "Line",
                "data": lower_bound,
                "options": {
                    "color": "red",
                    "lineStyle": 2, 
                }
            }
        ]

        st.write("Attempting to render Multi-series overlay + Ribbon...")
        renderLightweightCharts([
            {
                "chart": chartOptions,
                "series": seriesCandlestickChart
            }
        ], 'multipane')

        st.warning("FAIL_GATE: Ribbon overlay not supported in streamlit-lightweight-charts. "
                   "The library wraps TradingView Lightweight Charts, which natively lacks a "
                   "clean 'BandSeries' or 'fill-between' feature for plotting confidence interval ribbons. "
                   "AreaSeries only fills down to the zero line. Overlapping lines do not create a filled band.")

if __name__ == "__main__":
    main()
