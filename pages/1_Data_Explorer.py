import sys
import os

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_artifacts import load_raw_prices, load_engineered_features, list_tickers

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Data Explorer")
st.caption(
    "Raw 20-year daily OHLCV history and the 10 engineered technical-"
    "indicator features that both models were trained on."
)

try:
    tickers = list_tickers()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

ticker = st.selectbox("Ticker", tickers, index=0)

tab1, tab2 = st.tabs(["Price History", "Engineered Features"])

with tab1:
    try:
        raw = load_raw_prices()
        sub = raw[raw["Ticker"] == ticker].sort_values("Date")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=sub["Date"], open=sub["Open"], high=sub["High"],
            low=sub["Low"], close=sub["Close"], name=ticker,
        ))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                           title=f"{ticker} — Daily OHLC")
        st.plotly_chart(fig, width='stretch')

        vol_fig = px.bar(sub, x="Date", y="Volume", title=f"{ticker} — Daily Volume")
        vol_fig.update_layout(height=250)
        st.plotly_chart(vol_fig, width='stretch')
    except FileNotFoundError as e:
        st.info(str(e))

with tab2:
    try:
        feat = load_engineered_features()
        sub = feat[feat["Ticker"] == ticker].sort_values("Date")
        feature_cols = [c for c in feat.columns if c not in ("Date", "Ticker", "Close")]
        chosen = st.multiselect("Indicators to plot", feature_cols,
                                 default=feature_cols[:3])
        if chosen:
            fig = go.Figure()
            for c in chosen:
                fig.add_trace(go.Scatter(x=sub["Date"], y=sub[c], mode="lines", name=c))
            fig.update_layout(height=450, title=f"{ticker} — Engineered Indicators (raw, pre-normalization)")
            st.plotly_chart(fig, width='stretch')
        st.caption(
            "These are the raw (not yet expanding-window normalized) indicator "
            "values. The models actually train on a per-ticker expanding "
            "z-score of each column, computed with a 252-day warm-up window "
            "to avoid look-ahead bias."
        )
        with st.expander("Show underlying data"):
            st.dataframe(sub.tail(200), width='stretch')
    except FileNotFoundError as e:
        st.info(str(e))
