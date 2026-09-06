import sys
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_artifacts import load_test_predictions, list_tickers

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")
st.title("🎯 Predictions Explorer")
st.caption(
    "Actual vs predicted price on the held-out chronological test set "
    "(2020 onward) for both models."
)

try:
    df = load_test_predictions()
    tickers = list_tickers()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

ticker = st.selectbox("Ticker", tickers, index=0)
sub = df[df["ticker"] == ticker].sort_values("date")

tab1, tab2 = st.tabs(["Price Path", "Scatter: Predicted vs Actual"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["actual_price"], mode="lines",
                              name="Actual", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["transformer_pred"], mode="lines",
                              name="Transformer", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["cnn_lstm_pred"], mode="lines",
                              name="CNN-BiLSTM", line=dict(color="#EF553B")))
    fig.update_layout(height=550, title=f"{ticker} — Actual vs Predicted Price (T+5)",
                       xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, width='stretch')

with tab2:
    model_choice = st.radio("Model", ["transformer", "cnn_lstm"], horizontal=True,
                             format_func=lambda x: "Temporal Transformer" if x == "transformer" else "CNN-BiLSTM")
    pred_col = f"{model_choice}_pred"
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sub["actual_price"], y=sub[pred_col], mode="markers",
                               marker=dict(size=5, opacity=0.5), name=ticker))
    lo = min(sub["actual_price"].min(), sub[pred_col].min())
    hi = max(sub["actual_price"].max(), sub[pred_col].max())
    fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                               line=dict(color="gray", dash="dash"), name="Perfect prediction"))
    fig2.update_layout(height=550, title=f"{ticker} — Predicted vs Actual Price",
                        xaxis_title="Actual Price", yaxis_title="Predicted Price")
    st.plotly_chart(fig2, width='stretch')

st.divider()
st.subheader("All-ticker snapshot")
st.dataframe(
    df.groupby("ticker")[["actual_price", "cnn_lstm_pred", "transformer_pred"]]
    .agg(lambda s: round(s.mean(), 2)),
    width='stretch',
)
