import sys
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_artifacts import load_metrics, load_per_ticker_mape

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Comparison")
st.caption(
    "Temporal Transformer vs CNN-BiLSTM vs a random-walk (no-change) "
    "baseline, evaluated on the same held-out chronological test set."
)

try:
    metrics = load_metrics()
    per_ticker = load_per_ticker_mape()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

rows = []
label_map = {"transformer": "Temporal Transformer", "cnn_lstm": "CNN-BiLSTM",
             "naive_baseline": "Random-Walk Baseline"}
for key, label in label_map.items():
    m = metrics[key]
    rows.append({"Model": label, **m})
summary = pd.DataFrame(rows).set_index("Model")

st.subheader("Test-Set Metrics")
st.dataframe(summary.style.format("{:.3f}"), width='stretch')

st.markdown(
    """
- **MAPE / RMSPE** — mean / root-mean-square absolute percentage error (lower is better)
- **Bias** — mean signed percentage error (0 is unbiased; negative = model under-predicts price)
- **Directional Accuracy** — % of test windows where the model correctly predicted the direction of the T+5 move
"""
)

st.divider()
st.subheader("Metric Comparison Chart")
metric_choice = st.selectbox("Metric", ["MAPE", "RMSPE", "DirectionalAccuracy", "Bias", "MAE", "RMSE"])
fig = go.Figure(go.Bar(x=summary.index, y=summary[metric_choice],
                        marker_color=["#636EFA", "#EF553B", "#AAAAAA"]))
fig.update_layout(height=450, title=f"{metric_choice} by Model", yaxis_title=metric_choice)
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Per-Ticker MAPE")
fig2 = px.bar(per_ticker, x="ticker", y="MAPE", color="model", barmode="group",
              color_discrete_map={"transformer": "#636EFA", "cnn_lstm": "#EF553B"})
fig2.update_layout(height=500, xaxis_title="Ticker", yaxis_title="MAPE (%)")
st.plotly_chart(fig2, width='stretch')

pivot = per_ticker.pivot(index="ticker", columns="model", values="MAPE")
if {"transformer", "cnn_lstm"}.issubset(pivot.columns):
    n_transformer_wins = int((pivot["transformer"] < pivot["cnn_lstm"]).sum())
    st.metric("Tickers where Transformer beats CNN-BiLSTM (lower MAPE)",
              f"{n_transformer_wins} / {len(pivot)}")
