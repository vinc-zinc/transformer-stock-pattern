import sys
import os

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_artifacts import load_training_history

st.set_page_config(page_title="Training Curves", page_icon="📉", layout="wide")
st.title("📉 Training Curves")
st.caption(
    "Loss and MAE per epoch during the single fixed-hyperparameter training "
    "run for each model (early stopping on validation loss, patience=5)."
)

try:
    hist = load_training_history()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

metric = st.radio("Metric", ["loss", "mae", "rmse"], horizontal=True)

fig = go.Figure()
colors = {"cnn_lstm": "#EF553B", "transformer": "#636EFA"}
for name, h in hist.items():
    label = "CNN-BiLSTM" if name == "cnn_lstm" else "Temporal Transformer"
    epochs = list(range(1, len(h[metric]) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=h[metric], mode="lines+markers",
                              name=f"{label} (train)",
                              line=dict(color=colors.get(name), dash="solid")))
    val_key = f"val_{metric}"
    if val_key in h:
        fig.add_trace(go.Scatter(x=epochs, y=h[val_key], mode="lines+markers",
                                  name=f"{label} (val)",
                                  line=dict(color=colors.get(name), dash="dash")))

fig.update_layout(height=550, xaxis_title="Epoch", yaxis_title=metric.upper(),
                   title=f"Training vs Validation {metric.upper()} per Epoch")
st.plotly_chart(fig, width='stretch')

with st.expander("Raw history JSON"):
    st.json(hist)
