import sys
import os

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_artifacts import load_attention_weights

st.set_page_config(page_title="Attention Explorer", page_icon="🔦", layout="wide")
st.title("🔦 Transformer Self-Attention Explorer")
st.caption(
    "Encoder block 0 self-attention weights (averaged across heads), for "
    "6 sampled test windows: the 3 largest and 3 smallest actual T+5 price "
    "moves, so you can compare how the model attends within a 20-day window "
    "for a volatile move vs a near-flat one."
)

try:
    data = load_attention_weights()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

labels = [
    f"{i+1}. {t} — {d[:10]} ({k.replace('_', ' ')})"
    for i, (t, d, k) in enumerate(zip(data["tickers"], data["dates"], data["kind"]))
]
idx = st.selectbox("Sample window", list(range(len(labels))), format_func=lambda i: labels[i])

mat = data["attn_matrices"][idx]
days = [f"D{i+1}" for i in range(mat.shape[0])]

fig = go.Figure(go.Heatmap(z=mat, x=days, y=days, colorscale="Viridis"))
fig.update_layout(
    height=600,
    title=f"Self-Attention Heatmap — {labels[idx]}",
    xaxis_title="Key day (attended to)",
    yaxis_title="Query day (attending from)",
)
st.plotly_chart(fig, width='stretch')

st.markdown(
    """
Each cell `(row, col)` shows how much day **row** attends to day **col**
when building its contextual representation. Brighter cells indicate a
stronger attention weight. Rows sum to 1 (softmax over keys).
"""
)
