import sys
import os

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_artifacts import load_feature_importance

st.set_page_config(page_title="Feature Importance", page_icon="📊", layout="wide")
st.title("📊 Gradient-Based Feature Importance")
st.caption(
    "Computed via `tf.GradientTape` on 1,000 random test-set windows per "
    "model: the gradient of the model output w.r.t. each input value, "
    "averaged in magnitude across the batch."
)

try:
    fi = load_feature_importance()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

feature_cols = fi["feature_cols"]
model_name = st.radio("Model", ["transformer", "cnn_lstm"], horizontal=True,
                       format_func=lambda x: "Temporal Transformer" if x == "transformer" else "CNN-BiLSTM")

tab1, tab2 = st.tabs(["By Feature", "By Day-in-Window"])

with tab1:
    vals = fi[model_name]["by_feature"]
    fig = go.Figure(go.Bar(x=feature_cols, y=vals, marker_color="#636EFA"))
    fig.update_layout(height=450, title="Mean |Gradient| per Feature",
                       xaxis_title="Feature", yaxis_title="Mean |gradient|")
    st.plotly_chart(fig, width='stretch')
    st.markdown(
        "Larger bars mean the model's prediction is more sensitive to small "
        "changes in that feature, aggregated across all 20 days in the window "
        "and 1,000 sampled test windows."
    )

with tab2:
    vals = fi[model_name]["by_day"]
    days = [f"D{i+1}" for i in range(len(vals))]
    fig = go.Figure(go.Bar(x=days, y=vals, marker_color="#EF553B"))
    fig.update_layout(height=450, title="Mean |Gradient| per Day in the 20-Day Window "
                                        "(D20 = most recent day)",
                       xaxis_title="Day in window", yaxis_title="Mean |gradient|")
    st.plotly_chart(fig, width='stretch')
    st.markdown(
        "Shows whether the model leans more on recent days (right side) or "
        "distributes attention across the whole 20-day history."
    )
