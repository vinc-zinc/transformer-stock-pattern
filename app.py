"""
Temporal Transformer vs CNN-BiLSTM — Stock Price Forecasting Dashboard
=======================================================================
Landing page. See pages/ for the rest of the multipage app:
  1_Data_Explorer.py        - raw price history + engineered indicators
  2_Model_Architecture.py   - block diagrams / summaries of both models
  3_Training_Curves.py      - loss/MAE curves from the one-time training run
  4_Predictions.py          - actual vs predicted price, per ticker
  5_Feature_Importance.py   - gradient-based feature/day importance
  6_Attention_Explorer.py   - Transformer self-attention heatmaps
  7_Model_Comparison.py     - CNN-BiLSTM vs Transformer, side by side

This app never imports TensorFlow -- every number it shows was
precomputed once by scripts/train_and_export.py and saved under
data/processed/. That keeps the deployed app lightweight and fast,
with no live inference or GPU/CPU training cost per visitor.
"""
import streamlit as st

from src.load_artifacts import load_metrics

st.set_page_config(
    page_title="Transformer vs CNN-BiLSTM Stock Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Temporal Transformer vs CNN-BiLSTM")
st.subheader("Stock Price Forecasting — Interactive Dashboard")

st.markdown(
    """
This dashboard visualizes a deep-learning coursework project that forecasts
**5-trading-day-ahead stock prices** for 49 large-cap tickers, comparing two
architectures trained on the same 20-year daily OHLCV dataset:

- **Temporal Transformer** — a small self-attention encoder (sinusoidal
  positional encoding + multi-head self-attention blocks) operating on a
  20-day sliding window of 10 technical-indicator features.
- **CNN-BiLSTM** — a residual 1-D convolutional feature extractor with a
  squeeze-and-excite channel-attention block, feeding a bidirectional LSTM.

Both models predict a **price ratio** (`price[t+5] / price[t]`), which is
converted back to an absolute price for evaluation. All numbers shown in
this app were computed once, offline, in `scripts/train_and_export.py` —
this deployed app just reads and visualizes the results, so it stays fast
and has no TensorFlow / GPU dependency.

**Use the sidebar to navigate between pages.**
"""
)

st.divider()

try:
    metrics = load_metrics()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transformer Test MAPE", f"{metrics['transformer']['MAPE']:.2f}%")
    with col2:
        st.metric("CNN-BiLSTM Test MAPE", f"{metrics['cnn_lstm']['MAPE']:.2f}%")
    with col3:
        st.metric("Random-Walk Baseline MAPE", f"{metrics['naive_baseline']['MAPE']:.2f}%")

    st.caption(
        "Lower MAPE is better. The random-walk baseline predicts no price "
        "change at all (ratio = 1.0) — beating it is the real bar for a "
        "forecasting model, not just achieving a 'low-looking' error number."
    )
except FileNotFoundError:
    st.warning(
        "No precomputed artifacts found yet. Run "
        "`python scripts/train_and_export.py` first to generate "
        "`data/processed/*` (see README.md)."
    )

st.divider()
st.caption(
    "Source: original coursework notebooks `Temporal_Transformer_price_v2.ipynb` "
    "and `CNN_LSTM_price_fixed_v2.ipynb`, ported into `src/` for reuse by both "
    "the training script and this app."
)
