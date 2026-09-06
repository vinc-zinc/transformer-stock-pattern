import streamlit as st

st.set_page_config(page_title="Model Architecture", page_icon="🧠", layout="wide")
st.title("🧠 Model Architecture")

col1, col2 = st.columns(2)

with col1:
    st.header("Temporal Transformer")
    st.markdown(
        """
**Input:** 20-day window × 10 features

1. **Linear projection** (`Dense`) — features → `d_model=64`
2. **Sinusoidal positional encoding** — injects order information,
   since self-attention itself is permutation-invariant
3. **2× Transformer encoder blocks**, each:
   - Multi-head self-attention (4 heads, key_dim=16)
   - Residual add + LayerNorm
   - Feed-forward (Dense 256 → Dense 64, ReLU)
   - Residual add + LayerNorm
4. **Global average pooling** over the time axis
5. **Dense(32, ReLU) → Dropout(0.10) → Dense(1, linear)**

**Loss:** Huber (δ=0.03) on the price-ratio target
**Optimizer:** Adam, lr=1e-3
"""
    )
    st.code(
        '''class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ff1 = layers.Dense(ff_dim, activation='relu')
        self.ff2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False, return_attention=False):
        if return_attention:
            attn_out, attn_w = self.attn(
                x, x, return_attention_scores=True, training=training)
        else:
            attn_out, attn_w = self.attn(x, x, training=training), None
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff2(self.ff1(x)))
        return (x, attn_w) if return_attention else x''',
        language="python",
    )

with col2:
    st.header("CNN-BiLSTM")
    st.markdown(
        """
**Input:** 20-day window × 10 features

1. **Residual CNN block 1** — two `Conv1D(128)` + BatchNorm + ReLU, with
   a residual (skip) connection
2. **Channel (squeeze-excite) attention** — learns to re-weight the 128
   feature channels by their global importance
3. **Residual CNN block 2** — two `Conv1D(256)` + BatchNorm + ReLU + residual
4. **Bidirectional LSTM(128)** — reads the 20-day sequence forward and
   backward
5. **Dropout(0.30) → Dense(64, ReLU, L2) → Dropout(0.15) → Dense(1, linear)**

**Loss:** Huber (δ=0.03) on the price-ratio target
**Optimizer:** Adam, lr=5e-4
"""
    )
    st.code(
        '''def residual_cnn_block(x, filters, name_prefix, l2=1e-4):
    shortcut = x
    y = layers.Conv1D(filters, 3, padding='same',
                       kernel_regularizer=keras.regularizers.l2(l2))(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv1D(filters, 3, padding='same',
                       kernel_regularizer=keras.regularizers.l2(l2))(y)
    y = layers.BatchNormalization()(y)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
    y = layers.Add()([shortcut, y])
    return layers.Activation('relu')(y)''',
        language="python",
    )

st.divider()
st.markdown(
    """
Both models share the exact same input windows, target definition, and
training/validation/test split (chronological, not random, to prevent
look-ahead bias): a sliding 20-day window of features predicts the
5-trading-day-forward price **ratio**, which is converted back to an
absolute price for evaluation.
"""
)
