"""
Model architectures for the CNN-BiLSTM and Temporal Transformer regressors.

Ported verbatim (defaults preserved) from the original coursework notebooks:
CNN_LSTM_price_fixed_v2.ipynb and Temporal_Transformer_price_v2.ipynb.

Only imported by scripts/train_and_export.py -- the deployed Streamlit app
never imports TensorFlow, it only reads the exported artifacts in
data/processed/.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

W = 20
NUM_FEATURES = 10
INPUT_SHAPE = (W, NUM_FEATURES)


def huber(y_true, y_pred, delta=0.03):
    return keras.losses.Huber(delta=delta)(y_true, y_pred)


# ── CNN-BiLSTM ────────────────────────────────────────────────────────────
def residual_cnn_block(x, filters, name_prefix, l2=1e-4, kernel_size=3):
    shortcut = x
    y = layers.Conv1D(filters, kernel_size, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2),
                       name=f"{name_prefix}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name_prefix}_bn1")(y)
    y = layers.Activation("relu", name=f"{name_prefix}_relu1")(y)
    y = layers.Conv1D(filters, kernel_size, padding="same",
                       kernel_regularizer=keras.regularizers.l2(l2),
                       name=f"{name_prefix}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name_prefix}_bn2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same",
                                  name=f"{name_prefix}_proj")(shortcut)
    y = layers.Add(name=f"{name_prefix}_add")([shortcut, y])
    y = layers.Activation("relu", name=f"{name_prefix}_relu2")(y)
    return y


def channel_attention(x, ratio=4, name_prefix="se"):
    channels = x.shape[-1]
    squeeze = layers.GlobalAveragePooling1D(name=f"{name_prefix}_squeeze")(x)
    excite = layers.Dense(max(channels // ratio, 1), activation="relu",
                           name=f"{name_prefix}_excite1")(squeeze)
    excite = layers.Dense(channels, activation="sigmoid",
                           name=f"{name_prefix}_excite2")(excite)
    excite = layers.Reshape((1, channels), name=f"{name_prefix}_reshape")(excite)
    return layers.Multiply(name=f"{name_prefix}_scale")([x, excite])


def build_cnn_lstm(conv1_filters=128, conv2_filters=256, lstm_units=128,
                    dropout_rate=0.30, learning_rate=5e-4, l2_reg=1e-4):
    inp = layers.Input(shape=INPUT_SHAPE, name="features")
    x = residual_cnn_block(inp, conv1_filters, name_prefix="rb1", l2=l2_reg)
    x = channel_attention(x, ratio=4, name_prefix="se1")
    x = residual_cnn_block(x, conv2_filters, name_prefix="rb2", l2=l2_reg)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False), name="bilstm")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)
    x = layers.Dense(64, activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_reg), name="dense1")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop2")(x)
    out = layers.Dense(1, activation="linear", name="output")(x)
    m = models.Model(inp, out, name="CNN_BiLSTM_Regression")
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=huber,
              metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                       keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


# ── Temporal Transformer ─────────────────────────────────────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        pos = np.arange(max_len)[:, None]
        dims = np.arange(d_model)[None, :]
        angles = pos / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(angles[None, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.pe.shape[1], "d_model": self.pe.shape[2]})
        return cfg


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ff_dim, self.dropout_rate = d_model, num_heads, ff_dim, dropout
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ff1 = layers.Dense(ff_dim, activation="relu")
        self.ff2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False, return_attention=False):
        if return_attention:
            attn_out, attn_w = self.attn(x, x, return_attention_scores=True, training=training)
        else:
            attn_out = self.attn(x, x, training=training)
            attn_w = None
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ff_out = self.ff2(self.ff1(x))
        x = self.norm2(x + self.drop2(ff_out, training=training))
        return (x, attn_w) if return_attention else x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim, "dropout": self.dropout_rate})
        return cfg


_encoder_blocks = {}


def build_transformer(d_model=64, num_heads=4, ff_dim=256, num_blocks=2,
                       dropout=0.10, learning_rate=1e-3, model_id="main"):
    inp = layers.Input(shape=INPUT_SHAPE, name="features")
    x = layers.Dense(d_model, name="proj")(inp)
    x = PositionalEncoding(W, d_model, name="pos_enc")(x)
    x = layers.Dropout(dropout)(x)
    enc_blocks = []
    for i in range(num_blocks):
        blk = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout, name=f"enc_{i}")
        enc_blocks.append(blk)
        x = blk(x)
    _encoder_blocks[model_id] = enc_blocks
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="drop_final")(x)
    out = layers.Dense(1, activation="linear", name="output")(x)
    m = models.Model(inp, out, name="Temporal_Transformer_Regression")
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=huber,
              metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                       keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


def get_gradients(m, x_batch):
    x_var = tf.Variable(x_batch.astype(np.float32))
    with tf.GradientTape() as tape:
        scalar_out = tf.reduce_sum(m(x_var, training=False))
    return tape.gradient(scalar_out, x_var).numpy()


def extract_attention(mdl, X_samples, model_id="main"):
    enc0 = _encoder_blocks[model_id][0]
    proj_layer = mdl.get_layer("proj")
    pos_layer = mdl.get_layer("pos_enc")
    sub_inp = layers.Input(shape=INPUT_SHAPE)
    sub_x = pos_layer(proj_layer(sub_inp))
    sub_m = models.Model(sub_inp, sub_x)
    pos_out = sub_m.predict(X_samples, verbose=0)
    _, attn_w = enc0(tf.constant(pos_out, dtype=tf.float32), training=False, return_attention=True)
    return attn_w.numpy().mean(axis=1)   # average heads -> (N, W, W)
