"""
Shared data pipeline for the CNN-BiLSTM vs Temporal Transformer stock
regression project.

This module is intentionally standalone (only needs pandas/numpy) so it can
be imported both by the one-off training script (scripts/train_and_export.py)
and by the Streamlit app's Data Explorer page, without pulling in
TensorFlow at app-render time.

Ported from the original coursework notebooks (CNN_LSTM_price_fixed_v2.ipynb
and Temporal_Transformer_price_v2.ipynb), condensed into reusable functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Global hyper-parameters (must match the trained models) ──────────────────
W = 20            # sliding window length (days)
HORIZON = 5       # T+5 forward prediction horizon
MIN_HIST = 252    # minimum bars before expanding-window normalisation kicks in
CLIP_LO, CLIP_HI = 0.60, 1.55   # y_ratio outlier clipping bounds

FEATURE_COLS = [
    "log_return",
    "sma_spread",
    "ema_spread",
    "rsi",
    "bb_pctb",
    "atr_ratio",
    "vol_ratio",
    "obv_momentum",
    "vwap_dev",
    "vp_divergence",
]

TRAIN_END = np.datetime64("2018-01-01")
VAL_END = np.datetime64("2020-01-01")


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load, clean, and forward-fill the raw OHLCV CSV."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].str.strip()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Ticker", "Date"])

    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        df[col] = df[col].where(df[col] > 0, np.nan)

    fill_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df[fill_cols] = df.groupby("Ticker")[fill_cols].transform(lambda x: x.ffill())
    return df


def add_features(group: pd.DataFrame) -> pd.DataFrame:
    """Engineer the 10 technical-indicator features for one ticker."""
    g = group.copy()
    c, h, l = g["Close"], g["High"], g["Low"]
    pc = c.shift(1)

    g["log_return"] = np.log(c / pc.replace(0, np.nan))

    sma10 = c.rolling(10, min_periods=5).mean()
    sma50 = c.rolling(50, min_periods=25).mean()
    g["sma_spread"] = (sma10 - sma50) / sma50.replace(0, np.nan)

    ema10 = c.ewm(span=10, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    g["ema_spread"] = (ema10 - ema50) / ema50.replace(0, np.nan)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / (loss + 1e-9)
    g["rsi"] = 1 - 1 / (1 + rs)

    bb_mid = c.rolling(20, min_periods=10).mean()
    bb_std = c.rolling(20, min_periods=10).std()
    band = (4 * bb_std).replace(0, np.nan)
    g["bb_pctb"] = (c - (bb_mid - 2 * bb_std)) / band

    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=7).mean()
    g["atr_ratio"] = atr14 / c.replace(0, np.nan)

    adj_factor = g["Adj Close"] / c.replace(0, np.nan)
    adj_vol = g["Volume"] / adj_factor.replace(0, np.nan)
    vol_ma20 = adj_vol.rolling(20, min_periods=10).mean()
    g["vol_ratio"] = adj_vol / vol_ma20.replace(0, np.nan)

    direction = np.sign(c.diff()).fillna(0)
    obv = (adj_vol * direction).cumsum()
    obv_roc = (obv - obv.shift(10)) / (obv.shift(10).abs() + 1e-9)
    g["obv_momentum"] = obv_roc.clip(-1, 1)

    rolling_pv = (c * adj_vol).rolling(10, min_periods=5).sum()
    rolling_v = adj_vol.rolling(10, min_periods=5).sum()
    vwap10 = rolling_pv / rolling_v.replace(0, np.nan)
    g["vwap_dev"] = (c - vwap10) / vwap10.replace(0, np.nan)

    price_mom = c.pct_change(5)
    vol_mom = adj_vol.pct_change(5)
    g["vp_divergence"] = np.tanh(price_mom * 10) * np.tanh(vol_mom * 10)
    return g


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the T+5 price-ratio target and clip outliers."""
    df = df.copy()
    df["y_price"] = df.groupby("Ticker")["Close"].transform(lambda x: x.shift(-HORIZON))
    df["y_ratio"] = df["y_price"] / df["Close"].replace(0, np.nan)
    df_labeled = df.dropna(subset=["y_ratio", "y_price"] + FEATURE_COLS).copy()
    df_labeled["y_ratio"] = df_labeled["y_ratio"].clip(CLIP_LO, CLIP_HI)
    return df_labeled


def expanding_normalize(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    for col in FEATURE_COLS:
        roll_mean = g[col].expanding(min_periods=MIN_HIST).mean()
        roll_std = g[col].expanding(min_periods=MIN_HIST).std()
        g[col] = (g[col] - roll_mean) / (roll_std + 1e-9)
    return g


def build_windows(df_norm: pd.DataFrame):
    """Slide a length-W window per ticker; returns arrays + a per-sample
    DataFrame of metadata (ticker, date) aligned with the arrays."""
    windows_list, y_ratio_list, y_price_list = [], [], []
    last_close_list, dates_list, tickers_list = [], [], []

    for ticker, group in df_norm.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        feat = group[FEATURE_COLS].values.astype(np.float32)
        ratio = group["y_ratio"].values.astype(np.float32)
        price = group["y_price"].values.astype(np.float32)
        close_raw = group["Close"].values.astype(np.float32)
        dts = group["Date"].values

        for i in range(W, len(group)):
            win = feat[i - W:i]
            if not np.any(np.isnan(win)):
                windows_list.append(win)
                y_ratio_list.append(ratio[i])
                y_price_list.append(price[i])
                last_close_list.append(close_raw[i - 1])
                dates_list.append(dts[i])
                tickers_list.append(ticker)

    X = np.array(windows_list, dtype=np.float32)
    y = np.array(y_ratio_list, dtype=np.float32)
    y_price = np.array(y_price_list, dtype=np.float32)
    last_close = np.array(last_close_list, dtype=np.float32)
    dates_arr = np.array(dates_list, dtype="datetime64[ns]")
    tickers_arr = np.array(tickers_list)
    return X, y, y_price, last_close, dates_arr, tickers_arr


def chronological_split(dates_arr: np.ndarray):
    train_mask = dates_arr < TRAIN_END
    val_mask = (dates_arr >= TRAIN_END) & (dates_arr < VAL_END)
    test_mask = dates_arr >= VAL_END
    return train_mask, val_mask, test_mask


def run_pipeline(csv_path: str):
    """End-to-end: raw CSV -> windowed arrays ready for model training/eval.

    Returns a dict with X/y/y_price/last_close/dates/tickers plus the
    boolean train/val/test masks, and the cleaned+featured DataFrame (useful
    for the app's Data Explorer, which wants per-ticker OHLCV + indicators
    without needing to window/split).
    """
    df = load_raw(csv_path)
    # pandas >= 3.0 always drops the grouping column from the sub-frame
    # passed to groupby().apply() (the old include_groups=True escape hatch
    # was removed outright). Both add_features and expanding_normalize need
    # "Ticker" intact (build_windows groups on it below), so select all
    # columns explicitly first -- that sidesteps the auto-exclusion.
    df = df.groupby("Ticker", group_keys=False)[df.columns].apply(add_features)
    df_labeled = add_target(df)
    df_norm = (
        df_labeled.groupby("Ticker", group_keys=False)[df_labeled.columns]
        .apply(expanding_normalize)
        .dropna(subset=FEATURE_COLS)
    )
    X, y, y_price, last_close, dates_arr, tickers_arr = build_windows(df_norm)
    train_mask, val_mask, test_mask = chronological_split(dates_arr)

    return {
        "df_features": df,             # raw + engineered, UNnormalized (for charts)
        "X": X, "y": y, "y_price": y_price, "last_close": last_close,
        "dates": dates_arr, "tickers": tickers_arr,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
    }
