"""
Thin, cached loaders for the artifacts produced by scripts/train_and_export.py.
Every function here is safe to call repeatedly from a Streamlit page --
results are cached in-process via st.cache_data / st.cache_resource.

No TensorFlow import anywhere in this file: the deployed app only ever
reads CSV/JSON/NPZ files.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def _path(name: str) -> str:
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing '{name}' in data/processed/. Run "
            f"'python scripts/train_and_export.py' first."
        )
    return p


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(_path("metrics.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_training_history() -> dict:
    with open(_path("training_history.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> dict:
    with open(_path("feature_importance.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_attention_weights():
    data = np.load(_path("attention_weights.npz"), allow_pickle=True)
    return {
        "attn_matrices": data["attn_matrices"],   # (6, W, W)
        "tickers": data["tickers"],
        "dates": data["dates"],
        "kind": data["kind"],
    }


@st.cache_data(show_spinner=False)
def load_per_ticker_mape() -> pd.DataFrame:
    return pd.read_csv(_path("per_ticker_mape.csv"))


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    df = pd.read_csv(_path("test_predictions.csv"), parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_engineered_features() -> pd.DataFrame:
    df = pd.read_csv(_path("engineered_features.csv.gz"), parse_dates=["Date"],
                      compression="gzip")
    return df


@st.cache_data(show_spinner=False)
def load_raw_prices() -> pd.DataFrame:
    """Lazily load the full raw OHLCV history, used by the Data Explorer
    page for the candlestick / volume chart (separate from the smaller
    engineered-feature sample used for indicator charts)."""
    p = os.path.join(RAW_DIR, "top_companies_20y_daily_combined.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "Missing data/raw/top_companies_20y_daily_combined.csv -- "
            "see README.md for how to obtain the dataset."
        )
    df = pd.read_csv(p, parse_dates=["Date"])
    df["Ticker"] = df["Ticker"].str.strip()
    return df


def list_tickers() -> list:
    df = load_test_predictions()
    return sorted(df["ticker"].unique().tolist())
