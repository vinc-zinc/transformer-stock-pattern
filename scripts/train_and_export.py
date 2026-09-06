#!/usr/bin/env python3
"""
One-off offline script: trains both models once (fixed hyperparameters,
no grid search -- the notebook defaults already represent the tuned
configuration from the original assignment), evaluates them on the
held-out chronological test set, and dumps every number the Streamlit
dashboard needs into data/processed/ as small CSV/JSON/NPZ files.

Run once:
    python scripts/train_and_export.py

The deployed Streamlit app never imports TensorFlow or runs this script --
it only reads the artifacts this produces.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import run_pipeline, FEATURE_COLS, W, HORIZON  # noqa: E402

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "top_companies_20y_daily_combined.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 15
PATIENCE = 5
BATCH = 512
SEED = 42


def naive_baseline_metrics(y_true_ratio, y_true_price, last_close):
    """Random-walk baseline: predict no change (ratio = 1.0)."""
    pred_ratio = np.ones_like(y_true_ratio)
    pred_price = last_close * pred_ratio
    return evaluate_arrays(y_true_ratio, pred_ratio, y_true_price, pred_price)


def evaluate_arrays(y_ratio, pred_ratio, y_price, pred_price):
    err = pred_price - y_price
    mape = float(np.mean(np.abs(err) / np.abs(y_price)) * 100)
    rmspe = float(np.sqrt(np.mean((err / y_price) ** 2)) * 100)
    bias = float(np.mean(err / y_price) * 100)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    actual_dir = np.sign(y_ratio - 1.0)
    pred_dir = np.sign(pred_ratio - 1.0)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
    return {"MAPE": mape, "RMSPE": rmspe, "Bias": bias, "MAE": mae, "RMSE": rmse,
            "DirectionalAccuracy": dir_acc}


def main():
    import tensorflow as tf
    from tensorflow import keras
    from models import build_transformer, build_cnn_lstm, get_gradients, extract_attention

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    log("Running data pipeline...")
    data = run_pipeline(RAW_CSV)
    X, y, y_price, last_close = data["X"], data["y"], data["y_price"], data["last_close"]
    dates, tickers = data["dates"], data["tickers"]
    tr, va, te = data["train_mask"], data["val_mask"], data["test_mask"]

    log(f"X shape {X.shape} | train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}")

    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    y_price_te, last_close_te = y_price[te], last_close[te]
    dates_te, tickers_te = dates[te], tickers[te]

    results = {}
    predictions = {}
    histories = {}
    feature_importance = {}
    attention_data = None

    model_specs = [
        ("cnn_lstm", build_cnn_lstm, {}),
        ("transformer", build_transformer, {"model_id": "final"}),
    ]

    for name, builder, kwargs in model_specs:
        log(f"=== Training {name} ===")
        model = builder(**kwargs)
        cb = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                           restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                               patience=2, min_lr=1e-6),
        ]
        t0 = time.time()
        hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                          batch_size=BATCH, epochs=EPOCHS, callbacks=cb, verbose=2)
        log(f"{name} trained in {time.time()-t0:.1f}s, "
            f"{len(hist.history['loss'])} epochs run")

        histories[name] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

        pred_ratio = model.predict(Xte, batch_size=1024, verbose=0).flatten()
        pred_price = last_close_te * pred_ratio
        results[name] = evaluate_arrays(yte, pred_ratio, y_price_te, pred_price)
        log(f"{name} test metrics: {results[name]}")

        predictions[name] = {
            "pred_ratio": pred_ratio, "pred_price": pred_price,
        }

        log(f"Computing gradient-based feature importance for {name}...")
        n_samp = min(1000, len(Xte))
        idx = np.random.choice(len(Xte), n_samp, replace=False)
        grads = get_gradients(model, Xte[idx])
        feature_importance[name] = {
            "by_feature": np.abs(grads).mean(axis=(0, 1)).tolist(),
            "by_day": np.abs(grads).mean(axis=(0, 2)).tolist(),
        }

        if name == "transformer":
            log("Extracting attention weights (transformer, encoder block 0)...")
            price_chg_abs = np.abs(y_price_te - last_close_te)
            large_idx = np.argsort(price_chg_abs)[-3:]
            small_idx = np.argsort(price_chg_abs)[:3]
            sample_idx = np.concatenate([large_idx, small_idx])
            attn_mats = extract_attention(model, Xte[sample_idx], model_id="final")
            attention_data = {
                "attn_matrices": attn_mats,          # (6, W, W)
                "sample_idx": sample_idx,
                "tickers": tickers_te[sample_idx],
                "dates": dates_te[sample_idx],
                "kind": (["large_move"] * 3) + (["small_move"] * 3),
            }

        model.save(os.path.join(OUT_DIR, f"{name}.keras"))
        log(f"Saved {name}.keras")

    log("Computing random-walk baseline metrics...")
    results["naive_baseline"] = naive_baseline_metrics(yte, y_price_te, last_close_te)

    # ── Per-ticker MAPE (for the comparison bar charts) ──────────────────
    per_ticker_rows = []
    for name in ["cnn_lstm", "transformer"]:
        pr = predictions[name]["pred_price"]
        df_t = pd.DataFrame({"ticker": tickers_te, "y_price": y_price_te, "pred_price": pr})
        grp = df_t.groupby("ticker").apply(
            lambda g: float(np.mean(np.abs(g["pred_price"] - g["y_price"]) / np.abs(g["y_price"])) * 100),
            include_groups=False,
        )
        for tkr, mape in grp.items():
            per_ticker_rows.append({"model": name, "ticker": tkr, "MAPE": mape})
    pd.DataFrame(per_ticker_rows).to_csv(os.path.join(OUT_DIR, "per_ticker_mape.csv"), index=False)

    # ── Test-set predictions table (for the Predictions Explorer page) ──
    pred_rows = {
        "date": dates_te, "ticker": tickers_te,
        "last_close": last_close_te, "actual_price": y_price_te,
        "cnn_lstm_pred": predictions["cnn_lstm"]["pred_price"],
        "transformer_pred": predictions["transformer"]["pred_price"],
    }
    pd.DataFrame(pred_rows).to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # ── Scalars / small JSON-able results ────────────────────────────────
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
        json.dump(histories, f, indent=2)
    with open(os.path.join(OUT_DIR, "feature_importance.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "window": W, "horizon": HORIZON,
                    **feature_importance}, f, indent=2)

    # ── Attention weights (NPZ, since it's a small array not scalars) ───
    if attention_data is not None:
        np.savez(
            os.path.join(OUT_DIR, "attention_weights.npz"),
            attn_matrices=attention_data["attn_matrices"],
            tickers=attention_data["tickers"],
            dates=attention_data["dates"].astype(str),
            kind=np.array(attention_data["kind"]),
        )

    # ── Small sample of the featured/normalized data, for the Data
    #    Explorer page charts (full 230k-row raw history is in data/raw/,
    #    this is the lighter engineered version already computed above) ──
    feat_sample = data["df_features"][["Date", "Ticker", "Close"] + FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        feat_sample[c] = feat_sample[c].astype(np.float32).round(6)
    feat_sample.to_csv(os.path.join(OUT_DIR, "engineered_features.csv.gz"),
                        index=False, compression="gzip")

    log("All artifacts written to " + OUT_DIR)
    log("Done.")


if __name__ == "__main__":
    main()
