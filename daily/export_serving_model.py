"""Train the final LSTM exactly as walk_forward.py's closing forecast does (latest 365 labelled rows, window 20, LSTM(32), seeds averaged) and export it
for the numpy serving core: serving_model.npz (weights, scaler, past forecast errors) and serving_reference.json (Keras outputs for parity tests).

Usage: python export_serving_model.py --data data/processed/xauusd_d1_features.parquet --errors results/wf_predictions.csv --out-dir serving_artifacts
Run weekly (the model is retrained on a trailing window) and upload serving_model.npz to s3://<bucket>/model/serving/serving_model.npz."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hourly"))
from train_models import build_rnn, windows  # noqa: E402

from serving import predict_windows, standardise  # noqa: E402


def train_export(df: pd.DataFrame, errors_csv: Path, out_dir: Path, window: int = 20, units: int = 32, seeds: int = 3, epochs: int = 30, train_len: int = 365) -> Path:
    """df: labelled feature dataset from build_daily_dataset.build (any end date). Returns the path of serving_model.npz."""
    a = argparse.Namespace(errors=errors_csv, out_dir=out_dir, window=window, units=units, seeds=seeds, epochs=epochs, train_len=train_len)
    from tensorflow import keras

    df = df.sort_index()
    feats = [c for c in df.columns if c.startswith("f_")]
    F, y_ret, y_dir = df[feats].to_numpy(float), df["y_ret_next"].to_numpy(float), df["y_dir_next"].to_numpy(float)
    tr = np.arange(len(F) - a.train_len, len(F))
    ret_sd = float(y_ret[tr].std())
    mu, sd = np.nanmean(F[tr], axis=0), np.nanstd(F[tr], axis=0)
    sd[sd == 0] = 1.0
    meta = {"feats": np.array(feats), "window": a.window, "mu": mu, "sd": sd, "ret_sd": ret_sd, "n_seeds": a.seeds}
    Z = standardise(meta, F)
    lo = tr.min() - a.window + 1
    Zw = windows(Z[lo:], a.window)
    pick = lambda pos: Zw[pos - (lo + a.window - 1)]
    out: dict = dict(meta)
    for seed in range(a.seeds):
        keras.backend.clear_session()
        m = build_rnn(F.shape[1], seed, a.window, "lstm", a.units)
        m.fit(pick(tr), {"ret": (y_ret[tr] / ret_sd).astype(np.float32), "dir": y_dir[tr].astype(np.float32)}, validation_split=0.15, epochs=a.epochs,
              batch_size=64, verbose=0, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        lstm = next(l for l in m.layers if isinstance(l, keras.layers.LSTM))
        dense = [l for l in m.layers if isinstance(l, keras.layers.Dense)]
        d1 = next(l for l in dense if l.name not in ("ret", "dir"))
        k, r, b = lstm.get_weights()
        out.update({f"s{seed}_lstm_k": k, f"s{seed}_lstm_r": r, f"s{seed}_lstm_b": b})
        out.update({f"s{seed}_d1_w": d1.get_weights()[0], f"s{seed}_d1_b": d1.get_weights()[1]})
        for name in ("ret", "dir"):
            w, bb = m.get_layer(name).get_weights()
            out.update({f"s{seed}_{name}_w": w, f"s{seed}_{name}_b": bb})
        if seed == 0:  # Keras outputs of the first seed on the last 8 windows: the parity test compares the numpy forward pass with them
            ref_pos = np.arange(len(F) - 8, len(F))
            keras_out = m.predict(pick(ref_pos), verbose=0)
            ref0 = {"windows": pick(ref_pos).tolist(), "ret": (keras_out["ret"].ravel() * ret_sd).tolist(), "dir": keras_out["dir"].ravel().tolist(),
                    "rows": [str(df.index[i].date()) for i in ref_pos]}
    errs = pd.read_csv(a.errors)
    out["errors"] = (errs["y_ret_next"] - errs["pred_ret_lstm"]).to_numpy(float)
    out["trained_through"] = str(df.index[-1].date())
    out["model_version"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(a.out_dir / "serving_model.npz", **out)
    model = dict(np.load(a.out_dir / "serving_model.npz"))
    r_np, p_np = predict_windows({**model, "n_seeds": 1}, np.array(ref0["windows"]))  # seed 0 only, comparable with ref0
    ref0.update({"numpy_ret": r_np.tolist(), "numpy_dir": p_np.tolist(), "max_abs_diff_ret": float(np.max(np.abs(r_np - np.array(ref0["ret"])))),
                 "max_abs_diff_dir": float(np.max(np.abs(p_np - np.array(ref0["dir"]))))})
    (a.out_dir / "serving_reference.json").write_text(json.dumps(ref0, indent=1))
    print("trained through", out["trained_through"], "| parity max |diff| ret", ref0["max_abs_diff_ret"], "dir", ref0["max_abs_diff_dir"],
          "| size KB", round((a.out_dir / "serving_model.npz").stat().st_size / 1024))
    return a.out_dir / "serving_model.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--errors", type=Path, default=Path("results/wf_predictions.csv"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--units", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-len", type=int, default=365)
    a = ap.parse_args()
    train_export(pd.read_parquet(a.data), a.errors, a.out_dir, a.window, a.units, a.seeds, a.epochs, a.train_len)


if __name__ == "__main__":
    main()
