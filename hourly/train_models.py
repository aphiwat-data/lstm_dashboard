"""
Return-target models on the hourly dataset: drift / momentum / ridge / HistGB (regressor + classifier) / GRU with two heads.

Target for horizon h (in bars): y_ret = log(close[t+h] / close[t]); price forecast = close[t] * exp(predicted return).
Predicting returns (stationary) instead of price levels removes the unseen-price-level failure of the daily pipeline.

Protocol (fixed before running):
  * everything is fit on split == "train"; the GRU early-stops on "val"; "test" is scored once at the end
  * scalers use train statistics only; GRU windows are built inside each split (never across split boundaries)
  * every model is scored on identical rows (first WINDOW-1 rows of each split dropped for all models)
  * for h > 1 neighbouring labels overlap, so CIs/p-values use n_eff = n / h and the DM test uses a wider HAC lag
  * bar to beat for direction is the always-up rate of that split/horizon (gold trends up), not 50%

Usage: python train_models.py --data data/processed_gold/xauusd_h1_features.parquet --horizon 4 --seeds 2 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t as t_dist
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import make_models, score, selective

WINDOW = 48
MOMENTUM_FEATURE = {1: "f_ret_1", 4: "f_ret_3", 24: "f_ret_24"}  # closest available lookback to the horizon


def target_cols(h: int) -> tuple[str, str, str]:
    if h == 1:
        return "y_ret_next", "y_dir_next", "meta_next_close"
    return f"y_ret_h{h}", f"y_dir_h{h}", f"meta_close_h{h}"


def dm_pvalue(err_model: np.ndarray, err_ref: np.ndarray, lag: int) -> float:
    d = err_model**2 - err_ref**2
    n, dbar = len(d), d.mean()
    lrv = d.var()
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    return float(2 * (1 - norm.cdf(abs(dbar / np.sqrt(lrv / n)))))


def ic_pvalue(r: float, n: int, h: int) -> float:
    """Two-sided p for a correlation using the effective sample size n / h (labels overlap for h > 1)."""
    n_eff = max(n // h, 3)
    t = r * np.sqrt((n_eff - 2) / max(1 - r**2, 1e-12))
    return float(2 * (1 - t_dist.cdf(abs(t), n_eff - 2)))


def price_report(g: pd.DataFrame, h: int, pred_ret: np.ndarray | None, p_up: np.ndarray | None = None) -> dict:
    ycol, dcol, ccol = target_cols(h)
    y_ret, y_dir = g[ycol].to_numpy(), g[dcol].to_numpy()
    close, target_close = g["meta_close"].to_numpy(), g[ccol].to_numpy()
    mae0, rmse0 = np.mean(np.abs(close - target_close)), np.sqrt(np.mean((close - target_close) ** 2))
    out: dict = {"naive_mae_usd": float(mae0), "naive_rmse_usd": float(rmse0)}
    if pred_ret is not None:
        pred_price = close * np.exp(pred_ret)
        mae, rmse = np.mean(np.abs(pred_price - target_close)), np.sqrt(np.mean((pred_price - target_close) ** 2))
        varying = np.std(pred_ret) > 0
        ic = spearmanr(pred_ret, y_ret) if varying else None
        out.update(
            mae_usd=float(mae), rmse_usd=float(rmse),
            mae_skill_vs_naive=float(1 - mae / mae0), rmse_skill_vs_naive=float(1 - rmse / rmse0),
            dm_p_vs_naive=dm_pvalue(y_ret - pred_ret, y_ret, max(24, 2 * h)) if varying else float("nan"),
            ic_spearman=float(ic.statistic) if ic else float("nan"), ic_p=ic_pvalue(ic.statistic, len(y_ret), h) if ic else float("nan"),
        )
        out["direction_from_sign"] = score(y_dir, (pred_ret > 0).astype(float) * 0.999 + 0.0005, overlap=h)
    if p_up is not None:
        out["direction_from_head"] = score(y_dir, p_up, overlap=h)
        out["selective"] = selective(y_dir, p_up, overlap=h)
    return out


def windows(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, (window, x.shape[1]))[:, 0].astype(np.float32)


def build_gru(n_feat: int, seed: int):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(WINDOW, n_feat))
    h = keras.layers.GRU(64)(inp)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Dense(32, activation="relu")(h)
    outputs = {"ret": keras.layers.Dense(1, name="ret")(h), "dir": keras.layers.Dense(1, activation="sigmoid", name="dir")(h)}
    model = keras.Model(inp, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"ret": keras.losses.Huber(delta=1.0), "dir": "binary_crossentropy"},
        loss_weights={"ret": 1.0, "dir": 1.0},
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 4, 24])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--skip-gru", action="store_true")
    args = ap.parse_args()
    h = args.horizon
    ycol, dcol, _ = target_cols(h)

    df = pd.read_parquet(args.data)
    feats = [c for c in df.columns if c.startswith("f_")]
    split = {s: g for s, g in df.groupby("meta_split")}
    valid = {s: (split[s][[ycol, dcol]].notna().all(axis=1)).to_numpy()[WINDOW - 1:] for s in split}  # rows scored/trained (no NaN label)
    ev = {s: split[s].iloc[WINDOW - 1:][valid[s]] for s in ("val", "test")}
    tr = split["train"].iloc[WINDOW - 1:][valid["train"]]

    ret_sd, drift = float(tr[ycol].std()), float(tr[ycol].mean())
    results: dict = {"horizon_bars": h, "window": WINDOW, "n_features": len(feats), "train_ret_sd": ret_sd, "train_mean_ret": drift, "val": {}, "test": {}}
    results["typical_span_hours"] = float(split["test"][f"meta_span_h{h}"].median()) if h > 1 else 1.0

    def record(name: str, preds: dict[str, tuple[np.ndarray | None, np.ndarray | None]]) -> None:
        for s in ("val", "test"):
            results[s][name] = price_report(ev[s], h, *preds[s])

    record("drift_only", {s: (np.full(len(ev[s]), drift), None) for s in ("val", "test")})
    mom = {s: (ev[s][MOMENTUM_FEATURE[h]] > 0).astype(float).to_numpy() * 0.998 + 0.001 for s in ("val", "test")}
    record("momentum_rule", {s: (None, mom[s]) for s in ("val", "test")})
    record("reversal_rule", {s: (None, 1 - mom[s]) for s in ("val", "test")})

    z_tr = (tr[ycol] / ret_sd).to_numpy()
    tab = {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0)),
        "hist_gb_reg": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0),
    }
    for name, m in tab.items():
        m.fit(tr[feats], z_tr)
        record(name, {s: (m.predict(ev[s][feats]) * ret_sd, None) for s in ("val", "test")})
    clf = make_models()["hist_gb"].fit(tr[feats], tr[dcol])
    record("hist_gb_clf", {s: (None, clf.predict_proba(ev[s][feats])[:, 1]) for s in ("val", "test")})

    if not args.skip_gru:
        mu, sd = split["train"][feats].mean(), split["train"][feats].std().replace(0, 1)
        prep = lambda g: np.clip(((g[feats] - mu) / sd).fillna(0.0).to_numpy(), -10, 10)
        Xw = {s: windows(prep(split[s])) for s in ("train", "val", "test")}
        yr = {s: (split[s][ycol].to_numpy()[WINDOW - 1:] / ret_sd).astype(np.float32) for s in Xw}
        yd = {s: split[s][dcol].to_numpy()[WINDOW - 1:].astype(np.float32) for s in Xw}
        from tensorflow import keras

        seed_preds = {s: [] for s in ("val", "test")}
        for seed in range(args.seeds):
            model = build_gru(len(feats), seed)
            t0 = time.time()
            model.fit(
                Xw["train"][valid["train"]], {"ret": yr["train"][valid["train"]], "dir": yd["train"][valid["train"]]},
                validation_data=(Xw["val"][valid["val"]], {"ret": yr["val"][valid["val"]], "dir": yd["val"][valid["val"]]}),
                epochs=args.epochs, batch_size=256, verbose=2,
                callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            )
            print(f"seed {seed}: trained in {time.time() - t0:.0f}s", flush=True)
            for s in ("val", "test"):
                out = model.predict(Xw[s][valid[s]], batch_size=1024, verbose=0)
                seed_preds[s].append((out["ret"].ravel() * ret_sd, out["dir"].ravel()))
            record(f"gru_seed{seed}", {s: seed_preds[s][-1] for s in ("val", "test")})
        record("gru_ensemble", {s: (np.mean([p[0] for p in seed_preds[s]], axis=0), np.mean([p[1] for p in seed_preds[s]], axis=0)) for s in ("val", "test")})

    nan = float("nan")
    for s in ("val", "test"):
        n = len(ev[s])
        r0 = results[s]["drift_only"]
        print(f"\n== HORIZON {h} bars (~{results['typical_span_hours']:.0f}h) | {s.upper()} (n={n:,}, n_eff={n // h:,}; naive MAE ${r0['naive_mae_usd']:.2f}) ==")
        print(f"{'model':15s} {'MAE$':>7s} {'skill':>7s} {'DM p':>6s} {'IC':>7s} {'IC p':>6s} | {'acc(sign)':>9s} {'acc(head)':>9s} {'always-up':>9s} {'95% CI (n_eff)':>16s} {'p':>6s} {'AUC':>6s}")
        for name, r in results[s].items():
            sg, hd = r.get("direction_from_sign"), r.get("direction_from_head")
            d = hd or sg
            ci = f"[{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]"
            print(f"{name:15s} {r.get('mae_usd', nan):7.2f} {r.get('mae_skill_vs_naive', nan):+7.4f} {r.get('dm_p_vs_naive', nan):6.3f} {r.get('ic_spearman', nan):+7.4f} {r.get('ic_p', nan):6.3f} | "
                  f"{(sg['acc'] if sg else nan):9.4f} {(hd['acc'] if hd else nan):9.4f} {d['always_up_acc']:9.4f} {ci:>16s} {d['p_vs_always_up']:6.3f} {(hd.get('auc', nan) if hd else nan):6.3f}")

    out = args.data.with_name(f"model_results_h{h}.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
