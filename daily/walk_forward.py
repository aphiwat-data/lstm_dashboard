"""
Walk-forward evaluation of the corrected daily model (same scheme as notebook 05: retrain every 7 trading days on the
latest 365 days), but with scale-free features, a log-return target, and scalers re-fit inside every training window.

For each block of RETRAIN_EVERY test days: fit on the TRAIN_LEN rows before the block (labels known at that time),
predict the block, move on. Compared on identical rows: drift (rolling mean return), ridge, HistGB regressor and
classifier, and the LSTM (dual head: return + direction, averaged over seeds). Also writes tomorrow's forecast.

Usage: python walk_forward.py --data data/processed/xauusd_d1_features.parquet --latest data/processed/xauusd_d1_latest_features.parquet --out-dir results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hourly"))
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from baseline import make_models  # noqa: E402
from train_models import build_rnn, price_report, windows  # noqa: E402


def fit_predict(F: np.ndarray, y_ret: np.ndarray, y_dir: np.ndarray, tr: np.ndarray, targets: np.ndarray, args) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Fit every model on rows `tr` (positions) and predict the rows at positions `targets`. Returns name -> (pred_return, p_up)."""
    from tensorflow import keras

    ret_sd = float(y_ret[tr].std())
    out: dict[str, tuple[np.ndarray, np.ndarray | None]] = {"drift": (np.full(len(targets), float(y_ret[tr].mean())), None)}
    Xtr, Xte = F[tr], F[targets]
    for name, m in {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0)),
        "hist_gb_reg": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0),
    }.items():
        out[name] = (m.fit(Xtr, y_ret[tr] / ret_sd).predict(Xte) * ret_sd, None)
    clf = make_models()["hist_gb"].fit(Xtr, y_dir[tr])
    out["hist_gb_clf"] = (np.zeros(len(targets)), clf.predict_proba(Xte)[:, 1])

    mu, sd = np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0)
    sd[sd == 0] = 1.0
    Z = np.clip(np.nan_to_num((F - mu) / sd), -10, 10)
    W = args.window
    lo = tr.min() - W + 1
    hi = max(tr.max(), targets.max()) + 1
    if lo < 0:
        raise ValueError("not enough history before the first training row for the RNN window")
    Zw = windows(Z[lo:hi], W)  # window k ends at row lo + W - 1 + k
    pick = lambda pos: Zw[pos - (lo + W - 1)]
    Xw_tr, Xw_te = pick(tr), pick(targets)
    rets, ups = [], []
    for seed in range(args.seeds):
        keras.backend.clear_session()
        model = build_rnn(F.shape[1], seed, W, "lstm", args.units)
        model.fit(Xw_tr, {"ret": (y_ret[tr] / ret_sd).astype(np.float32), "dir": y_dir[tr].astype(np.float32)}, validation_split=0.15,
                  epochs=args.epochs, batch_size=64, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        o = model.predict(Xw_te, batch_size=512, verbose=0)
        rets.append(o["ret"].ravel() * ret_sd)
        ups.append(o["dir"].ravel())
    out["lstm"] = (np.mean(rets, axis=0), np.mean(ups, axis=0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--latest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--units", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--retrain-every", type=int, default=7)
    ap.add_argument("--train-len", type=int, default=365)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).sort_index()
    feats = [c for c in df.columns if c.startswith("f_")]
    F, y_ret, y_dir = df[feats].to_numpy(float), df["y_ret_next"].to_numpy(float), df["y_dir_next"].to_numpy(float)
    start = int(np.flatnonzero((df["meta_split"] == "test").to_numpy())[0])

    names = ["drift", "ridge", "hist_gb_reg", "hist_gb_clf", "lstm"]
    preds = {n: (np.zeros(len(df) - start), np.full(len(df) - start, np.nan)) for n in names}
    t0, s, block = time.time(), start, 0
    while s < len(df):
        e = min(s + args.retrain_every, len(df))
        tr = np.arange(max(0, s - args.train_len), s)
        res = fit_predict(F, y_ret, y_dir, tr, np.arange(s, e), args)
        for n in names:
            preds[n][0][s - start:e - start] = res[n][0]
            if res[n][1] is not None:
                preds[n][1][s - start:e - start] = res[n][1]
        block += 1
        print(f"block {block}: rows {s}->{e} ({df.index[s].date()}), {time.time() - t0:.0f}s elapsed", flush=True)
        s = e

    ev = df.iloc[start:]
    results = {
        "protocol": {"retrain_every_days": args.retrain_every, "train_len_days": args.train_len, "window": args.window, "lstm_units": args.units,
                     "seeds": args.seeds, "target": "log-return of next-day close", "test_start": str(ev.index[0].date()), "test_end": str(ev.index[-1].date()), "n_test": len(ev)},
        "models": {},
    }
    for n in names:
        pr, pu = preds[n]
        results["models"][n] = price_report(ev, 1, None if n == "hist_gb_clf" else pr, None if n in ("drift", "ridge", "hist_gb_reg") else pu)
    mom = (ev["f_ret_1"] > 0).astype(float).to_numpy() * 0.998 + 0.001
    results["models"]["momentum_rule"] = price_report(ev, 1, None, mom)
    resid = ev["y_ret_next"].to_numpy() - preds["lstm"][0]
    lo90, hi90 = (float(np.percentile(resid, q)) for q in (5, 95))
    results["lstm_residual_90pct_logret"] = [lo90, hi90]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = ev[["meta_close", "meta_next_close", "y_ret_next", "y_dir_next"]].copy()
    for n in names:
        table[f"pred_ret_{n}"] = preds[n][0]
        table[f"p_up_{n}"] = preds[n][1]
    table.index.name = "date"
    table.to_csv(args.out_dir / "wf_predictions.csv")

    latest = pd.read_parquet(args.latest).sort_index()
    L = np.vstack([F, latest[feats].to_numpy(float)[latest.index > df.index[-1]]])
    tr = np.arange(len(F) - args.train_len, len(F))
    target_pos = np.array([len(L) - 1])
    fut = fit_predict(L, np.r_[y_ret, np.zeros(len(L) - len(F))], np.r_[y_dir, np.zeros(len(L) - len(F))], tr, target_pos, args)
    last_close = float(latest["meta_close"].iloc[-1])
    r_hat, p_hat = float(fut["lstm"][0][0]), float(fut["lstm"][1][0])
    results["latest_forecast"] = {
        "as_of_date": str(latest.index[-1].date()), "latest_close": round(last_close, 2),
        "predicted_return": r_hat, "predicted_next_close": round(last_close * float(np.exp(r_hat)), 2), "p_up": p_hat,
        "interval_90_close": [round(last_close * float(np.exp(r_hat + lo90)), 2), round(last_close * float(np.exp(r_hat + hi90)), 2)],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out_dir / "walk_forward_results.json").write_text(json.dumps(results, indent=2, default=float))
    (args.out_dir / "latest_forecast.json").write_text(json.dumps(results["latest_forecast"], indent=2))

    up = results["models"]["drift"]["direction_from_sign"]["always_up_acc"]
    print(f"\n== WALK-FORWARD daily (test {results['protocol']['test_start']} -> {results['protocol']['test_end']}, n={len(ev)}) | always-up {up:.4f} | naive MAE ${results['models']['drift']['naive_mae_usd']:.2f} ==")
    for n, r in results["models"].items():
        hd, sg = r.get("direction_from_head"), r.get("direction_from_sign")
        d = hd or sg
        print(f"{n:14s} MAE ${r.get('mae_usd', float('nan')):7.2f} skill {r.get('mae_skill_vs_naive', float('nan')):+.4f} DMp {r.get('dm_p_vs_naive', float('nan')):.3f} IC {r.get('ic_spearman', float('nan')):+.3f} | "
              f"acc {d['acc']:.4f} CI [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}] p_vs_always_up {d['p_vs_always_up']:.3f} AUC {hd.get('auc', float('nan')) if hd else float('nan'):.3f}")
    print("\nlatest forecast:", json.dumps(results["latest_forecast"]))


if __name__ == "__main__":
    main()
