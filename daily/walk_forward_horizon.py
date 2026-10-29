"""Walk-forward price forecasts h trading days ahead (h = 5, 20, ...), same protocol as walk_forward.py (retrain every 7 days on the latest 365
labelled days, window 20, LSTM(32), seeds averaged) judged against the no-change forecast "price in h days = price today".

Differences forced by h > 1: a label for row j (return from day j to j+h) is only known h days later, so the training rows for a block starting
at s end at s-h; neighbouring labels overlap, so p-values use n_eff = n / h and Diebold-Mariano uses a HAC variance (price_report).

Usage: python walk_forward_horizon.py --data data/processed/xauusd_d1_features.parquet --horizon 20 --out-dir results/horizon_h20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walk_forward import fit_predict  # noqa: E402
from train_models import price_report  # noqa: E402


def add_targets(df: pd.DataFrame, h: int) -> pd.DataFrame:
    df = df.copy()
    df[f"meta_close_h{h}"] = df["meta_close"].shift(-h)
    df[f"y_ret_h{h}"] = np.log(df[f"meta_close_h{h}"] / df["meta_close"])
    df[f"y_dir_h{h}"] = np.where(df[f"y_ret_h{h}"].notna(), (df[f"y_ret_h{h}"] > 0).astype(float), np.nan)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--units", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--retrain-every", type=int, default=7)
    ap.add_argument("--train-len", type=int, default=365)
    args = ap.parse_args()
    h = args.horizon

    df = add_targets(pd.read_parquet(args.data).sort_index(), h)
    feats = [c for c in df.columns if c.startswith("f_")]
    F, y_ret, y_dir = df[feats].to_numpy(float), df[f"y_ret_h{h}"].to_numpy(float), df[f"y_dir_h{h}"].to_numpy(float)
    start = int(np.flatnonzero((df["meta_split"] == "test").to_numpy())[0])
    names = ["drift", "ridge", "hist_gb_reg", "hist_gb_clf", "lstm"]
    preds = {n: (np.zeros(len(df) - start), np.full(len(df) - start, np.nan)) for n in names}

    s, t0, block = start, time.time(), 0
    while s < len(df):
        e = min(s + args.retrain_every, len(df))
        tr = np.arange(max(0, s - args.train_len - h + 1), s - h + 1)  # labels of rows > s-h are not known yet
        res = fit_predict(F, y_ret, y_dir, tr, np.arange(s, e), args)
        for n in names:
            preds[n][0][s - start:e - start] = res[n][0]
            if res[n][1] is not None:
                preds[n][1][s - start:e - start] = res[n][1]
        block += 1
        print(f"block {block}: rows {s}->{e} ({df.index[s].date()}), {time.time() - t0:.0f}s elapsed", flush=True)
        s = e

    ev = df.iloc[start:]
    ok = ev[f"y_ret_h{h}"].notna().to_numpy()  # the last h test rows have no realised label yet
    ev = ev[ok]
    out = {"protocol": {"horizon_days": h, "retrain_every_days": args.retrain_every, "train_len_days": args.train_len, "window": args.window, "lstm_units": args.units,
                        "seeds": args.seeds, "test_start": str(ev.index[0].date()), "test_end": str(ev.index[-1].date()), "n_test": len(ev), "n_eff": len(ev) // h,
                        "note": "labels overlap for h > 1: p-values use n_eff = n / h and a HAC Diebold-Mariano variance"}, "models": {}}
    y_h = ev[f"y_ret_h{h}"].to_numpy()
    drift = preds["drift"][0][ok]
    for n in names:
        pr, pu = preds[n][0][ok], preds[n][1][ok]
        r = price_report(ev, h, None if n == "hist_gb_clf" else pr, None if n in ("drift", "ridge", "hist_gb_reg") else pu)
        if n != "hist_gb_clf":
            sse = float(np.sum((y_h - pr) ** 2))
            r["mape_pct"] = float(np.mean(np.abs(ev["meta_close"].to_numpy() * np.exp(pr) - ev[f"meta_close_h{h}"].to_numpy()) / ev[f"meta_close_h{h}"].to_numpy()) * 100)
            r["r2_os_vs_zero"] = 1 - sse / float(np.sum(y_h**2))
            r["r2_os_vs_rolling_mean"] = 1 - sse / float(np.sum((y_h - drift) ** 2))
        out["models"][n] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = ev[["meta_close", f"meta_close_h{h}", f"y_ret_h{h}", f"y_dir_h{h}"]].copy()
    for n in names:
        table[f"pred_ret_{n}"] = preds[n][0][ok]
        table[f"p_up_{n}"] = preds[n][1][ok]
    table.index.name = "date"
    table.to_csv(args.out_dir / "wf_predictions.csv")
    (args.out_dir / "walk_forward_results.json").write_text(json.dumps(out, indent=2, default=float))
    up = out["models"]["drift"]["direction_from_sign"]["always_up_acc"]
    print(f"\n== WALK-FORWARD h={h} (test {out['protocol']['test_start']} -> {out['protocol']['test_end']}, n={len(ev)}, n_eff={len(ev) // h}) | always-up {up:.3f} | naive MAE ${out['models']['drift']['naive_mae_usd']:.2f} ==")
    for n, r in out["models"].items():
        if "mae_usd" in r:
            print(f"{n:12s} MAE ${r['mae_usd']:7.2f} MAPE {r['mape_pct']:.2f}% skill {r['mae_skill_vs_naive']:+.4f} DMp {r['dm_p_vs_naive']:.3f} IC {r['ic_spearman']:+.3f} (p {r['ic_p']:.2f}) R2os {r['r2_os_vs_rolling_mean']:+.3f}")


if __name__ == "__main__":
    main()
