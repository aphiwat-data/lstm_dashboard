"""
Baseline models for next-bar direction on the hourly dataset.

Protocol (fixed before running, no tuning on test):
  * fit on meta_split == "train" only; scaler/imputer are inside the pipeline so they are fit on train only
  * report val and test; the "always up" rule (train majority class) is the bar to beat, not 50%
  * accuracy comes with a Wilson 95% CI and a two-sided binomial test against the always-up accuracy on that split
  * control: same model fit on label-shuffled train data must land near always-up, otherwise something leaks

Usage: python baseline.py --data data/processed_gold/xauusd_h1_features.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COVERAGES = (1.0, 0.5, 0.2, 0.1, 0.05)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def score(y: np.ndarray, p_up: np.ndarray, overlap: int = 1) -> dict:
    """overlap = label horizon in bars: consecutive labels share bars, so CI/p use n_eff = n / overlap."""
    n = len(y)
    always_up = float(y.mean())
    correct = int(((p_up > 0.5) == (y == 1)).sum())
    n_eff = max(int(round(n / overlap)), 1)
    k_eff = int(round(correct / n * n_eff))
    lo, hi = wilson(k_eff, n_eff)
    out = {
        "n": n,
        "n_eff": n_eff,
        "acc": correct / n,
        "ci95": [lo, hi],
        "always_up_acc": always_up,
        "p_vs_always_up": binomtest(k_eff, n_eff, always_up).pvalue,
    }
    if 0 < len(np.unique(p_up)) and len(np.unique(y)) == 2 and len(np.unique(p_up)) > 2:
        out["auc"] = float(roc_auc_score(y, p_up))
    return out


def selective(y: np.ndarray, p_up: np.ndarray, overlap: int = 1) -> list[dict]:
    conf = np.abs(p_up - 0.5)
    rows = []
    for cov in COVERAGES:
        keep = conf >= np.quantile(conf, 1 - cov)
        yk, pk = y[keep], p_up[keep]
        k = int(((pk > 0.5) == (yk == 1)).sum())
        n_eff = max(int(round(len(yk) / overlap)), 1)
        lo, hi = wilson(int(round(k / len(yk) * n_eff)), n_eff)
        rows.append({"coverage": cov, "n": int(len(yk)), "acc": k / len(yk), "ci95": [lo, hi], "always_up_acc_same_rows": float(yk.mean())})
    return rows


def make_models(seed: int = 0) -> dict:
    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "hist_gb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    feats = [c for c in df.columns if c.startswith("f_")]
    parts = {s: g for s, g in df.groupby("meta_split")}
    Xtr, ytr = parts["train"][feats], parts["train"]["y_dir_next"].to_numpy()
    results: dict = {"features": feats, "n_train": len(ytr), "train_share_up": float(ytr.mean())}

    for split in ("val", "test"):
        y = parts[split]["y_dir_next"].to_numpy()
        results[split] = {"always_up": score(y, np.ones(len(y)))}
        mom = (parts[split]["f_ret_1"] > 0).astype(float).to_numpy()
        results[split]["momentum_rule"] = score(y, mom)
        results[split]["reversal_rule"] = score(y, 1 - mom)

    fitted = {}
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        for split in ("val", "test"):
            g = parts[split]
            p = model.predict_proba(g[feats])[:, 1]
            results[split][name] = score(g["y_dir_next"].to_numpy(), p)
            results[split][name]["selective"] = selective(g["y_dir_next"].to_numpy(), p)
            contiguous = (g["meta_gap_to_next_min"] <= 60).to_numpy()
            results[split][name]["acc_contiguous_targets"] = score(g["y_dir_next"].to_numpy()[contiguous], p[contiguous])["acc"]
            results[split][name]["acc_gap_spanning_targets"] = score(g["y_dir_next"].to_numpy()[~contiguous], p[~contiguous])["acc"]

    rng = np.random.default_rng(0)
    control = make_models()["hist_gb"].fit(Xtr, rng.permutation(ytr))
    for split in ("val", "test"):
        g = parts[split]
        results[split]["control_shuffled_labels_hist_gb"] = score(g["y_dir_next"].to_numpy(), control.predict_proba(g[feats])[:, 1])

    print(f"train rows {len(ytr):,} | train share up {ytr.mean():.4f} | features {len(feats)}")
    for split in ("val", "test"):
        print(f"\n== {split.upper()} (n={results[split]['always_up']['n']:,}) ==")
        print(f"{'model':34s} {'acc':>7s} {'95% CI':>17s} {'vs always-up':>13s} {'p-value':>8s} {'AUC':>6s}")
        for name, r in results[split].items():
            ci = f"[{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]"
            auc = f"{r['auc']:.3f}" if "auc" in r else "  -  "
            print(f"{name:34s} {r['acc']:7.4f} {ci:>17s} {r['acc'] - r['always_up_acc']:+13.4f} {r['p_vs_always_up']:8.3f} {auc:>6s}")
    for name in ("logreg", "hist_gb"):
        print(f"\n{name} - test accuracy when keeping only the most confident predictions:")
        for row in results["test"][name]["selective"]:
            print(f"  coverage {row['coverage']:>4.0%}  n={row['n']:>6,}  acc={row['acc']:.4f}  CI=[{row['ci95'][0]:.4f},{row['ci95'][1]:.4f}]  always-up on same rows={row['always_up_acc_same_rows']:.4f}")

    out = args.out or args.data.with_name("baseline_results.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
