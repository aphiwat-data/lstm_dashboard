"""E3 (README section 15.8): could a "best" variant be luck, and is anything better than the benchmark after correcting for how many were tried?

PROTOCOL (frozen on 2026-11-28, before the first run on the real forecasts; every choice comes from the literature, none was tuned on the data;
the synthetic tests in test_multiplicity.py are the only runs made before the freeze):

Families (each is tested against its own benchmark on the rows of one window; nothing is pooled across families):
  P1  next-day return forecasts, loss = squared return error, benchmark = no-change (return forecast 0), 449 test days. Candidates (16): the rolling
      drift, and ridge, gradient-boosting regressor and LSTM from each of the five input sets (price only, + macro, + NFP, + FOMC and CPI, + COT).
  P2  the same candidates, loss = absolute price error in dollars (the headline MAE), benchmark = no-change.
  D   next-day direction, loss = 0-1 error, benchmark = always up. Candidates (20): ridge sign, gradient-boosting regressor sign, gradient-boosting
      classifier and LSTM direction head (p_up > 0.5) from each of the five input sets. The momentum rule is not in the prediction files and is left out.
  V1  next-day variance forecasts of E6 on its primary window (428 days), loss = QLIKE against the realised variance, benchmark = EWMA 0.94 (level-matched).
      Candidates (7): the unmatched shipped EWMA, GARCH(1,1), EWMA of RV, HAR-RV levels, HAR-RV logs, HAR-RV-X, LSTM-RV.
  V2  the same forecasts, loss = QLIKE against the squared futures return.
Tests (arch package, stationary bootstrap, expected block length 10, 10,000 replications, seed 20261128):
  White's reality check (non-studentised) and Hansen's SPA (studentised, consistent p-value): H0 = no candidate has a lower expected loss than the benchmark.
  Romano-Wolf StepM at the 5% family-wise level: the set of candidates that beat the benchmark.
  Model confidence set (Hansen, Lunde and Nason; range statistic) at the 10% level over the benchmark and all candidates.
  Per candidate: mean loss difference (benchmark minus candidate, positive = better), Newey-West t (5 lags), one-sided p and its Holm adjustment within the family.
  Robustness: the SPA p-value with expected block lengths 5 and 20. The expected number of nominal one-sided rejections at 5% under the global null is 0.05 K.
All families and tests are reported whatever they show.

Usage: python multiplicity_experiment.py [--check] [--out-dir DIR]   (needs: pip install arch; --check only assembles and counts the candidates)"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SETS = {"price only": "", "+ macro": "with_macro", "+ NFP": "with_macro_calendar", "+ FOMC+CPI": "with_macro_calendar_full", "+ COT": "with_cot"}
PRICE_MODELS = {"ridge": "Ridge", "hist_gb_reg": "Boosting", "lstm": "LSTM"}
DIRECTION_MODELS = {"ridge": "Ridge", "hist_gb_reg": "Boosting regressor", "hist_gb_clf": "Boosting classifier", "lstm": "LSTM"}
BLOCK, REPS, SEED, SIZE_STEPM, SIZE_MCS, HAC_LAG = 10, 10_000, 20261128, 0.05, 0.10, 5


# ------------------------------------------------------------------------------------------------------------------------ losses
def loss_squared_return(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return (y - yhat) ** 2


def loss_absolute_price(close: np.ndarray, next_close: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.abs(close * np.exp(yhat) - next_close)


def loss_direction(up: np.ndarray, p_up: np.ndarray) -> np.ndarray:
    return ((p_up > 0.5) != (up == 1)).astype(float)


def qlike(target: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.log(f) + target / f


# ----------------------------------------------------------------------------------------------------------------------- helpers
def hac_t(d: np.ndarray, lag: int = HAC_LAG) -> float:
    db = d.mean()
    lrv = d.var() + 2 * sum((1 - j / (lag + 1)) * np.mean((d[j:] - db) * (d[:-j] - db)) for j in range(1, lag + 1))
    return float(db / np.sqrt(max(lrv, 1e-18) / len(d)))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (k - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def drop_degenerate(bench: np.ndarray, cands: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """A candidate whose loss equals the benchmark's on every day is the benchmark itself, and one equal to an earlier candidate adds nothing; both make the
    studentised statistics undefined, so they are dropped and listed."""
    keep, dropped = [], []
    for c in cands.columns:
        x = cands[c].to_numpy()
        if np.allclose(x, bench):
            dropped.append([c, "identical to the benchmark"])
            continue
        twin = next((k for k in keep if np.allclose(x, cands[k].to_numpy())), None)
        if twin is not None:
            dropped.append([c, f"identical to {twin}"])
            continue
        keep.append(c)
    return cands[keep], dropped


def family_tests(bench: np.ndarray, cands: pd.DataFrame, block: int = BLOCK, reps: int = REPS, seed: int = SEED, with_sensitivity: bool = True) -> dict:
    """White's reality check, Hansen's SPA, StepM and the model confidence set for one family; losses are 'lower is better'."""
    from arch.bootstrap import MCS, SPA, StepM

    cands, dropped = drop_degenerate(bench, cands)
    spa = SPA(bench, cands, block_size=block, reps=reps, bootstrap="stationary", seed=seed, studentize=True)
    spa.compute()
    rc = SPA(bench, cands, block_size=block, reps=reps, bootstrap="stationary", seed=seed, studentize=False)
    rc.compute()
    step = StepM(bench, cands, size=SIZE_STEPM, block_size=block, reps=reps, bootstrap="stationary", seed=seed)
    step.compute()
    allloss = cands.copy()
    allloss.insert(0, "BENCHMARK", bench)
    mcs = MCS(allloss, size=SIZE_MCS, block_size=block, reps=reps, bootstrap="stationary", seed=seed)
    mcs.compute()
    out = {"k": int(cands.shape[1]), "n": int(len(bench)), "dropped_degenerate": dropped, "spa_p": {k: float(v) for k, v in spa.pvalues.items()}, "reality_check_p": float(rc.pvalues["upper"]),
           "stepm_superior": [str(m) for m in step.superior_models], "mcs_included": [str(m) for m in mcs.included], "mcs_excluded": [str(m) for m in mcs.excluded],
           "mcs_pvalues": {str(k): float(v) for k, v in mcs.pvalues["Pvalue"].items()}}
    if with_sensitivity:
        out["spa_consistent_p_by_block"] = {}
        for b in (5, 20):
            s = SPA(bench, cands, block_size=b, reps=reps, bootstrap="stationary", seed=seed, studentize=True)
            s.compute()
            out["spa_consistent_p_by_block"][str(b)] = float(s.pvalues["consistent"])
        out["spa_consistent_p_by_block"][str(block)] = out["spa_p"]["consistent"]
    t = {c: hac_t(bench - cands[c].to_numpy()) for c in cands.columns}
    p1 = np.array([1 - norm.cdf(t[c]) for c in cands.columns])
    adj = holm(p1)
    out["candidates"] = [{"name": c, "mean_loss": float(cands[c].mean()), "benchmark_mean_loss": float(bench.mean()), "mean_diff": float((bench - cands[c].to_numpy()).mean()), "hac_t": t[c], "p_one_sided": float(p1[i]), "holm_p": float(adj[i]),
                          "mcs_p": out["mcs_pvalues"].get(str(c))} for i, c in enumerate(cands.columns)]
    out["nominal_rejections_5pct"] = int((p1 < 0.05).sum())
    out["expected_rejections_under_null"] = float(0.05 * len(p1))
    return out


# ------------------------------------------------------------------------------------------------------------------------ data
def load_sets() -> dict[str, pd.DataFrame]:
    sets = {}
    for label, folder in SETS.items():
        p = pd.read_csv(RESULTS / folder / "wf_predictions.csv", parse_dates=["date"]) if folder else pd.read_csv(RESULTS / "wf_predictions.csv", parse_dates=["date"])
        sets[label] = p
    ref = sets["price only"]
    for label, p in sets.items():
        assert (p["date"].to_numpy() == ref["date"].to_numpy()).all() and np.allclose(p["meta_close"], ref["meta_close"]) and np.allclose(p["y_ret_next"], ref["y_ret_next"]), f"rows differ in {label}"
    return sets


def build_price_family(sets: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ref = sets["price only"]
    y, close, nxt = ref["y_ret_next"].to_numpy(float), ref["meta_close"].to_numpy(float), ref["meta_next_close"].to_numpy(float)
    preds = {"Drift": ref["pred_ret_drift"].to_numpy(float)}
    for label, p in sets.items():
        for key, name in PRICE_MODELS.items():
            preds[f"{name} [{label}]"] = p[f"pred_ret_{key}"].to_numpy(float)
    assert all(np.isfinite(v).all() for v in preds.values()), "a return forecast has missing values"
    p1 = pd.DataFrame({k: loss_squared_return(y, v) for k, v in preds.items()})
    p2 = pd.DataFrame({k: loss_absolute_price(close, nxt, v) for k, v in preds.items()})
    bench = {"P1": loss_squared_return(y, np.zeros(len(y))), "P2": loss_absolute_price(close, nxt, np.zeros(len(y)))}
    return bench, p1, p2


HEAD_MODELS = {"hist_gb_clf", "lstm"}  # models with a direction head; ridge and the boosting regressor have no probability, their direction is the sign of the return forecast


def direction_probability(p: pd.DataFrame, key: str) -> np.ndarray:
    if key in HEAD_MODELS:
        return p[f"p_up_{key}"].to_numpy(float)
    return np.where(p[f"pred_ret_{key}"].to_numpy(float) > 0, 0.999, 0.001)


def build_direction_family(sets: dict[str, pd.DataFrame]) -> tuple[np.ndarray, pd.DataFrame]:
    ref = sets["price only"]
    up = ref["y_dir_next"].to_numpy(int)
    bench = loss_direction(up, np.ones(len(up)))
    cands = {}
    for label, p in sets.items():
        for key, name in DIRECTION_MODELS.items():
            pu = direction_probability(p, key)
            assert np.isfinite(pu).all(), f"{key} [{label}]: direction forecast has missing values"
            cands[f"{name} [{label}]"] = loss_direction(up, pu)
    return bench, pd.DataFrame(cands)


def build_volatility_family() -> tuple[dict, dict]:
    v = pd.read_csv(RESULTS / "paper_e6" / "e6_primary_forecasts.csv")
    bench_name = "EWMA 0.94 (matched)"
    names = [c for c in v.columns if c not in ("date", "rv_next", "r2_next", bench_name)]
    assert len(names) == 7 and np.isfinite(v[[bench_name, *names, "rv_next", "r2_next"]].to_numpy()).all()
    out_b, out_c = {}, {}
    for tag, col in (("V1", "rv_next"), ("V2", "r2_next")):
        out_b[tag] = qlike(v[col].to_numpy(float), v[bench_name].to_numpy(float))
        out_c[tag] = pd.DataFrame({n: qlike(v[col].to_numpy(float), v[n].to_numpy(float)) for n in names})
    return out_b, out_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "paper_e3")
    ap.add_argument("--check", action="store_true", help="only assemble the families and print their sizes")
    args = ap.parse_args()
    t0 = time.time()
    sets = load_sets()
    bench_p, p1, p2 = build_price_family(sets)
    bench_d, d = build_direction_family(sets)
    bench_v, cands_v = build_volatility_family()
    sizes = {"P1": (len(bench_p["P1"]), p1.shape[1]), "P2": (len(bench_p["P2"]), p2.shape[1]), "D": (len(bench_d), d.shape[1]), "V1": (len(bench_v["V1"]), cands_v["V1"].shape[1]), "V2": (len(bench_v["V2"]), cands_v["V2"].shape[1])}
    if args.check:
        print("families assembled (rows, candidates):", sizes)
        return
    families = {"P1": (bench_p["P1"], p1), "P2": (bench_p["P2"], p2), "D": (bench_d, d), "V1": (bench_v["V1"], cands_v["V1"]), "V2": (bench_v["V2"], cands_v["V2"])}
    out = {"protocol": {"block": BLOCK, "reps": REPS, "seed": SEED, "stepm_size": SIZE_STEPM, "mcs_size": SIZE_MCS, "hac_lag": HAC_LAG}, "families": {}}
    for name, (b, c) in families.items():
        out["families"][name] = family_tests(b, c)
        print(f"{name}: K={c.shape[1]} n={len(b)} SPA(consistent)={out['families'][name]['spa_p']['consistent']:.3f} StepM superior={out['families'][name]['stepm_superior']} ({time.time() - t0:.0f}s)")
    out["runtime_seconds"] = round(time.time() - t0)
    out["arch_version"] = __import__("arch").__version__
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "e3_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("written", args.out_dir)


if __name__ == "__main__":
    main()
