"""E2b (README section 15.8): placebo calibration of the E2 selection pipeline.

Question. When gold returns are unpredictable BY CONSTRUCTION, how often does the E2 decision rule call a result positive, how large is the TEST R-squared of
the two primary choices (P-A, the forecaster picked on DEV; P-B, the online selector), and how much would a naive "best forecaster on the evaluation window"
overstate? The last number tells the reader how much of an exploratory leaderboard to believe.

PROTOCOL (frozen on 2026-11-26, before the first run of this script on the real inputs; the unit test in test_search.py::test_placebo_is_unpredictable is the only
earlier run, on synthetic prices):
Placebo gold. r*_t = s_t * e_pi(t): s_t is the RiskMetrics volatility (EWMA, lambda 0.94, of the real squared returns up to t - 1, so it keeps the real volatility
clustering and level), e_u = r_u / s_u the real standardised returns, pi a random permutation (so the placebo returns are exchangeable given their volatility and have
the real mean, skewness and tails). The six macro series and the calendar are the real ones, so every relation between the placebo return and any feature is
zero by construction. Gold's own features are recomputed from the placebo prices.
Pipeline. Exactly E2 on h = 1 with a reduced family so that many replicates are affordable: ridge (alpha 1000), elastic net (alpha 0.1) and one-component PLS on the nine
feature sets and the three training lengths (81 forecasters, 162 with the shrunk versions), the same blocks, benchmark, shrinkage, online selectors, windows
(DEV 2011-2019, TEST 2020 onwards) and rules as in E2. No recurrent or tree models (too slow for replicates).
Reported over the replicates: the share of replicates in which P-A and P-B are called positive (TEST R-squared against b above 0 and Clark-West p < 0.025; the
nominal one-sided rate is 0.025 each), the mean and quartiles of their TEST R-squared and Clark-West p, the share in which the BEST forecaster on TEST (selected on the
evaluation window, the exploratory leaderboard's habit) has a positive R-squared and Clark-West p < 0.05, and the mean R-squared of that best forecaster.
Added before the first run (the E2 amendment): the IC p-value, the R-squared against the expanding historical mean and the STRICT call (positive by the rule above AND IC p < 0.05
AND R-squared against the expanding mean > 0) are recorded for P-A and P-B, because a two-replicate test of this script showed a "positive" call in a pure null (a long-memory mean
beats the 365-day mean). Replicates: 40 (seeds 20261126 + i); no replicate is dropped or added after the results are seen.

Usage: python search_placebo.py [--reps 40] [--workers 3] [--out-dir results/paper_e2]   (writes e2b_placebo.json and e2b_placebo.csv)"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per replicate: the replicates run in parallel

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search_experiment as se  # noqa: E402

LAM = 0.94
SPECS = [se.Spec(f"{fam}[{par}]|{s}|{L}", fam, (par,), s, L) for fam, par in (("ridge", 1000), ("enet", 0.1), ("pls", 1)) for s in "PMCAXTRVK" for L in se.LENGTHS]


def placebo_prices(prices: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Gold-like prices whose returns are exchangeable given the real volatility path (see the protocol)."""
    p = prices[["date", "close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    p = p[p["close"] > 0].reset_index(drop=True)
    r = np.log(p["close"]).diff().to_numpy()[1:]
    v, s2 = float(np.var(r[:50])), np.empty(len(r))
    for t in range(len(r)):
        s2[t] = v                                  # variance forecast for day t from returns up to t - 1
        v = LAM * v + (1 - LAM) * r[t] ** 2
    s = np.sqrt(s2)
    e = r / s
    r_star = s * e[rng.permutation(len(e))]
    out = p.copy()
    out["close"] = float(p["close"].iloc[0]) * np.exp(np.r_[0.0, np.cumsum(r_star)])
    return out


def one_replicate(args: tuple) -> dict:
    i, prices, macro = args
    rng = np.random.default_rng(se.SEED + i)
    panel = se.build_panel(placebo_prices(prices, rng), macro)
    T, h = len(panel.dates), 1
    starts = se.block_starts(T, panel.first)
    y, close, cl = panel.y[h], panel.close, panel.cl[h]
    mats = {k: panel.X[se.SETS[k]].to_numpy(float) for k in {sp.fset for sp in SPECS}}
    F = np.full((T, len(SPECS)), np.nan)
    for s in starts:
        e = min(s + se.BLOCK, T)
        for k, sp in enumerate(SPECS):
            F[s:e, k] = se.forecast_block(sp, mats[sp.fset], y, s, e)
    b = se.rolling_mean_benchmark(y, h, starts, T)
    b2 = se.expanding_mean_benchmark(y, h, starts, T)
    Fall = np.hstack([F, se.shrink(F, y, b, h, starts)])
    top1, top5, _ = se.online_select(Fall, y, b, h, starts)
    lab = np.isfinite(y)
    dev = se.window_mask(panel.dates, se.DEV[0], se.DEV[1]) & lab
    dev &= np.arange(T) >= np.flatnonzero(np.isfinite(Fall).all(1) & np.isfinite(top1))[0]
    test = se.window_mask(panel.dates, se.TEST_START, None) & lab

    def sc(f, m):
        return se.score(np.where(m, y, np.nan), np.where(m, f, np.nan), np.where(m, b, np.nan), close, cl, h, np.where(m, b2, np.nan))

    dev_r2 = np.array([sc(Fall[:, k], dev)["r2_vs_b"] for k in range(Fall.shape[1])])
    best_dev = int(np.nanargmax(dev_r2))
    test_scores = [sc(Fall[:, k], test) for k in range(Fall.shape[1])]
    best_test = int(np.nanargmax([t["r2_vs_b"] for t in test_scores]))
    pa, pb, p5 = test_scores[best_dev], sc(top1, test), sc(top5, test)
    bt = test_scores[best_test]
    pos = lambda c: bool(c["r2_vs_b"] > 0 and c["cw_p"] < se.PRIMARY_LEVEL)  # noqa: E731
    strict = lambda c: bool(pos(c) and c["ic_p"] < 0.05 and c["r2_vs_expmean"] > 0)  # noqa: E731
    return {"rep": i, "n_test": pa["n"], "PA_r2": pa["r2_vs_b"], "PA_cw_p": pa["cw_p"], "PA_positive": pos(pa), "PA_ic_p": pa["ic_p"], "PA_r2_expmean": pa["r2_vs_expmean"], "PA_strict": strict(pa), "PB_ic_p": pb["ic_p"], "PB_r2_expmean": pb["r2_vs_expmean"], "PB_strict": strict(pb), "PB_r2": pb["r2_vs_b"], "PB_cw_p": pb["cw_p"], "PB_positive": pos(pb),
            "P5_r2": p5["r2_vs_b"], "P5_cw_p": p5["cw_p"], "best_on_test_r2": bt["r2_vs_b"], "best_on_test_cw_p": bt["cw_p"],
            "best_on_test_positive_05": bool(bt["r2_vs_b"] > 0 and bt["cw_p"] < 0.05), "n_positive_r2_of_pool": int(sum(t["r2_vs_b"] > 0 for t in test_scores))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=se.OUT)
    a = ap.parse_args()
    prices, macro = pd.read_parquet(se.PRICES), pd.read_parquet(se.MACRO)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one_replicate, [(i, prices, macro) for i in range(a.reps)]):
            rows.append(r)
            print(f"replicate {r['rep']}: P-A R2 {r['PA_r2']:+.5f} p {r['PA_cw_p']:.3f} | P-B R2 {r['PB_r2']:+.5f} p {r['PB_cw_p']:.3f} | best-on-test R2 {r['best_on_test_r2']:+.5f} ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out_dir / "e2b_placebo.csv", index=False)
    q = lambda c: [float(np.percentile(d[c], x)) for x in (25, 50, 75)]  # noqa: E731
    summary = {"reps": len(d), "forecasters": len(SPECS), "pool_with_shrunk": 2 * len(SPECS), "nominal_rate_per_primary_test": se.PRIMARY_LEVEL,
               "PA_positive_share": float(d["PA_positive"].mean()), "PB_positive_share": float(d["PB_positive"].mean()),
               "PA_strict_share": float(d["PA_strict"].mean()), "PB_strict_share": float(d["PB_strict"].mean()),
               "PA_r2_vs_expmean_mean": float(d["PA_r2_expmean"].mean()), "PB_r2_vs_expmean_mean": float(d["PB_r2_expmean"].mean()),
               "PA_ic_p_below_0.05_share": float((d["PA_ic_p"] < 0.05).mean()), "PB_ic_p_below_0.05_share": float((d["PB_ic_p"] < 0.05).mean()),
               "PA_r2_mean": float(d["PA_r2"].mean()), "PA_r2_quartiles": q("PA_r2"), "PB_r2_mean": float(d["PB_r2"].mean()), "PB_r2_quartiles": q("PB_r2"),
               "PA_share_r2_positive": float((d["PA_r2"] > 0).mean()), "PB_share_r2_positive": float((d["PB_r2"] > 0).mean()),
               "PA_cw_p_median": float(d["PA_cw_p"].median()), "PB_cw_p_median": float(d["PB_cw_p"].median()),
               "best_on_test_r2_mean": float(d["best_on_test_r2"].mean()), "best_on_test_r2_quartiles": q("best_on_test_r2"),
               "best_on_test_positive_at_0.05_share": float(d["best_on_test_positive_05"].mean()), "best_on_test_cw_p_median": float(d["best_on_test_cw_p"].median()),
               "pool_share_with_positive_test_r2_mean": float(d["n_positive_r2_of_pool"].mean() / (2 * len(SPECS)))}
    (a.out_dir / "e2b_placebo.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
