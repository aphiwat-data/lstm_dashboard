"""Does VIX improve the 90% forecast interval? Scale model sigma_t = exp(a + b*log EWMA_t + c*log VIX_t) for the next-day log-return, fitted by Gaussian
quasi-maximum likelihood on the last 1000 days known at each refit (every 7 days). Compared on the same 389 scored test days as interval_experiment.py.
Usage: python interval_vix.py FEATURES.parquet MACRO_CACHE.parquet  (writes results/interval_vix.json)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_daily_dataset import align_macro
from regression_metrics import Z90, interval_metrics, rolling_bands

WARM, EVERY, FIT_LEN, LAM = 60, 7, 1000, 0.94
df = pd.read_parquet(sys.argv[1]).sort_index(); macro = pd.read_parquet(sys.argv[2])
r = df["y_ret_next"].to_numpy(float); n = len(df); start = int(np.flatnonzero((df["meta_split"] == "test").to_numpy())[0])
vix = align_macro(df.index, macro[["vix"]])["vix"].to_numpy(float)         # close of the same day (or earlier), never later
vix_lag = np.r_[np.nan, vix[:-1]]                                              # conservative: yesterday's VIX
p = pd.read_csv(HERE / "results" / "wf_predictions.csv")
close, y, pred = p["meta_close"].to_numpy(float), p["y_ret_next"].to_numpy(float), p["pred_ret_lstm"].to_numpy(float)

# EWMA sigma for forecast row t uses returns of rows < t only
ew = np.empty(n); v = np.nanvar(r[:250]); 
for t in range(n):
    ew[t] = np.sqrt(v); v = LAM * v + (1 - LAM) * (r[t] ** 2 if np.isfinite(r[t]) else v)
lE = np.log(ew)

def fit(X, target):
    """sigma = exp(X @ theta); minimise the Gaussian negative log-likelihood of `target`."""
    def nll(th): s = X @ th; return float(np.sum(s + 0.5 * target**2 * np.exp(-2 * s)))
    th0 = np.zeros(X.shape[1]); th0[1] = 1.0
    return minimize(nll, th0, method="BFGS").x

def band_sigma(cols, name_vix):
    """sigma for test rows from a scale model with the given design columns (list of arrays, first = ones)"""
    out, coefs = np.zeros(n - start), []
    for s in range(start, n, EVERY):
        e = min(s + EVERY, n); idx = np.arange(max(0, s - FIT_LEN), s)
        idx = idx[np.isfinite(r[idx]) & np.all([np.isfinite(c[idx]) for c in cols], axis=0)]
        th = fit(np.column_stack([c[idx] for c in cols]), r[idx]); coefs.append(th)
        for g in range(s, e): out[g - start] = np.exp(np.dot(th, [c[g] for c in cols]))
    return out, np.array(coefs)

ones = np.ones(n); lV = np.log(vix / 100 / np.sqrt(252)); lVl = np.log(vix_lag / 100 / np.sqrt(252))
variants = {
    "EWMA lambda 0.94 (current)": None,
    "EWMA, intercept+slope refit (no VIX)": [ones, lE],
    "EWMA + VIX (same-day close)": [ones, lE, lV],
    "EWMA + VIX (previous day)": [ones, lE, lVl],
    "VIX only (previous day)": [ones, lVl],
}
sig, coefs = {}, {}
for k, cols in variants.items():
    if cols: sig[k], coefs[k] = band_sigma(cols, k)
bands = {"EWMA lambda 0.94 (current)": rolling_bands(pred, y - pred, "ewma_gaussian")}
for k in sig: bands[k] = ((pred - Z90 * sig[k])[WARM:], (pred + Z90 * sig[k])[WARM:])
rows, scores = [], {}
for name, (lo, hi) in bands.items():
    m = interval_metrics(close, y, lo, hi); c, N = m["coverage"], m["n"]; k = round(c * N)
    lr = -2 * (np.log(0.9 ** k * 0.1 ** (N - k)) - np.log(max(c, 1e-9) ** k * max(1 - c, 1e-9) ** (N - k)))
    lo_p, hi_p, act = close[WARM:] * np.exp(lo), close[WARM:] * np.exp(hi), close[WARM:] * np.exp(y[WARM:])
    scores[name] = (hi_p - lo_p) + 20 * np.maximum(lo_p - act, 0) + 20 * np.maximum(act - hi_p, 0)
    ins = (act >= lo_p) & (act <= hi_p); h = len(ins) // 2
    rows.append({"method": name, **m, "kupiec_p": float(1 - chi2.cdf(lr, 1)), "cov_first_half": float(ins[:h].mean()), "cov_second_half": float(ins[h:].mean())})
base = scores["EWMA lambda 0.94 (current)"]
for rw in rows:
    d = scores[rw["method"]] - base; db = d.mean(); lrv = d.var() + 2 * sum((1 - k / 6) * np.mean((d[k:] - db) * (d[:-k] - db)) for k in range(1, 6))
    rw["score_diff_vs_ewma"] = float(db); rw["score_diff_p"] = float(2 * (1 - norm.cdf(abs(db / np.sqrt(max(lrv, 1e-12) / len(d))))))
    if rw["method"] in coefs: rw["mean_coefs"] = [float(x) for x in coefs[rw["method"]].mean(axis=0)]
(HERE / "results" / "interval_vix.json").write_text(json.dumps(rows, indent=1, default=float))
print(f"{'method':40s} cov    width$ score$  Kupiec p  halves     score vs EWMA (p)   mean coefs (a, b[, c])")
for rw in rows:
    print(f"{rw['method']:40s} {rw['coverage']:.1%}  {rw['mean_width_usd']:5.0f}  {rw['interval_score_usd']:5.0f}   {rw['kupiec_p']:.3f}   {rw['cov_first_half']:.0%}/{rw['cov_second_half']:.0%}    {rw['score_diff_vs_ewma']:+6.1f} ({rw['score_diff_p']:.2f})   {np.round(rw.get('mean_coefs', []), 2)}")
