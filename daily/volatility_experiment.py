"""Volatility forecasting for the next-day gold return, and what it does to the 90% price interval.

Part 1 scores variance forecasts of r_{t+1}^2 on the 449 test days with QLIKE (Patton 2011; lower is better) and a Diebold-Mariano test (HAC lag 5)
against EWMA(0.94): EWMA 0.94 / 0.97, GARCH(1,1), scaled VIX, and log-linear scale models that add VIX and a slower EWMA.
Part 2 builds 90% bands from the best volatility forecasts with empirical quantiles of past standardised errors (no Gaussian assumption) and scores
them like interval_experiment.py (389 days after a 60-day warm-up).
Usage: python volatility_experiment.py FEATURES.parquet MACRO_CACHE.parquet  (writes results/volatility_comparison.json and volatility_forecast.csv)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_daily_dataset import align_macro
from regression_metrics import interval_metrics

EVERY, FIT_LEN, WARM = 7, 1000, 60
df = pd.read_parquet(sys.argv[1]).sort_index(); macro = pd.read_parquet(sys.argv[2])
r = df["y_ret_next"].to_numpy(float); n = len(df); start = int(np.flatnonzero((df["meta_split"] == "test").to_numpy())[0]); m = n - start
vix = align_macro(df.index, macro[["vix"]])["vix"].to_numpy(float); vix_lag = np.r_[np.nan, vix[:-1]]     # yesterday's VIX: strictly known
vvar = (vix_lag / 100) ** 2 / 252                                                                         # VIX as a daily variance
p = pd.read_csv(HERE / "results" / "wf_predictions.csv")
close, y, pred = p["meta_close"].to_numpy(float), p["y_ret_next"].to_numpy(float), p["pred_ret_lstm"].to_numpy(float)

def ewma_sigma(lam):
    out, v = np.empty(n), np.nanvar(r[:250])
    for t in range(n):
        out[t] = np.sqrt(v); v = lam * v + (1 - lam) * (r[t] ** 2 if np.isfinite(r[t]) else v)
    return out
E94, E97, E985 = ewma_sigma(0.94), ewma_sigma(0.97), ewma_sigma(0.985)

def filt(e, om, a, b):
    v = np.empty(len(e) + 1); v[0] = e.var()
    for t in range(len(e)): v[t + 1] = om + a * e[t] ** 2 + b * v[t]
    return v
def fit_garch(x):
    mu = x.mean(); e = x - mu; v0 = e.var()
    def unpack(q): rho = 1 / (1 + np.exp(-q[0])); u = 1 / (1 + np.exp(-q[1])); return v0 * (1 - rho) * np.exp(q[2]), rho * u, rho * (1 - u)
    nll = lambda q: 0.5 * np.sum(np.log(filt(e, *unpack(q))[:-1]) + e ** 2 / filt(e, *unpack(q))[:-1])
    best = min((minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400}) for x0 in ([2.5, -1.5, 0.0], [3.5, -2.0, 0.0])), key=lambda o: o.fun)
    return mu, unpack(best.x)
def fit_scale(X, target):
    nll = lambda th: float(np.sum(X @ th + 0.5 * target**2 * np.exp(-2 * (X @ th))))
    th0 = np.zeros(X.shape[1]); th0[1] = 1.0
    return minimize(nll, th0, method="BFGS").x

ones = np.ones(n); lE94, lE985, lV = np.log(E94), np.log(E985), 0.5 * np.log(vvar)
def refit(design):
    out = np.zeros(m)
    for s in range(start, n, EVERY):
        e = min(s + EVERY, n); idx = np.arange(max(0, s - FIT_LEN), s); idx = idx[np.isfinite(r[idx]) & np.all([np.isfinite(c[idx]) for c in design], axis=0)]
        th = fit_scale(np.column_stack([c[idx] for c in design]), r[idx])
        for g in range(start if False else s, e): out[g - start] = np.exp(np.dot(th, [c[g] for c in design]))
    return out
sig = {"EWMA 0.94 (current)": E94[start:], "EWMA 0.97": E97[start:]}
g_sig = np.zeros(m)
for s in range(start, n, EVERY):
    e_ = min(s + EVERY, n); lo = max(0, s - FIT_LEN); mu, (om, a, b) = fit_garch(r[lo:s]); v = filt(r[lo:e_] - mu, om, a, b)
    for g in range(s, e_): g_sig[g - start] = np.sqrt(v[g - lo])
sig["GARCH(1,1)"] = g_sig
k = np.array([np.nanmean(r[max(0, g - FIT_LEN):g] ** 2) / np.nanmean(vvar[max(0, g - FIT_LEN):g]) for g in range(start, n)])
sig["VIX scaled (ratio refit)"] = np.sqrt(k * vvar[start:])
sig["EWMA 0.94 + VIX"] = refit([ones, lE94, lV])
sig["EWMA 0.94 + EWMA 0.985"] = refit([ones, lE94, lE985])
sig["EWMA 0.94 + EWMA 0.985 + VIX"] = refit([ones, lE94, lE985, lV])
r2 = r[start:] ** 2

def hac_p(d, lag=5):
    db = d.mean(); lrv = d.var() + 2 * sum((1 - j / (lag + 1)) * np.mean((d[j:] - db) * (d[:-j] - db)) for j in range(1, lag + 1))
    return float(2 * (1 - norm.cdf(abs(db / np.sqrt(max(lrv, 1e-18) / len(d))))))
qlike = {k_: np.log(s_ ** 2) + r2 / s_ ** 2 for k_, s_ in sig.items()}
base = qlike["EWMA 0.94 (current)"]
part1 = [{"model": k_, "qlike": float(q.mean()), "qlike_vs_ewma": float((q - base).mean()), "dm_p_vs_ewma": hac_p(q - base) if k_ != "EWMA 0.94 (current)" else None,
          "mse_variance_1e8": float(np.mean((s_ ** 2 - r2) ** 2) * 1e8), "mean_sigma_pct": float(s_.mean() * 100)} for (k_, q), s_ in zip(qlike.items(), sig.values())]

# ---- part 2: 90% bands from sigma forecasts with empirical quantiles of past standardised errors (centre = LSTM forecast)
def band(sg, empirical):
    z = (y - pred) / sg; lo, hi = np.zeros(m - WARM), np.zeros(m - WARM)
    for i in range(WARM, m):
        q5, q95 = np.percentile(z[:i], [5, 95]) if empirical else (-1.6449, 1.6449)
        lo[i - WARM], hi[i - WARM] = pred[i] + q5 * sg[i], pred[i] + q95 * sg[i]
    return lo, hi
part2, scores = [], {}
for name in ("EWMA 0.94 (current)", "EWMA 0.94 + VIX", "EWMA 0.94 + EWMA 0.985 + VIX", "GARCH(1,1)"):
    for emp in (False, True):
        lo, hi = band(sig[name], emp); mm = interval_metrics(close, y, lo, hi); c, N = mm["coverage"], mm["n"]; kk = round(c * N)
        lr = -2 * (np.log(0.9 ** kk * 0.1 ** (N - kk)) - np.log(max(c, 1e-9) ** kk * max(1 - c, 1e-9) ** (N - kk)))
        lo_p, hi_p, act = close[WARM:] * np.exp(lo), close[WARM:] * np.exp(hi), close[WARM:] * np.exp(y[WARM:])
        label = f"{name}, {'empirical z' if emp else 'Gaussian z'}"; scores[label] = (hi_p - lo_p) + 20 * np.maximum(lo_p - act, 0) + 20 * np.maximum(act - hi_p, 0)
        part2.append({"method": label, **mm, "kupiec_p": float(1 - chi2.cdf(lr, 1))})
b0 = scores["EWMA 0.94 (current), Gaussian z"]
for row in part2: d = scores[row["method"]] - b0; row["score_diff_vs_current"] = float(d.mean()); row["score_diff_p"] = hac_p(d)
(HERE / "results" / "volatility_comparison.json").write_text(json.dumps({"variance_forecasts": part1, "intervals": part2}, indent=1, default=float))
best = min(part1, key=lambda x: x["qlike"])["model"]
pd.DataFrame({"date": p["date"], "abs_return_pct": np.abs(y) * 100, "vix_prev_close": vix_lag[start:], "sigma_ewma_pct": sig["EWMA 0.94 (current)"] * 100,
              "sigma_best_pct": sig[best] * 100}).to_csv(HERE / "results" / "volatility_forecast.csv", index=False)
print("best QLIKE:", best); print(f"{'variance forecast':34s} QLIKE   vs EWMA  DM p   MSE(var)e8  mean sigma%")
for r_ in part1: print(f"{r_['model']:34s} {r_['qlike']:.4f} {r_['qlike_vs_ewma']:+.4f} {'' if r_['dm_p_vs_ewma'] is None else format(r_['dm_p_vs_ewma'], '.3f'):>6s}  {r_['mse_variance_1e8']:8.2f}   {r_['mean_sigma_pct']:.3f}")
print(f"\n{'interval':56s} cov    width  score  Kupiec  score vs current (p)")
for r_ in part2: print(f"{r_['method']:56s} {r_['coverage']:.1%} {r_['mean_width_usd']:6.0f} {r_['interval_score_usd']:5.0f}  {r_['kupiec_p']:.3f}   {r_['score_diff_vs_current']:+6.1f} ({r_['score_diff_p']:.2f})")
