"""Usage: python interval_experiment.py FEATURES.parquet OUT_DIR  (needs OUT_DIR/lstm_quantile_bands.npz from lstm_quantile.py; writes results/interval_comparison.json)
Compare 90% forecast-interval methods on the same 389 scored test days (after a 60-day warm-up). All bands use only information known before each forecast."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_metrics import Z90, interval_metrics, rolling_bands

FEATURES, SCR = Path(sys.argv[1]), Path(sys.argv[2]); WARM, EVERY, FIT_LEN = 60, 7, 1000
df = pd.read_parquet(FEATURES).sort_index()
r_all = df["y_ret_next"].to_numpy(float); start = int(np.flatnonzero((df["meta_split"] == "test").to_numpy())[0]); n = len(df)
p = pd.read_csv(str(Path(__file__).resolve().parent / "results" / "wf_predictions.csv"))
close, y, pred = p["meta_close"].to_numpy(float), p["y_ret_next"].to_numpy(float), p["pred_ret_lstm"].to_numpy(float)
errors = y - pred

# ---------------- GARCH(1,1), Gaussian quasi-ML, constant mean; params refit every 7 days on the last 1000 returns known at that time
def filt(e, om, a, b):
    v = np.empty(len(e) + 1); v[0] = e.var()
    for t in range(len(e)): v[t + 1] = om + a * e[t] ** 2 + b * v[t]
    return v          # v[t] = variance forecast for step t from errors before t; v[len(e)] = next step
def fit_garch(r):
    mu = r.mean(); e = r - mu; v0 = e.var()
    def unpack(q):
        rho = 1 / (1 + np.exp(-q[0])); u = 1 / (1 + np.exp(-q[1])); return v0 * (1 - rho) * np.exp(q[2]), rho * u, rho * (1 - u)
    def nll(q):
        om, a, b = unpack(q); v = filt(e, om, a, b)[:-1]; return 0.5 * np.sum(np.log(v) + e ** 2 / v)
    best = min((minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400}) for x0 in ([2.5, -1.5, 0.0], [3.5, -2.0, 0.0])), key=lambda o: o.fun)
    return mu, unpack(best.x)
sig_g, z_fhs = np.zeros(n - start), np.zeros((n - start, 2))
for s in range(start, n, EVERY):
    e_ = min(s + EVERY, n); lo = max(0, s - FIT_LEN)
    mu, (om, a, b) = fit_garch(r_all[lo:s]); eps = r_all[lo:e_] - mu
    v = filt(eps, om, a, b)                       # v[k] uses errors before index lo+k
    zfit = eps[:s - lo] / np.sqrt(v[:s - lo]); q5, q95 = np.percentile(zfit, [5, 95])
    for g in range(s, e_): sig_g[g - start] = np.sqrt(v[g - lo]); z_fhs[g - start] = (q5, q95)
    print("garch", df.index[s].date(), round(a, 3), round(b, 3), flush=True) if (s - start) % 98 == 0 else None
# ---------------- assemble bands for days WARM..N-1 (log-return bounds)
def sl(x): return x[WARM:]
bands = {}
bands["EWMA Gaussian, lambda 0.94 (current)"] = rolling_bands(pred, errors, "ewma_gaussian")
bands["Empirical quantiles, all past errors"] = rolling_bands(pred, errors, "empirical")
bands["GARCH(1,1), Gaussian"] = (sl(pred - Z90 * sig_g), sl(pred + Z90 * sig_g))
bands["GARCH(1,1), filtered historical sim."] = (sl(pred + z_fhs[:, 0] * sig_g), sl(pred + z_fhs[:, 1] * sig_g))
bands["No-change centre, EWMA (reference)"] = rolling_bands(np.zeros(len(y)), y, "ewma_gaussian")
q = np.load(SCR / "lstm_quantile_bands.npz") if (SCR / "lstm_quantile_bands.npz").exists() else None
if q is not None: bands["LSTM quantile (pinball loss)"] = (sl(q["lo"]), sl(q["hi"]))
rows, scores = [], {}
for name, (lo, hi) in bands.items():
    m = interval_metrics(close, y, lo, hi); c = m["coverage"]; k = round(c * m["n"])
    p0 = 0.9; lr = -2 * (np.log(p0 ** k * (1 - p0) ** (m["n"] - k)) - np.log(max(c, 1e-9) ** k * max(1 - c, 1e-9) ** (m["n"] - k)))
    lo_p, hi_p, act = close[WARM:] * np.exp(lo), close[WARM:] * np.exp(hi), close[WARM:] * np.exp(y[WARM:])
    sc = (hi_p - lo_p) + 20 * np.maximum(lo_p - act, 0) + 20 * np.maximum(act - hi_p, 0); scores[name] = sc
    ins = (act >= lo_p) & (act <= hi_p); h = len(ins) // 2
    rows.append({"method": name, **m, "kupiec_p": float(1 - chi2.cdf(lr, 1)), "cov_first_half": float(ins[:h].mean()), "cov_second_half": float(ins[h:].mean())})
base = scores["EWMA Gaussian, lambda 0.94 (current)"]
for r in rows:   # paired score difference vs the current method, Newey-West (lag 5) t-test
    d = scores[r["method"]] - base; dbar = d.mean(); lrv = d.var() + 2 * sum((1 - k / 6) * np.mean((d[k:] - dbar) * (d[:-k] - dbar)) for k in range(1, 6))
    r["score_diff_vs_ewma"] = float(dbar); r["score_diff_p"] = float(2 * (1 - norm.cdf(abs(dbar / np.sqrt(max(lrv, 1e-12) / len(d))))))
(Path(__file__).resolve().parent / "results" / "interval_comparison.json").write_text(json.dumps(rows, indent=1, default=float))
print(f"{'method':42s} cov    width$  score$   Kupiec p  1st/2nd half   score vs EWMA (p)")
for r in rows: print(f"{r['method']:42s} {r['coverage']:.1%}  {r['mean_width_usd']:6.0f}  {r['interval_score_usd']:6.0f}   {r['kupiec_p']:.3f}    {r['cov_first_half']:.0%}/{r['cov_second_half']:.0%}      {r['score_diff_vs_ewma']:+6.1f} ({r['score_diff_p']:.2f})")
