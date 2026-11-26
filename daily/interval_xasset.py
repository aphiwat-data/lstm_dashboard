"""90% interval methods around the cross-asset model's forecasts (the same comparison as interval_experiment.py, which is for the LSTM), on the same scored test days.

Methods: EWMA Gaussian (the shipped band: sigma = EWMA 0.94 of the model's own past errors), empirical quantiles of all past errors, GARCH(1,1) Gaussian and with filtered
historical simulation (parameters refit every 7 days on the last 1,000 returns known then), and the no-change centre with the EWMA band as a reference. Every band uses only
information known before its forecast. The LSTM-only methods (quantile network, VIX scale model) are not repeated here.
Usage: python interval_xasset.py   (needs results/wf_predictions.csv with pred_ret_xasset and data/e2_prices.parquet; writes results/interval_comparison_xasset.json)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from regression_metrics import Z90, interval_metrics, rolling_bands  # noqa: E402

WARM, EVERY, FIT_LEN = 60, 7, 1000


def filt(e, om, a, b):
    v = np.empty(len(e) + 1)
    v[0] = e.var()
    for t in range(len(e)):
        v[t + 1] = om + a * e[t] ** 2 + b * v[t]
    return v


def fit_garch(r):
    mu = r.mean()
    e = r - mu
    v0 = e.var()

    def unpack(q):
        rho, u = 1 / (1 + np.exp(-q[0])), 1 / (1 + np.exp(-q[1]))
        return v0 * (1 - rho) * np.exp(q[2]), rho * u, rho * (1 - u)

    def nll(q):
        om, a, b = unpack(q)
        v = filt(e, om, a, b)[:-1]
        return 0.5 * np.sum(np.log(v) + e**2 / v)

    best = min((minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400}) for x0 in ([2.5, -1.5, 0.0], [3.5, -2.0, 0.0])), key=lambda o: o.fun)
    return mu, unpack(best.x)


def compare(results: Path, prices_path: Path) -> list[dict]:
    p = pd.read_csv(results / "wf_predictions.csv", parse_dates=["date"])
    close, y, pred = p["meta_close"].to_numpy(float), p["y_ret_next"].to_numpy(float), p["pred_ret_xasset"].to_numpy(float)
    errors = y - pred
    px = pd.read_parquet(prices_path)
    px["date"] = pd.to_datetime(px["date"])
    c = px.drop_duplicates("date").sort_values("date").set_index("date")["close"]
    r_all = np.log(c.shift(-1) / c).to_numpy(float)                        # row t: the return from t to t + 1 (what the main run calls y_ret_next)
    idx = c.index
    start, end = int(idx.get_loc(p["date"].iloc[0])), int(idx.get_loc(p["date"].iloc[-1])) + 1
    n = end
    sig_g, z_fhs = np.zeros(end - start), np.zeros((end - start, 2))
    for s in range(start, end, EVERY):
        e_ = min(s + EVERY, end)
        lo = max(0, s - FIT_LEN)
        mu, (om, a, b) = fit_garch(r_all[lo:s])
        eps = r_all[lo:e_] - mu
        v = filt(eps, om, a, b)
        zfit = eps[:s - lo] / np.sqrt(v[:s - lo])
        q5, q95 = np.percentile(zfit, [5, 95])
        for g in range(s, e_):
            sig_g[g - start] = np.sqrt(v[g - lo])
            z_fhs[g - start] = (q5, q95)
    keep = idx[start:end].get_indexer(pd.DatetimeIndex(p["date"]))      # the main run lacks two trading days that the price calendar has
    assert (keep >= 0).all()
    sig_g, z_fhs = sig_g[keep], z_fhs[keep]
    sl = lambda x: x[WARM:]  # noqa: E731
    bands = {"EWMA Gaussian, lambda 0.94 (current)": rolling_bands(pred, errors, "ewma_gaussian"),
             "Empirical quantiles, all past errors": rolling_bands(pred, errors, "empirical"),
             "GARCH(1,1), Gaussian": (sl(pred - Z90 * sig_g), sl(pred + Z90 * sig_g)),
             "GARCH(1,1), filtered historical sim.": (sl(pred + z_fhs[:, 0] * sig_g), sl(pred + z_fhs[:, 1] * sig_g)),
             "No-change centre, EWMA (reference)": rolling_bands(np.zeros(len(y)), y, "ewma_gaussian")}
    rows, scores = [], {}
    for name, (lo, hi) in bands.items():
        m = interval_metrics(close, y, lo, hi)
        cov, k = m["coverage"], round(m["coverage"] * m["n"])
        lr = -2 * (np.log(0.9**k * 0.1 ** (m["n"] - k)) - np.log(max(cov, 1e-9) ** k * max(1 - cov, 1e-9) ** (m["n"] - k)))
        lo_p, hi_p, act = close[WARM:] * np.exp(lo), close[WARM:] * np.exp(hi), close[WARM:] * np.exp(y[WARM:])
        scores[name] = (hi_p - lo_p) + 20 * np.maximum(lo_p - act, 0) + 20 * np.maximum(act - hi_p, 0)
        ins = (act >= lo_p) & (act <= hi_p)
        h = len(ins) // 2
        rows.append({"method": name, **m, "kupiec_p": float(1 - chi2.cdf(lr, 1)), "cov_first_half": float(ins[:h].mean()), "cov_second_half": float(ins[h:].mean())})
    base = scores["EWMA Gaussian, lambda 0.94 (current)"]
    for r in rows:
        d = scores[r["method"]] - base
        dbar = d.mean()
        lrv = d.var() + 2 * sum((1 - k / 6) * np.mean((d[k:] - dbar) * (d[:-k] - dbar)) for k in range(1, 6))
        r["score_diff_vs_ewma"] = float(dbar)
        r["score_diff_p"] = float(2 * (1 - norm.cdf(abs(dbar / np.sqrt(max(lrv, 1e-12) / len(d))))))
    return rows


def main() -> None:
    rows = compare(HERE / "results", HERE / "data" / "e2_prices.parquet")
    (HERE / "results" / "interval_comparison_xasset.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"{'method':40s} cov    width$  score$   Kupiec p  1st/2nd half   score vs EWMA (p)")
    for r in rows:
        print(f"{r['method']:40s} {r['coverage']:.1%}  {r['mean_width_usd']:6.0f}  {r['interval_score_usd']:6.0f}   {r['kupiec_p']:.3f}    {r['cov_first_half']:.0%}/{r['cov_second_half']:.0%}      {r['score_diff_vs_ewma']:+6.1f} ({r['score_diff_p']:.2f})")


if __name__ == "__main__":
    main()
