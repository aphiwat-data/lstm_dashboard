"""E7 (README section 15.8): can distribution-free (conformal) 90% intervals match the shipped EWMA Gaussian band?

PROTOCOL (frozen on 2026-11-26, before the first run on any reported window; every choice comes from the literature or from the existing
pipeline, none was tuned on the data):

Intervals. For the next-day log-return y(t+1), centre c(t) and half-width h(t): [c - h, c + h], converted to a price interval close(t) * exp(.)
exactly as in Table VII. Target coverage 90% (alpha = 0.10). The first band is scored after 60 rows (WARMUP_DAYS), so the PRIMARY window is the
389 scored origins of the main walk-forward run (daily/results/wf_predictions.csv): the same rows as Table VII.

Centres. LSTM forecast (primary) and no-change (c = 0). The LONG window (2010-01-01 to the last date with a next-day return) uses the no-change
centre only, because LSTM forecasts exist only for the 449 test days; the recursions start at the first row of the price history and are scored
from 2010.

Scale. sigma(t) = EWMA(0.94) of the errors e = y - c known before the forecast (the shipped definition, regression_metrics.ewma_sigma).

Methods (per centre):
  Gaussian        c +- 1.645 sigma                       (the shipped band when the centre is the LSTM)
  SCP raw         split conformal on |e|                 half-width q, the finite-sample rank ceil((n+1)(1-alpha)) of the calibration scores
  SCP norm        split conformal on |e| / sigma         half-width q * sigma(t)
  ACI norm g=.005 adaptive conformal inference (Gibbs and Candes 2021) on |e| / sigma, alpha(t+1) = alpha(t) + g (alpha - miss(t)), alpha(1) = alpha
  ACI norm g=.02  the same with the faster step
Calibration set = the last 250 scores known at the origin, from rows with at least 10 errors behind sigma. ACI takes the empirical quantile at level
clip(1 - alpha(t), 0, 1) (alpha(t) itself is not clipped). The outcome of origin t is used from origin t+1 on.

Scoring (same as Table VII, price space): coverage with a Wilson 95% interval, Kupiec test, Christoffersen independence and conditional-coverage tests
(first-order Markov chain of the hits; this also covers experiment E8), mean width in dollars and in % of the price, interval score (Gneiting-Raftery),
a paired test of the interval score against the shipped band (Newey-West, 5 lags, as in interval_experiment.py), coverage in each half, the worst
60-day rolling coverage and, on the long window, coverage by year. Nine conformal variants are compared with the shipped band, so the Bonferroni level
for any "better" or "worse" claim is 0.05 / 9; "comparable" claims rest on coverage, its interval and the Kupiec and Christoffersen tests.
All variants are reported; none is added after the results are seen.

Usage: python conformal_experiment.py [--dev] [--prices FILE] [--out-dir DIR]   (--dev: debug run on 2006-2007, no output)"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from regression_metrics import WARMUP_DAYS, Z90, interval_metrics, rolling_bands  # noqa: E402
from volatility_har_experiment import load_prices  # noqa: E402

ALPHA, LAM, WINDOW, WARM, J0 = 0.10, 0.94, 250, WARMUP_DAYS, 10
LONG_START, DEV = "2010-01-01", ("2006-01-01", "2007-12-31")
METHODS = {"Gaussian": ("gauss", None), "SCP raw": ("scp_raw", None), "SCP norm": ("scp_norm", None), "ACI norm g=0.005": ("aci_norm", 0.005), "ACI norm g=0.02": ("aci_norm", 0.02)}


# ------------------------------------------------------------------------------------------------------------------------ methods
def sigma_path(e: np.ndarray, lam: float = LAM) -> np.ndarray:
    """sig[i] = EWMA sigma of the errors known before origin i (e[:i]), the same weights as regression_metrics.ewma_sigma; NaN for i = 0."""
    sig = np.full(len(e), np.nan)
    num = den = 0.0
    for i in range(1, len(e)):
        num = lam * num + e[i - 1] ** 2
        den = lam * den + 1.0
        sig[i] = np.sqrt(num / den)
    return sig


def conformal_rank(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score, capped at the largest."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


def make_bands(y: np.ndarray, c: np.ndarray, sig: np.ndarray, kind: str, gamma: float | None = None, win: int = WINDOW, warm: int = WARM, alpha: float = ALPHA, j0: int = J0) -> tuple[np.ndarray, np.ndarray]:
    """Return-space bounds (lo, hi) for every origin; NaN before `warm`. Row i uses only y[:i], c[:i+1] and sig[:i+1]."""
    n = len(y)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    if kind == "gauss":
        lo[warm:], hi[warm:] = c[warm:] - Z90 * sig[warm:], c[warm:] + Z90 * sig[warm:]
        return lo, hi
    e = y - c
    scale = np.ones(n) if kind.endswith("raw") else sig
    s = np.abs(e) / scale
    a_t = alpha
    for i in range(warm, n):
        sc = s[np.arange(max(j0, i - win), i)]
        q = conformal_rank(sc, alpha) if kind.startswith("scp") else float(np.quantile(sc, min(max(1 - a_t, 0.0), 1.0)))
        lo[i], hi[i] = c[i] - q * scale[i], c[i] + q * scale[i]
        if kind.startswith("aci"):
            a_t += gamma * (alpha - float(abs(e[i]) > q * scale[i]))
    return lo, hi


# ------------------------------------------------------------------------------------------------------------------------ scoring
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(mid - half), float(mid + half)


def kupiec_lr(hits: np.ndarray, alpha: float = ALPHA) -> float:
    n, k = len(hits), int(hits.sum())
    p = k / n
    ll0 = xlogy(k, 1 - alpha) + xlogy(n - k, alpha)
    ll1 = xlogy(k, p) + xlogy(n - k, 1 - p)
    return float(-2 * (ll0 - ll1))


def christoffersen(hits: np.ndarray, alpha: float = ALPHA) -> dict:
    """hits = True when the outcome is inside the interval. Independence of the miss sequence and conditional coverage (Christoffersen 1998)."""
    miss = (~hits).astype(int)
    a, b = miss[:-1], miss[1:]
    n00, n01, n10, n11 = (int(((a == i) & (b == j)).sum()) for i in (0, 1) for j in (0, 1))
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01, pi11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr_ind = float(max(-2 * (ll0 - ll1), 0.0))
    lr_cc = kupiec_lr(hits, alpha) + lr_ind
    return {"ind_p": float(1 - chi2.cdf(lr_ind, 1)), "cc_p": float(1 - chi2.cdf(lr_cc, 2))}


def hac_p(d: np.ndarray, lag: int = 5) -> float:
    db = d.mean()
    lrv = d.var() + 2 * sum((1 - j / (lag + 1)) * np.mean((d[j:] - db) * (d[:-j] - db)) for j in range(1, lag + 1))
    return float(2 * (1 - norm.cdf(abs(db / np.sqrt(max(lrv, 1e-18) / len(d))))))


def evaluate(close: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA) -> tuple[dict, np.ndarray, np.ndarray]:
    """Scores of one band on the origins where it exists; returns the metrics, the per-origin interval scores and the inside flags."""
    lo_p, hi_p, act = close * np.exp(lo), close * np.exp(hi), close * np.exp(y)
    inside = (act >= lo_p) & (act <= hi_p)
    width = hi_p - lo_p
    isc = width + (2 / alpha) * np.maximum(lo_p - act, 0) + (2 / alpha) * np.maximum(act - hi_p, 0)
    n, k = len(inside), int(inside.sum())
    w = wilson(k, n)
    roll = pd.Series(inside.astype(float)).rolling(60).mean()
    h = n // 2
    m = {"n": n, "coverage": k / n, "coverage_ci95": [w[0], w[1]], "mean_width_usd": float(width.mean()), "mean_width_pct": float(np.mean(width / close) * 100), "interval_score_usd": float(isc.mean()),
         "kupiec_p": float(1 - chi2.cdf(kupiec_lr(inside, alpha), 1)), **christoffersen(inside, alpha), "cov_first_half": float(inside[:h].mean()), "cov_second_half": float(inside[h:].mean()),
         "worst_60d_coverage": float(roll.min())}
    return m, isc, inside


def run_centre(close: np.ndarray, y: np.ndarray, c: np.ndarray, rows: np.ndarray) -> dict:
    """Every method for one centre; `rows` are the origins that are scored."""
    sig = sigma_path(y - c)
    out = {}
    for name, (kind, gamma) in METHODS.items():
        lo, hi = make_bands(y, c, sig, kind, gamma)
        out[name] = (lo, hi)
    res, base_isc = {}, None
    for name, (lo, hi) in out.items():
        m, isc, inside = evaluate(close[rows], y[rows], lo[rows], hi[rows])
        res[name] = {"metrics": m, "isc": isc, "inside": inside, "lo": lo[rows], "hi": hi[rows]}
    return res


def paired(res_a: dict, res_base: dict) -> dict:
    d = res_a["isc"] - res_base["isc"]
    return {"score_diff": float(d.mean()), "score_diff_p": hac_p(d)}


# ------------------------------------------------------------------------------------------------------------------------- the run
def reproduce_shipped(close: np.ndarray, y: np.ndarray, pred: np.ndarray, rows: np.ndarray) -> dict:
    """The Gaussian band with the LSTM centre must be the shipped band: same bounds as rolling_bands and the coverage and interval score of Table VII."""
    sig = sigma_path(y - pred)
    lo, hi = make_bands(y, pred, sig, "gauss")
    lo_ref, hi_ref = rolling_bands(pred, y - pred, "ewma_gaussian")
    assert np.allclose(lo[rows], lo_ref) and np.allclose(hi[rows], hi_ref), "the Gaussian band differs from the shipped one"
    m, _, _ = evaluate(close[rows], y[rows], lo[rows], hi[rows])
    ship = json.loads((HERE / "results" / "interval_comparison.json").read_text())[0]
    assert abs(m["coverage"] - ship["coverage"]) < 1e-12 and abs(m["interval_score_usd"] - ship["interval_score_usd"]) < 1e-6, "the shipped metrics are not reproduced"
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "paper_e7")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--check", action="store_true", help="only check that the shipped Gaussian band is reproduced; prints no conformal result")
    args = ap.parse_args()
    t0 = time.time()
    px = load_prices(args.prices)
    dates = pd.DatetimeIndex(px["date"])
    close_all = px["close"].to_numpy(float)
    r = np.r_[np.nan, np.diff(np.log(close_all))]
    y_all = np.r_[r[1:], np.nan]                                   # y[t] = return of day t+1, known at the close of day t+1
    if args.dev:
        rows = np.flatnonzero((dates >= DEV[0]) & (dates <= DEV[1]))
        res = run_centre(close_all, y_all, np.zeros(len(y_all)), rows)
        print("DEV (debug only):", {k: (round(v["metrics"]["coverage"], 3), round(v["metrics"]["mean_width_pct"], 2)) for k, v in res.items()})
        return
    # ------------------------------------------------------------ primary window: the 449 test origins of the main run
    p = pd.read_csv(HERE / "results" / "wf_predictions.csv", parse_dates=["date"])
    pos = {d: i for i, d in enumerate(dates)}
    idx = np.array([pos[d] for d in p["date"]])
    close, y, pred = p["meta_close"].to_numpy(float), p["y_ret_next"].to_numpy(float), p["pred_ret_lstm"].to_numpy(float)
    assert np.allclose(close_all[idx], close, atol=1e-6)
    rows = np.arange(WARM, len(p))
    ship = reproduce_shipped(close, y, pred, rows)
    if args.check:
        print(f"shipped band reproduced: coverage {ship['coverage']:.4f}, interval score {ship['interval_score_usd']:.2f}, n {ship['n']}")
        return
    out = {"protocol": {"alpha": ALPHA, "lambda": LAM, "window": WINDOW, "warmup": WARM, "min_scores_row": J0, "gammas": [0.005, 0.02], "long_start": LONG_START, "bonferroni_level": 0.05 / 9}, "primary": {}, "long": {}}
    prim = {"LSTM centre": run_centre(close, y, pred, rows), "No-change centre": run_centre(close, y, np.zeros(len(y)), rows)}
    base = prim["LSTM centre"]["Gaussian"]
    for centre, res in prim.items():
        out["primary"][centre] = {name: {**v["metrics"], **paired(v, base)} for name, v in res.items()}
    # ------------------------------------------------------------ long window: no-change centre, 2010 onwards
    keep = np.flatnonzero((dates >= LONG_START) & np.isfinite(y_all))
    longres = run_centre(close_all, y_all, np.zeros(len(y_all)), keep)
    lbase = longres["Gaussian"]
    years = dates[keep].year
    out["long"] = {"n": int(len(keep)), "first": str(dates[keep[0]].date()), "last": str(dates[keep[-1]].date()),
                   "methods": {name: {**v["metrics"], **paired(v, lbase)} for name, v in longres.items()},
                   "by_year": [{"year": int(yr), "n": int((years == yr).sum()), **{name: float(v["inside"][years == yr].mean()) for name, v in longres.items()}} for yr in sorted(set(years))]}
    out["runtime_seconds"] = round(time.time() - t0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "e7_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    cols = {"date": p["date"].to_numpy()[rows], "close": close[rows], "next_close": p["meta_next_close"].to_numpy(float)[rows]}
    for centre, res in prim.items():
        for name, v in res.items():
            tag = f"{centre.split()[0].lower()}_{name.replace(' ', '_').replace('=', '').replace('.', '')}"
            cols[f"{tag}_lo"], cols[f"{tag}_hi"] = close[rows] * np.exp(v["lo"]), close[rows] * np.exp(v["hi"])
    pd.DataFrame(cols).to_csv(args.out_dir / "e7_primary_intervals.csv", index=False)
    print(f"written {args.out_dir}  ({out['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
