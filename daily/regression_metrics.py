"""Regression-first evaluation of the daily walk-forward forecasts, computed from wf_predictions.csv alone (no retraining).

A price forecast is judged by how far it is from the realised price and by whether it beats the "tomorrow = today" forecast, not by
calling up/down. The headline numbers are therefore:

  * point error in USD and in % of price: MAE, RMSE, MAPE;
  * the same relative to the no-change forecast: MAE skill (1 - MAE/MAE_naive), RMSE skill, Theil's U, Diebold-Mariano p-value;
  * return fit: R2 (sklearn definition, centred on the test mean), out-of-sample R2 against the no-change and the rolling-mean forecasts
    (Campbell & Thompson, 2008) and the Spearman IC;
  * uncertainty: coverage, width and interval score of a 90% band built only from earlier errors (no look-ahead).

Direction accuracy and AUC stay in walk_forward_results.json as a secondary, classification-style check.

Usage:  python regression_metrics.py [results_dir ...] [--refresh-latest-interval]
        (default: results and results/with_*; --refresh-latest-interval rewrites the shipped 90% band in latest_forecast.json from the saved errors)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hourly"))
from train_models import dm_pvalue, ic_pvalue, r2_returns  # noqa: E402  (same definitions as the published walk-forward results)

MODELS = ("drift", "ridge", "hist_gb_reg", "lstm")  # the models that output a return (hence price) forecast
LABELS = {"drift": "Rolling drift", "ridge": "Ridge", "hist_gb_reg": "HistGB regressor", "lstm": "LSTM (walk-forward)"}
Z90 = 1.6448536269514722  # two-sided 90% normal quantile
EWMA_LAMBDA = 0.94  # RiskMetrics' standard daily decay; fixed in advance, not tuned on the test window
WARMUP_DAYS = 60  # past errors required before the first band is scored
DM_LAGS = (0, 24)  # 24 is what the published results used (carried over from the hourly pipeline); 0 is the textbook one-step choice


# ----------------------------------------------------------------------------------------------------------------- point accuracy
def point_metrics(close: np.ndarray, next_close: np.ndarray, y_ret: np.ndarray, pred_ret: np.ndarray, drift_ret: np.ndarray) -> dict:
    """Metrics of one return forecast. Row i is a forecast made at the close of day i for day i+1; pred_price = close * exp(pred_ret)."""
    pred_price = close * np.exp(pred_ret)
    err, err0 = pred_price - next_close, close - next_close  # model and no-change price errors
    mae, mae0 = float(np.mean(np.abs(err))), float(np.mean(np.abs(err0)))
    rmse, rmse0 = float(np.sqrt(np.mean(err**2))), float(np.sqrt(np.mean(err0**2)))
    e_ret = y_ret - pred_ret
    varying = float(np.std(pred_ret)) > 0
    ic = spearmanr(pred_ret, y_ret) if varying else None
    sse = float(np.sum(e_ret**2))
    out = {
        "mae_usd": mae, "rmse_usd": rmse, "mape_pct": float(np.mean(np.abs(err) / next_close) * 100),
        "mae_skill_vs_naive": 1 - mae / mae0, "rmse_skill_vs_naive": 1 - rmse / rmse0, "theil_u": rmse / rmse0,
        "r2_returns": r2_returns(y_ret, pred_ret),
        "r2_os_vs_zero": 1 - sse / float(np.sum(y_ret**2)),
        "r2_os_vs_rolling_mean": 1 - sse / float(np.sum((y_ret - drift_ret) ** 2)),
        "ic_spearman": float(ic.statistic) if ic else float("nan"),
        "ic_p": ic_pvalue(ic.statistic, len(y_ret), 1) if ic else float("nan"),
    }
    for lag in DM_LAGS:  # Diebold-Mariano on squared return errors against the no-change forecast
        out[f"dm_p_vs_naive_lag{lag}"] = dm_pvalue(e_ret, y_ret, lag) if varying else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------------------- intervals
def ewma_sigma(past_errors: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """Root of the exponentially weighted mean of squared past errors (newest error weighs most)."""
    w = lam ** np.arange(len(past_errors))[::-1]
    return float(np.sqrt(np.sum(w * past_errors**2) / w.sum()))


def band_offsets(past_errors: np.ndarray, method: str, lam: float = EWMA_LAMBDA, window: int | None = None) -> tuple[float, float]:
    """(lower, upper) 90% offsets in log-return units around the point forecast, from errors observed before the forecast date only."""
    if method == "ewma_gaussian":
        s = ewma_sigma(past_errors, lam)
        return -Z90 * s, Z90 * s
    past = past_errors[-window:] if window else past_errors
    lo, hi = np.percentile(past, [5, 95])
    return float(lo), float(hi)


def rolling_bands(pred_ret: np.ndarray, errors: np.ndarray, method: str, warmup: int = WARMUP_DAYS, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper return bounds for forecast days warmup..n-1. Row i only sees errors[:i]: the error of row j is known at the close of
    day j+1, so for a forecast made at the close of day i the latest known error is row i-1."""
    lo, hi = zip(*(band_offsets(errors[:i], method, **kw) for i in range(warmup, len(errors))))
    return pred_ret[warmup:] + np.array(lo), pred_ret[warmup:] + np.array(hi)


def interval_metrics(close: np.ndarray, y_ret: np.ndarray, lo_ret: np.ndarray, hi_ret: np.ndarray, alpha: float = 0.10) -> dict:
    """Coverage, width and interval score (Gneiting & Raftery 2007; lower is better) of price bands, scored on the matching days."""
    c, y = close[-len(lo_ret):], y_ret[-len(lo_ret):]
    lo_p, hi_p, actual = c * np.exp(lo_ret), c * np.exp(hi_ret), c * np.exp(y)
    width = hi_p - lo_p
    score = width + (2 / alpha) * np.maximum(lo_p - actual, 0) + (2 / alpha) * np.maximum(actual - hi_p, 0)
    return {"n": int(len(y)), "coverage": float(np.mean((actual >= lo_p) & (actual <= hi_p))), "mean_width_usd": float(width.mean()),
            "mean_width_pct": float(np.mean(width / c) * 100), "interval_score_usd": float(score.mean())}


def latest_band(resid: np.ndarray, last_close: float, r_hat: float) -> list[float]:
    """The shipped 90% price band for tomorrow: Gaussian with sigma = EWMA(0.94) of every error observed so far."""
    lo, hi = band_offsets(resid, "ewma_gaussian")
    return [round(last_close * float(np.exp(r_hat + lo)), 2), round(last_close * float(np.exp(r_hat + hi)), 2)]


# ------------------------------------------------------------------------------------------------------------------------ driver
def compute(preds: pd.DataFrame) -> dict:
    close, nxt, y = preds["meta_close"].to_numpy(float), preds["meta_next_close"].to_numpy(float), preds["y_ret_next"].to_numpy(float)
    drift = preds["pred_ret_drift"].to_numpy(float)
    zero = np.zeros_like(y)
    naive = point_metrics(close, nxt, y, zero, drift)
    out: dict = {
        "protocol": {"n_test": len(y), "test_start": str(preds["date"].iloc[0])[:10], "test_end": str(preds["date"].iloc[-1])[:10],
                     "interval_warmup_days": WARMUP_DAYS, "ewma_lambda": EWMA_LAMBDA, "dm_lags": list(DM_LAGS),
                     "note": "row i = forecast made at the close of day i for day i+1; intervals use only errors known before the forecast"},
        "no_change": {k: naive[k] for k in ("mae_usd", "rmse_usd", "mape_pct", "r2_returns", "r2_os_vs_rolling_mean")},
        "models": {},
    }
    for name in MODELS:
        pred = preds[f"pred_ret_{name}"].to_numpy(float)
        m = point_metrics(close, nxt, y, pred, drift)
        p_up = preds[f"p_up_{name}"] if f"p_up_{name}" in preds else None
        m["says_up_share"] = float((p_up > 0.5).mean()) if p_up is not None and p_up.notna().all() else float((pred > 0).mean())
        out["models"][name] = m
    errors = y - preds["pred_ret_lstm"].to_numpy(float)
    pred = preds["pred_ret_lstm"].to_numpy(float)
    iv: dict = {"nominal": 0.90, "warmup_days": WARMUP_DAYS}
    for key, method, kw in (("lstm_ewma_gaussian", "ewma_gaussian", {}), ("lstm_empirical_expanding", "empirical", {})):
        iv[key] = interval_metrics(close, y, *rolling_bands(pred, errors, method, **kw))
    iv["no_change_ewma_gaussian"] = interval_metrics(close, y, *rolling_bands(zero, y, "ewma_gaussian"))
    iv["sensitivity"] = [
        {"method": f"empirical quantiles, last {w} errors" if w else "empirical quantiles, all past errors",
         **interval_metrics(close, y, *rolling_bands(pred, errors, "empirical", window=w))} for w in (60, 120, 250, None)
    ] + [{"method": f"Gaussian, EWMA lambda {lam}", **interval_metrics(close, y, *rolling_bands(pred, errors, "ewma_gaussian", lam=lam))} for lam in (0.94, 0.97)]
    out["interval"] = iv
    return out


def write_metrics(results_dir: Path, refresh_latest_interval: bool = False) -> Path:
    results_dir = Path(results_dir)
    preds = pd.read_csv(results_dir / "wf_predictions.csv")
    out = compute(preds)
    path = results_dir / "regression_metrics.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    if refresh_latest_interval:
        wf = json.loads((results_dir / "walk_forward_results.json").read_text())
        lf = wf["latest_forecast"]
        errors = preds["y_ret_next"].to_numpy(float) - preds["pred_ret_lstm"].to_numpy(float)
        lf["interval_90_close"] = latest_band(errors, lf["latest_close"], lf["predicted_return"])
        lf["interval_method"] = f"Gaussian, sigma = EWMA(lambda {EWMA_LAMBDA}) of all past test errors, z = {Z90:.3f}"
        (results_dir / "walk_forward_results.json").write_text(json.dumps(wf, indent=2, default=float))
        (results_dir / "latest_forecast.json").write_text(json.dumps(lf, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--refresh-latest-interval", action="store_true")
    args = ap.parse_args()
    root = Path("results")
    dirs = args.dirs or [root] + sorted(p for p in root.glob("with_*") if (p / "wf_predictions.csv").exists())
    for d in dirs:
        path = write_metrics(d, args.refresh_latest_interval and d == dirs[0])
        m = json.loads(path.read_text())
        lstm, iv = m["models"]["lstm"], m["interval"]["lstm_ewma_gaussian"]
        print(f"{d}: LSTM MAE ${lstm['mae_usd']:.2f} RMSE ${lstm['rmse_usd']:.2f} MAPE {lstm['mape_pct']:.3f}% | R2_OS vs rolling mean {lstm['r2_os_vs_rolling_mean']:+.4f} | "
              f"90% band coverage {iv['coverage']:.1%} (n={iv['n']}) -> {path}")


if __name__ == "__main__":
    main()
