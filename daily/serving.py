"""Serving core shared by the model export and the daily Lambda: latest price-only features, an LSTM forward pass in plain numpy
(no TensorFlow at inference time) and the 90% band. Only numpy and pandas are needed, so it runs in a small Lambda zip.

The numpy forward pass reproduces Keras: LSTM(units) with gate order i, f, c, o -> Dense(32, relu) -> heads ret (linear) and dir (sigmoid);
dropout is inactive at inference. test_serving.py checks it against Keras on real windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_daily_dataset import clean_and_report, make_features

EWMA_LAMBDA = 0.94  # same band as regression_metrics.latest_band (a test keeps the two identical)
Z90 = 1.6448536269514722
PRICE_ROWS = 200  # >= 60 (longest feature lookback) + 20 (window) + margin


def latest_feature_rows(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: columns date, open, high, low, close, volume (any length >= PRICE_ROWS). Returns feature rows (f_*) plus meta_close, newest last."""
    d, _ = clean_and_report(prices)
    f = make_features(d)
    feats = [c for c in f.columns if c.startswith("f_")]
    return f.dropna(subset=feats).join(d["close"].rename("meta_close"))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lstm_last_hidden(x: np.ndarray, kernel: np.ndarray, recurrent: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x: (N, T, F) -> last hidden state (N, units), Keras LSTM defaults."""
    n, steps, _ = x.shape
    units = recurrent.shape[0]
    h, c = np.zeros((n, units)), np.zeros((n, units))
    for t in range(steps):
        z = x[:, t] @ kernel + h @ recurrent + bias
        i, f, g, o = (z[:, k * units:(k + 1) * units] for k in range(4))
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
    return h


def predict_windows(model: dict, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """windows: standardised (N, T, F). Returns (predicted log-return, p_up) averaged over seeds, like walk_forward.py."""
    rets, ups = [], []
    for s in range(int(model["n_seeds"])):
        g = lambda k: model[f"s{s}_{k}"]
        h = lstm_last_hidden(windows, g("lstm_k"), g("lstm_r"), g("lstm_b"))
        h = np.maximum(h @ g("d1_w") + g("d1_b"), 0.0)
        rets.append((h @ g("ret_w") + g("ret_b")).ravel() * float(model["ret_sd"]))
        ups.append(_sigmoid(h @ g("dir_w") + g("dir_b")).ravel())
    return np.mean(rets, axis=0), np.mean(ups, axis=0)


def standardise(model: dict, feature_rows: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num((feature_rows - model["mu"]) / model["sd"]), -10, 10)


def ewma_sigma(errors: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    w = lam ** np.arange(len(errors))[::-1]
    return float(np.sqrt(np.sum(w * errors**2) / w.sum()))


def band(errors: np.ndarray, last_close: float, r_hat: float) -> list[float]:
    s = ewma_sigma(errors)
    return [round(last_close * float(np.exp(r_hat - Z90 * s)), 2), round(last_close * float(np.exp(r_hat + Z90 * s)), 2)]


def forecast(model: dict, feature_rows: pd.DataFrame, errors: np.ndarray) -> dict:
    """Tomorrow's forecast from the newest feature row; `errors` are the realised log-return errors known so far (oldest first)."""
    window = int(model["window"])
    feats = [str(x) for x in model["feats"]]
    if len(feature_rows) < window:
        raise ValueError(f"need {window} feature rows, got {len(feature_rows)}")
    tail = feature_rows.tail(window)
    z = standardise(model, tail[feats].to_numpy(float))[None]
    r_hat, p_up = (float(a[0]) for a in predict_windows(model, z))
    last_close = float(tail["meta_close"].iloc[-1])
    return {"as_of_date": str(tail.index[-1].date()), "latest_close": round(last_close, 2), "predicted_return": r_hat,
            "predicted_next_close": round(last_close * float(np.exp(r_hat)), 2), "p_up": p_up, "interval_90_close": band(errors, last_close, r_hat)}
