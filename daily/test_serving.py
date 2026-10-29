"""Tests for serving.py. Run: python test_serving.py (or pytest)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import serving
from regression_metrics import latest_band

HERE = Path(__file__).resolve().parent
ART = HERE / "serving_artifacts"


def _prices(n=900, seed=0):
    rng = np.random.default_rng(seed)
    close = 1500 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    return pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=n), "open": close, "high": close * 1.004, "low": close * 0.996, "close": close, "volume": 1e5})


def test_numpy_forward_matches_keras():
    model, ref = dict(np.load(ART / "serving_model.npz")), json.loads((ART / "serving_reference.json").read_text())
    r, p = serving.predict_windows({**model, "n_seeds": 1}, np.array(ref["windows"]))
    assert np.max(np.abs(r - np.array(ref["ret"]))) < 1e-6 and np.max(np.abs(p - np.array(ref["dir"]))) < 1e-6


def test_forward_is_not_a_constant():  # the parity test must be able to fail: different windows give different outputs
    model, ref = dict(np.load(ART / "serving_model.npz")), json.loads((ART / "serving_reference.json").read_text())
    r, _ = serving.predict_windows({**model, "n_seeds": 1}, np.array(ref["windows"]))
    assert r.std() > 0 and not np.allclose(r, r[0])
    shifted = np.array(ref["windows"]) + 1.0
    r2, _ = serving.predict_windows({**model, "n_seeds": 1}, shifted)
    assert not np.allclose(r, r2)


def test_band_equals_the_evaluation_band():
    errs = np.random.default_rng(1).normal(0, 0.012, 300)
    assert serving.band(errs, 4000.0, 0.002) == latest_band(errs, 4000.0, 0.002)


def test_features_from_a_short_tail_equal_features_from_full_history():
    prices = _prices()
    full = serving.latest_feature_rows(prices)
    short = serving.latest_feature_rows(prices.tail(serving.PRICE_ROWS))
    feats = [c for c in full.columns if c.startswith("f_")]
    pd.testing.assert_frame_equal(full[feats].tail(20), short[feats].tail(20), check_exact=False, rtol=1e-9)


def test_forecast_is_consistent():
    model = dict(np.load(ART / "serving_model.npz"))
    rows = serving.latest_feature_rows(_prices().tail(serving.PRICE_ROWS))
    f = serving.forecast(model, rows, model["errors"])
    lo, hi = f["interval_90_close"]
    assert lo < f["predicted_next_close"] < hi and 0 <= f["p_up"] <= 1 and f["as_of_date"] == str(rows.index[-1].date())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
