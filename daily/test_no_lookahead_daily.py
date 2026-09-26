"""Causality tests for build_daily_dataset. Run: python test_no_lookahead_daily.py (or pytest)."""
import numpy as np
import pandas as pd

from build_daily_dataset import HORIZONS, make_features, make_targets


def _synthetic_days(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 1500 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.random(n) * 0.006)
    low = np.minimum(open_, close) * (1 - rng.random(n) * 0.006)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": rng.integers(1, 1000, n)}, index=idx)


def _corrupt_after(df, cut_pos, seed=1):
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in ("open", "high", "low", "close"):
        out.iloc[cut_pos + 1 :, out.columns.get_loc(col)] *= rng.uniform(0.5, 2.0, len(out) - cut_pos - 1)
    return out


def test_features_and_targets_are_causal():
    d = _synthetic_days()
    cut = 900
    base_f, base_y = make_features(d), make_targets(d)
    d2 = _corrupt_after(d, cut)
    new_f, new_y = make_features(d2), make_targets(d2)
    cols = [c for c in base_f.columns if c.startswith("f_")]
    pd.testing.assert_frame_equal(base_f.iloc[: cut + 1][cols], new_f.iloc[: cut + 1][cols], check_exact=False, rtol=1e-12, atol=1e-12)
    hmax = max(HORIZONS)
    pd.testing.assert_frame_equal(base_y.iloc[: cut - hmax], new_y.iloc[: cut - hmax])
    assert not np.isclose(base_y["y_ret_next"].iloc[cut], new_y["y_ret_next"].iloc[cut])
    for h in HORIZONS:
        assert np.isclose(base_y[f"y_ret_h{h}"].iloc[cut - h], new_y[f"y_ret_h{h}"].iloc[cut - h])
        assert not np.isclose(base_y[f"y_ret_h{h}"].iloc[cut - h + 1], new_y[f"y_ret_h{h}"].iloc[cut - h + 1])


def test_features_are_scale_free():
    d = _synthetic_days(seed=2)
    a, b = make_features(d), make_features(d.assign(**{k: d[k] * 3.7 for k in ("open", "high", "low", "close")}))
    cols = [c for c in a.columns if c.startswith("f_")]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-9)  # multiplying prices must not change any feature


if __name__ == "__main__":
    test_features_and_targets_are_causal()
    test_features_are_scale_free()
    print("OK: no look-ahead, features are scale-free")
