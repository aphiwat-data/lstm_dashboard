"""Causality tests for build_daily_dataset. Run: python test_no_lookahead_daily.py (or pytest)."""
import numpy as np
import pandas as pd

from build_daily_dataset import HORIZONS, MACRO_TICKERS, align_macro, make_features, make_macro_features, make_targets


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


def _synthetic_macro(seed=3):
    # A different, sparser calendar than gold's (e.g. a bond-market holiday gold still trades on) is the whole
    # point: it's what actually exercises the as-of backward-join logic, not a calendar that happens to match.
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-12-01", periods=1400)[::2]  # every other business day: deliberately sparse/offset
    macro = pd.DataFrame(index=idx)
    for name in MACRO_TICKERS:
        level = 50 + np.cumsum(rng.normal(0, 0.3, len(idx)))
        macro[name] = np.abs(level) + 1.0  # keep strictly positive (some features take log())
    return macro


def test_macro_alignment_and_features_are_causal():
    d = _synthetic_days(n=1500, seed=4)
    macro = _synthetic_macro()
    cut_date = macro.index[300]  # corrupt macro rows from this date onward only
    aligned = align_macro(d.index, macro)
    base_mf = make_macro_features(aligned, d["close"])

    macro2 = macro.copy()
    corrupt_pos = macro.index.get_indexer([cut_date])[0]
    for name in MACRO_TICKERS:
        rng = np.random.default_rng(5)
        macro2.iloc[corrupt_pos:, macro2.columns.get_loc(name)] *= rng.uniform(0.3, 3.0, len(macro2) - corrupt_pos)
    aligned2 = align_macro(d.index, macro2)
    new_mf = make_macro_features(aligned2, d["close"])

    unaffected = d.index[d.index < cut_date]  # gold days whose as-of macro row is strictly before the corruption
    cols = [c for c in base_mf.columns if c.startswith("f_")]
    pd.testing.assert_frame_equal(base_mf.loc[unaffected, cols], new_mf.loc[unaffected, cols], check_exact=False, rtol=1e-12, atol=1e-12)
    affected = d.index[d.index >= macro.index[corrupt_pos + 20]]  # well past the corruption, rolling windows included
    assert not np.allclose(base_mf.loc[affected, cols].to_numpy(), new_mf.loc[affected, cols].to_numpy(), equal_nan=True)


def test_macro_features_are_scale_free():
    d = _synthetic_days(n=1500, seed=6)
    macro = _synthetic_macro(seed=7)
    aligned = align_macro(d.index, macro)
    a = make_macro_features(aligned, d["close"])
    scaled = macro.copy()
    for name in ("dxy", "silver", "spx", "tip"):  # ret_n features use log(), so must be scale-invariant; yield10y/vix use diff(), not scale-invariant by design (already stationary in their native units)
        scaled[name] = scaled[name] * 2.9
    b = make_macro_features(align_macro(d.index, scaled), d["close"] * 2.9)
    cols = [c for c in a.columns if any(c.startswith(f"f_{name}_ret") for name in ("dxy", "silver", "spx", "tip")) or c == "f_gold_silver_ratio_ret_20"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    test_features_and_targets_are_causal()
    test_features_are_scale_free()
    test_macro_alignment_and_features_are_causal()
    test_macro_features_are_scale_free()
    print("OK: no look-ahead, features are scale-free (price + macro)")
