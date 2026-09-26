"""Causality tests for build_hourly_dataset. Run: python test_no_lookahead.py (or pytest)."""
import numpy as np
import pandas as pd

from build_hourly_dataset import AUX_TAGS, HORIZONS, make_features, make_targets


def _synthetic_bars(n=3000, seed=0, drop_frac=0.25):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    idx = idx[rng.random(n) > drop_frac]  # irregular gaps like closed-market hours
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.random(len(idx)) * 0.002)
    low = np.minimum(open_, close) * (1 - rng.random(len(idx)) * 0.002)
    vol = rng.lognormal(0, 0.5, len(idx)) + 0.01
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _corrupt_after(df, cut_pos, seed=1):
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in ("open", "high", "low", "close"):
        out.iloc[cut_pos + 1 :, out.columns.get_loc(col)] *= rng.uniform(0.5, 2.0, len(out) - cut_pos - 1)
    out.iloc[cut_pos + 1 :, out.columns.get_loc("volume")] *= rng.uniform(0.1, 10.0, len(out) - cut_pos - 1)
    return out


def _aux_bars(gold, seed):
    a = _synthetic_bars(len(gold) + 200, seed=seed, drop_frac=0.4)
    return a[a.index.isin(gold.index) | (np.random.default_rng(seed).random(len(a)) < 0.3)]


def test_features_and_targets_are_causal():
    gold = _synthetic_bars()
    aux = {name: _aux_bars(gold, seed=10 + i) for i, name in enumerate(AUX_TAGS)}
    cut = 1500

    base_f = make_features(gold, aux)
    base_y = make_targets(gold)

    gold2 = _corrupt_after(gold, cut)
    aux2 = {name: _corrupt_after(a, int(a.index.searchsorted(gold.index[cut], side="right")) - 1, seed=5) for name, a in aux.items()}
    new_f = make_features(gold2, aux2)
    new_y = make_targets(gold2)

    feat_cols = [c for c in base_f.columns if c.startswith("f_")]
    pd.testing.assert_frame_equal(
        base_f.iloc[: cut + 1][feat_cols], new_f.iloc[: cut + 1][feat_cols], check_exact=False, rtol=1e-12, atol=1e-12
    )
    # row i's target for horizon h reads bar i+h: unchanged while i+h <= cut, changed once it reaches a corrupted bar
    hmax = max(HORIZONS)
    pd.testing.assert_frame_equal(base_y.iloc[: cut - hmax], new_y.iloc[: cut - hmax])
    assert not np.isclose(base_y["y_ret_next"].iloc[cut], new_y["y_ret_next"].iloc[cut])
    for h in HORIZONS:
        assert np.isclose(base_y[f"y_ret_h{h}"].iloc[cut - h], new_y[f"y_ret_h{h}"].iloc[cut - h])
        assert not np.isclose(base_y[f"y_ret_h{h}"].iloc[cut - h + 1], new_y[f"y_ret_h{h}"].iloc[cut - h + 1])


def test_features_do_not_contain_target_information():
    gold = _synthetic_bars(seed=3)
    f = make_features(gold)
    y = make_targets(gold)["y_ret_next"]
    aligned = pd.concat([f.filter(like="f_"), y], axis=1).dropna()
    corr = aligned.corr()["y_ret_next"].drop("y_ret_next").abs()
    assert corr.max() < 0.1, corr.sort_values(ascending=False).head(3)


if __name__ == "__main__":
    test_features_and_targets_are_causal()
    test_features_do_not_contain_target_information()
    print("OK: no look-ahead detected")
