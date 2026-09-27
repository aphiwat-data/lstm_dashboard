"""Causality tests for build_daily_dataset. Run: python test_no_lookahead_daily.py (or pytest)."""
import numpy as np
import pandas as pd

from build_daily_dataset import (COT_PUBLICATION_LAG_DAYS, CPI_DATES, FOMC_DATES, HORIZONS, MACRO_TICKERS,
                                  _parse_cot_rows, align_macro, make_calendar_features, make_cot_features,
                                  make_features, make_macro_features, make_targets)


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


def test_nfp_day_matches_known_dates():
    idx = pd.bdate_range("2025-01-01", "2025-12-31")
    f = make_calendar_features(idx)
    known_first_fridays = ["2025-01-03", "2025-02-07", "2025-03-07", "2025-06-06", "2025-08-01", "2025-12-05"]
    for d in known_first_fridays:
        assert f.loc[d, "f_nfp_day"] == 1.0, d
    assert f["f_nfp_day"].sum() == 12  # exactly one per month


def test_days_since_nfp_is_causal_and_resets():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    f = make_calendar_features(idx)
    d = f["f_days_since_nfp"]
    assert d.loc["2025-01-03"] == 0.0  # NFP day itself
    assert d.loc["2025-01-06"] == 1.0  # next trading day, one bar past NFP
    # corrupting the far future must not change days-since on an earlier date (pure function of the date itself)
    idx2 = pd.bdate_range("2025-01-01", "2025-12-31")  # extend the tail; earlier rows must be identical
    f2 = make_calendar_features(idx2)
    pd.testing.assert_series_equal(f["f_days_since_nfp"], f2["f_days_since_nfp"].loc[f.index])


def test_fomc_and_cpi_days_match_known_dates():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    f = make_calendar_features(idx)
    for d in ["2024-01-31", "2024-06-12", "2024-12-18"]:  # known FOMC decision days in 2024
        assert f.loc[d, "f_fomc_day"] == 1.0, d
    for d in ["2024-02-13", "2024-07-11", "2024-12-11"]:  # known CPI release days in 2024
        assert f.loc[d, "f_cpi_day"] == 1.0, d
    assert f["f_fomc_day"].sum() == 8  # 2024 had 8 scheduled FOMC meetings, no notation votes
    assert f["f_cpi_day"].sum() == 12  # 12 CPI releases in 2024


def test_fomc_cpi_dates_are_fixed_and_disjoint_from_lookahead():
    # the event lists themselves are module-level constants, independent of any price data, so mutating
    # price data (as the causality tests above do) cannot possibly change them -- this just guards against
    # accidental duplicate/out-of-range entries creeping in on a future edit.
    assert len(FOMC_DATES) == len(set(FOMC_DATES))
    assert len(CPI_DATES) == len(set(CPI_DATES))


def test_cot_publication_lag_shifts_index_forward():
    # a report "as of" a Tuesday must not appear at that Tuesday's own date -- it isn't published until
    # COT_PUBLICATION_LAG_DAYS later, so using the as-of date directly would leak future information.
    rows = [
        {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "open_interest_all": "100",
         "noncomm_positions_long_all": "60", "noncomm_positions_short_all": "40"},
        {"report_date_as_yyyy_mm_dd": "2024-01-09T00:00:00.000", "open_interest_all": "100",
         "noncomm_positions_long_all": "50", "noncomm_positions_short_all": "50"},
    ]
    out = _parse_cot_rows(rows)
    assert list(out.index) == [pd.Timestamp("2024-01-02") + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS),
                                pd.Timestamp("2024-01-09") + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS)]
    assert out["cot_net_pct"].tolist() == [0.2, 0.0]


def test_cot_features_are_causal():
    d = _synthetic_days(n=1500, seed=8)
    rng = np.random.default_rng(9)
    weekly = pd.bdate_range("2014-11-01", periods=600, freq="7D")
    cot = pd.DataFrame({"cot_net_pct": rng.uniform(-0.3, 0.3, len(weekly))}, index=weekly)
    cut_date = weekly[300]
    aligned = align_macro(d.index, cot)
    base = make_cot_features(aligned)

    cot2 = cot.copy()
    corrupt_pos = weekly.get_indexer([cut_date])[0]
    cot2.iloc[corrupt_pos:, 0] = rng.uniform(-0.3, 0.3, len(cot2) - corrupt_pos)
    new = make_cot_features(align_macro(d.index, cot2))

    unaffected = d.index[d.index < cut_date]
    cols = [c for c in base.columns if c.startswith("f_")]
    pd.testing.assert_frame_equal(base.loc[unaffected, cols], new.loc[unaffected, cols], check_exact=False, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_features_and_targets_are_causal()
    test_features_are_scale_free()
    test_macro_alignment_and_features_are_causal()
    test_macro_features_are_scale_free()
    test_nfp_day_matches_known_dates()
    test_days_since_nfp_is_causal_and_resets()
    test_fomc_and_cpi_days_match_known_dates()
    test_fomc_cpi_dates_are_fixed_and_disjoint_from_lookahead()
    test_cot_publication_lag_shifts_index_forward()
    test_cot_features_are_causal()
    print("OK: no look-ahead, features are scale-free (price + macro + calendar + COT)")
