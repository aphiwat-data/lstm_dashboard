"""Tests for the E2d mechanism check (run: python test_mechanism.py). Synthetic data only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mechanism_check as mc  # noqa: E402


def _hourly(day_utc_start: str, rets: np.ndarray, drop: int | None = None) -> pd.DataFrame:
    """Consecutive hourly bars starting at day_utc_start 00:00 UTC; closes are exp(cumsum(rets))."""
    opens = pd.date_range(day_utc_start, periods=len(rets), freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rets))
    h = pd.DataFrame({"open_utc": opens, "close": close})
    return h.drop(index=drop).reset_index(drop=True) if drop is not None else h


def test_post_settlement_returns_handle_summer_and_winter_time_and_gaps():
    rng = np.random.default_rng(0)
    for day, shift in (("2024-07-10", 4), ("2024-01-10", 5)):               # EDT is UTC-4, EST is UTC-5
        r = rng.normal(0, 0.002, 24)
        h = _hourly(day, r)
        out = mc.post_settlement_returns(h)
        # the bar that closes at k o'clock ET opens at (k - 1 + shift) o'clock UTC, which is row k - 1 + shift
        want3 = sum(r[k - 1 + shift] for k in (15, 16, 17))
        want4 = want3 + r[14 - 1 + shift]
        d = pd.Timestamp(day)
        assert abs(out.loc[d, "r_post"] - want3) < 1e-12 and abs(out.loc[d, "r_post4"] - want4) < 1e-12, (day, out)
        hg = _hourly(day, r, drop=16 - 1 + shift)                           # the bar closing at 16:00 ET is missing
        og = mc.post_settlement_returns(hg)
        assert d not in og.index or np.isnan(og.loc[d, "r_post"]), "a missing bar must not be bridged"


def test_ols_hac_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 2))
    y = 1 + 2 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 1, 4000)
    r = mc.ols_hac(y, X)
    assert np.allclose(r["beta"], [1, 2, -3], atol=0.1) and min(abs(t) for t in r["t"]) > 20 and 0.90 < r["r2"] < 0.95
    z = mc.ols_hac(rng.normal(size=4000), X)
    assert max(abs(t) for t in z["t"]) < 4


def test_a_forecast_that_only_knows_r_post_has_no_information_about_r_rest():
    rng = np.random.default_rng(2)
    n = 3000
    rpost, rrest = rng.normal(0, 0.003, n), rng.normal(0, 0.009, n)
    y, b = rpost + rrest, np.zeros(n)
    f = 0.3 * rpost + rng.normal(0, 0.001, n)
    a = mc.analyse(y, rpost, f, b)
    assert a["pearson_r_post"] > 0.5 and abs(a["pearson_r_rest"]) < 0.06 and a["cw_p_target_y"] < 0.01 and a["cw_p_target_r_rest"] > 0.01
    assert a["ols_y_on_r_post_and_f"]["t"][2] < 3, "given r_post, the forecast adds nothing"
    assert abs(a["var_share_r_post"] - rpost.var() / y.var()) < 1e-12 and a["r2_f_vs_no_change_from_17h_spot"] < 0
    g = 0.3 * rrest + rng.normal(0, 0.001, n)                               # a forecast that knows the genuinely later part
    c = mc.analyse(y, rpost, g, b)
    assert c["cw_p_target_r_rest"] < 0.01 and c["ols_y_on_r_post_and_f"]["t"][2] > 3


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
        print("ok  ", name)
    print("all tests passed")
