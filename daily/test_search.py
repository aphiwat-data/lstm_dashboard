"""Tests for the E2 search experiment (run: python test_search.py; the end-to-end tests need arch and scikit-learn). Synthetic data only."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search_experiment as se  # noqa: E402


def _market(n_days=3200, seed=0, ar=0.0):
    """Gold-like prices (business days from 2004-06-01) with an optional AR(1) return component, and six macro series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-06-01", periods=n_days)
    e = rng.normal(0, 0.01, n_days)
    r = np.zeros(n_days)
    for t in range(1, n_days):
        r[t] = ar * r[t - 1] + e[t]
    prices = pd.DataFrame({"date": dates, "close": 400 * np.exp(np.cumsum(r + 0.0002))})
    macro = pd.DataFrame({k: 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n_days))) for k in ("dxy", "silver", "spx", "tip")}, index=dates)
    macro["vix"] = 15 + 5 * np.abs(np.sin(np.arange(n_days) / 50)) + rng.normal(0, 0.3, n_days)
    macro["yield10y"] = 4 + np.cumsum(rng.normal(0, 0.01, n_days))
    return prices, macro[["dxy", "silver", "spx", "vix", "yield10y", "tip"]]


def _panel(**kw):
    prices, macro = _market(**kw)
    return se.build_panel(prices, macro, first_origin="2009-03-02")


def test_panel_never_carries_a_stale_macro_value_and_labels_are_right():
    prices, macro = _market(n_days=2600)
    cut = macro.iloc[:2300]
    p = se.build_panel(prices, cut, first_origin="2009-03-02")
    assert p.dates.max() <= cut.index[-1], "rows after the macro end must be dropped"
    c = prices.set_index("date")["close"]
    t = p.dates[10]
    assert abs(p.y[1][10] - np.log(c.shift(-1)[t] / c[t])) < 1e-12 and abs(p.y[5][10] - np.log(c.shift(-5)[t] / c[t])) < 1e-12
    assert np.isfinite(p.y[1]).all() and np.isfinite(p.y[20]).all(), "prices continue past the cap, so every row has its labels"
    p2 = se.build_panel(prices.iloc[:2300], cut, first_origin="2009-03-02")
    assert len(p2.dates) == len(p.dates) and np.isnan(p2.y[1][-1]) and np.isnan(p2.y[20][-20:]).all() and np.isfinite(p2.y[20][-21])
    assert p.first >= 1100 and set(se.SETS) == set("PMCAXTRVK")
    for s, cols in se.SETS.items():
        assert set(cols) <= set(p.X.columns), s


def test_forecasts_use_only_labels_known_at_the_block_start():
    p = _panel(n_days=2500)
    s = p.first + 21 * 3
    e = s + 21
    for h in (1, 5):
        X, y = p.X[se.SETS["P"]].to_numpy(float), p.y[h]
        for fam, par in (("ridge", (1000,)), ("enet", (0.02,)), ("pls", (1,)), ("hgb", ("project",)), ("et", (20,))):
            for L in ("L365", "Lexp"):
                spec = se.Spec("t", fam, par, "P", L, h)
                f0 = se.forecast_block(spec, X, y, s, e)
                X2, y2 = X.copy(), y.copy()
                rng = np.random.default_rng(1)
                X2[e:] = rng.normal(size=X2[e:].shape)                       # features after the block are unknown at s
                y2[s - h + 1:] = rng.normal(size=y2[s - h + 1:].shape)       # labels of rows u > s - h are unknown at s
                f1 = se.forecast_block(spec, X2, y2, s, e)
                assert np.allclose(f0, f1), (fam, L, h)
                if fam in ("ridge", "enet", "pls"):
                    y3 = y.copy()
                    y3[s - h] += 0.5                                          # the newest known label must matter
                    assert not np.allclose(f0, se.forecast_block(spec, X, y3, s, e)), (fam, L, h)


def test_training_rows():
    assert list(se.train_rows(1500, 1, "Lexp")) == list(range(0, 1500))
    assert list(se.train_rows(1500, 5, "Lexp")) == list(range(0, 1496))
    r = se.train_rows(1500, 1, "L365")
    assert len(r) == 365 and r[-1] == 1499 and r[0] == 1135
    r = se.train_rows(1500, 20, "L1000")
    assert len(r) == 1000 and r[-1] == 1480


def test_benchmark_is_the_mean_of_the_labelled_returns_and_causal():
    rng = np.random.default_rng(0)
    T, h, first = 2000, 5, 1200
    y = rng.normal(0, 0.01, T)
    starts = se.block_starts(T, first)
    b = se.rolling_mean_benchmark(y, h, starts, T)
    s = starts[7]
    assert np.isclose(b[s], y[s - h - 364:s - h + 1].mean()) and np.isclose(b[s + 20], b[s]) and np.isnan(b[first - 1])
    y2 = y.copy()
    y2[s - h + 1:] = 9.0
    assert np.isclose(se.rolling_mean_benchmark(y2, h, starts, T)[s], b[s])


def test_shrink_is_causal_collapses_noise_and_keeps_signal():
    rng = np.random.default_rng(1)
    T, h, first = 3000, 1, 1200
    sig = rng.normal(0, 0.004, T)
    y = sig + rng.normal(0, 0.01, T)
    starts = se.block_starts(T, first)
    b = se.rolling_mean_benchmark(y, h, starts, T)
    good, noise = sig + rng.normal(0, 0.002, T), rng.normal(0, 0.01, T)
    F = np.column_stack([good, noise, y.copy()])                             # column 2 is a (leaky) perfect forecast, only to test the clip
    S = se.shrink(F, y, b, h, starts)
    ok = np.arange(T) >= starts[13]
    assert np.isnan(S[starts[0]]).all() and np.isfinite(S[ok]).all()
    mse = lambda f: np.mean((y[ok] - f[ok]) ** 2)  # noqa: E731
    assert mse(S[:, 1]) < mse(F[:, 1]) and mse(S[:, 1]) <= np.mean((y[ok] - b[ok]) ** 2) * 1.01, "noise collapses to the benchmark"
    assert mse(S[:, 0]) < np.mean((y[ok] - b[ok]) ** 2), "a real signal survives"
    assert np.allclose(S[ok, 2], F[ok, 2], atol=1e-9), "a forecast equal to the outcome has slope 1 and is not shrunk"
    s = starts[20]
    F2, y2 = F.copy(), y.copy()
    F2[s:] = rng.normal(size=F2[s:].shape)
    y2[s - h + 1:] = rng.normal(size=y2[s - h + 1:].shape)
    Fm, ym = np.where(np.arange(T)[:, None] < s, F, F2), np.where(np.arange(T) <= s - h, y, y2)
    lam_real = (se.shrink(F, y, b, h, starts)[s] - b[s]) / (F[s] - b[s])
    lam_mod = (se.shrink(Fm, ym, b, h, starts)[s] - b[s]) / (Fm[s] - b[s])
    assert np.allclose(lam_real, lam_mod), "the shrink factor at s must not depend on forecasts at or after s or on labels not yet known"
    assert ((lam_real >= -1e-12) & (lam_real <= 1 + 1e-12)).all()


def test_online_selector_finds_the_planted_signal_and_is_causal():
    rng = np.random.default_rng(2)
    T, h, first = 3500, 1, 1200
    sig = rng.normal(0, 0.004, T)
    y = sig + rng.normal(0, 0.01, T)
    starts = se.block_starts(T, first)
    b = se.rolling_mean_benchmark(y, h, starts, T)
    F = np.column_stack([sig + rng.normal(0, 0.002, T)] + [rng.normal(0, 0.01, T) for _ in range(30)])
    top1, top5, picks = se.online_select(F, y, b, h, starts)
    late = [k for s, k in picks if s >= starts[40]]
    assert np.mean(np.array(late) == 0) > 0.8, "the informative candidate should be picked almost always"
    ok = np.isfinite(top1)
    assert ok.sum() > 1000 and np.mean((y[ok] - top1[ok]) ** 2) < np.mean((y[ok] - b[ok]) ** 2)
    s = starts[60]
    F2, y2 = F.copy(), y.copy()
    F2[s:] = rng.normal(size=F2[s:].shape)
    y2[s - h + 1:] = rng.normal(size=y2[s - h + 1:].shape)
    t1, t5, _ = se.online_select(np.where(np.arange(T)[:, None] < s, F, F2), np.where(np.arange(T) <= s - h, y, y2), b, h, starts)
    assert np.allclose(t1[s - 21:s], top1[s - 21:s], equal_nan=True) and np.isfinite(t1[s]) and np.isfinite(t5[s])
    # the pick at block s depends on F[s:...] only through the picked column, so check the selection itself on a modified future
    F3 = F.copy()
    F3[s:] = rng.normal(size=F3[s:].shape)
    _, _, p3 = se.online_select(F3, y, b, h, starts)
    assert dict(p3)[s] == dict(picks)[s], "the choice at s must not depend on forecasts made at or after s"


def test_expanding_mean_benchmark_is_causal_and_scores_report_it():
    rng = np.random.default_rng(4)
    T, h, first = 2000, 5, 1200
    y = rng.normal(0.0003, 0.01, T)
    starts = se.block_starts(T, first)
    b2 = se.expanding_mean_benchmark(y, h, starts, T)
    s = starts[9]
    assert np.isclose(b2[s], y[:s - h + 1].mean()) and np.isclose(b2[s + 20], b2[s]) and np.isnan(b2[first - 1])
    y2 = y.copy()
    y2[s - h + 1:] = 5.0
    assert np.isclose(se.expanding_mean_benchmark(y2, h, starts, T)[s], b2[s])
    b = se.rolling_mean_benchmark(y, h, starts, T)
    f = b2 + rng.normal(0, 0.002, T)
    close, cl = np.full(T, 100.0), 100 * np.exp(y)
    sc = se.score(y, f, b, close, cl, h, b2)
    ok = np.isfinite(b2)
    assert np.isclose(sc["r2_vs_expmean"], 1 - np.sum((y[ok] - f[ok]) ** 2) / np.sum((y[ok] - b2[ok]) ** 2)) and 0 <= sc["cw_p_vs_expmean"] <= 1
    assert 0 <= sc["ic_p"] <= 1 and se.ic_p_value(0.0, 400, 1) > 0.99 and se.ic_p_value(0.5, 400, 1) < 1e-10 and np.isnan(se.ic_p_value(float("nan"), 400, 1))
    assert "r2_vs_expmean" not in se.score(y, f, b, close, cl, h)


def test_clark_west_identity_scores_and_null_calibration():
    rng = np.random.default_rng(3)
    y, f, b = rng.normal(0, 0.01, 800), rng.normal(0, 0.01, 800), rng.normal(0, 0.001, 800)
    assert np.allclose(2 * (y - b) * (f - b), (y - b) ** 2 - (y - f) ** 2 + (b - f) ** 2)
    close, cl = np.full(800, 100.0), 100 * np.exp(y)
    s = se.score(y, f, b, close, cl, 1)
    assert np.isclose(s["r2_vs_b"], 1 - np.sum((y - f) ** 2) / np.sum((y - b) ** 2)) and np.isclose(s["r2_vs_zero"], 1 - np.sum((y - f) ** 2) / np.sum(y**2))
    z = se.score(y, b, b, close, cl, 1)
    assert abs(z["r2_vs_b"]) < 1e-12 and abs(z["cw_t"]) < 1e-9
    perfect = se.score(y, y, b, close, cl, 1)
    assert perfect["r2_vs_b"] == 1.0 and perfect["mae_usd"] < 1e-9 and perfect["mae_skill"] > 0.999
    draws = ((rng.normal(0, 0.01, 1500), rng.normal(0, 0.01, 1500)) for _ in range(400))
    rej = np.mean([se.score(y_, f_, np.zeros(1500), np.full(1500, 100.0), 100 * np.exp(y_), 1)["cw_p"] < 0.05 for y_, f_ in draws])
    assert 0.02 < rej < 0.09, f"Clark-West rejects {rej:.3f} of noise forecasts at a nominal 5%"


def _small_groups():
    sp = [se.Spec(f"ridge[a{a}]|{s}|{L}", "ridge", (a,), s, L) for a in (100, 1000) for s in "PR" for L in ("L365", "Lexp")]
    sp.append(se.Spec("pls[1]|R|Lexp", "pls", (1,), "R", "Lexp"))
    sp.append(se.Spec("ridge[a1000]|X|Lexp", "ridge", (1000,), "X", "Lexp"))
    return {"linear_h1": sp}


def test_chunked_runs_resume_and_equal_one_run():
    old = se.all_groups
    try:
        se.all_groups = _small_groups
        p = _panel(n_days=2500)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            se.run_group("linear_h1", p, Path(d1))
            calls = 0
            while not (Path(d2) / "forecasts" / "linear_h1.npz").exists():
                se.run_group("linear_h1", p, Path(d2), chunk=7)
                calls += 1
                assert calls < 40
            a = np.load(Path(d1) / "forecasts" / "linear_h1.npz")["F"]
            b = np.load(Path(d2) / "forecasts" / "linear_h1.npz")["F"]
            assert calls > 3 and np.array_equal(a, b, equal_nan=True), "chunked and one-shot runs must agree exactly"
            assert not (Path(d2) / "forecasts" / "linear_h1.partial.npz").exists()
    finally:
        se.all_groups = old


def _end_to_end(ar: float):
    old = (se.DEV, se.TEST_START, se.PROJECT, se.FRESH_START, se.all_groups)
    try:
        se.DEV, se.TEST_START, se.PROJECT, se.FRESH_START = ("2010-04-01", "2012-12-31"), "2013-01-01", ("2014-01-01", "2015-06-30"), "2015-07-01"
        se.all_groups = _small_groups
        p = _panel(n_days=3300, ar=ar)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            se.run_group("linear_h1", p, out)
            assert (out / "forecasts" / "linear_h1.npz").exists() and not (out / "forecasts" / "linear_h1.partial.npz").exists()
            return se.evaluate_track(p, 1, out, reps=400)
    finally:
        se.DEV, se.TEST_START, se.PROJECT, se.FRESH_START, se.all_groups = old


def test_end_to_end_finds_a_planted_autocorrelation_and_not_a_null():
    try:
        import arch  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print("  (skipped: needs arch and scikit-learn)")
        return
    real = _end_to_end(ar=0.15)
    assert real["primary"]["P-A dev-selected best"]["positive"] and real["primary"]["P-B online selector (top 1)"]["positive"], real["primary"]
    assert real["primary"]["P-A dev-selected best"]["test"]["r2_vs_b"] > 0.005
    assert real["primary"]["P-A dev-selected best"]["strict_positive"], real["primary"]["P-A dev-selected best"]["strict"]
    assert "expanding mean" in real["reference_forecasters_vs_b"] and "test_r2_vs_expmean" in real["exploratory"]["top10_test"][0]
    assert real["exploratory"]["multiplicity_test_window_cw_adjusted"]["spa_p"]["consistent"] < 0.01
    assert real["exploratory"]["multiplicity_test_window"]["spa_p"]["lower"] < 0.10, "the squared-error SPA is conservative for nested models but must lean the right way"
    null = _end_to_end(ar=0.0)
    for k, c in null["primary"].items():
        assert not c["positive"] and not c["strict_positive"], f"false positive under the null: {k} {c['test']}"
    assert null["exploratory"]["multiplicity_test_window"]["spa_p"]["consistent"] > 0.01
    assert null["exploratory"]["multiplicity_test_window_cw_adjusted"]["spa_p"]["consistent"] > 0.01


def test_placebo_is_unpredictable_but_keeps_the_volatility_clustering():
    import search_placebo as sp

    rng = np.random.default_rng(5)
    n = 3000
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = 0.97 * v[t - 1] + 0.15 * rng.normal()
    sig = 0.008 * np.exp(v)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.2 * r[t - 1] + sig[t] * rng.normal()                       # the source is autocorrelated and heteroskedastic
    prices = pd.DataFrame({"date": pd.bdate_range("2004-06-01", periods=n), "close": 400 * np.exp(np.cumsum(r))})
    pp = sp.placebo_prices(prices, np.random.default_rng(0))
    r0, r1 = np.diff(np.log(prices["close"])), np.diff(np.log(pp["close"]))
    ac = lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]  # noqa: E731
    assert len(pp) == n and (pp["date"] == prices["date"]).all() and pp["close"].iloc[0] == prices["close"].iloc[0]
    assert ac(r0) > 0.12 and abs(ac(r1)) < 0.06, "the placebo has no return autocorrelation"
    assert ac(np.abs(r1)) > 0.15 and abs(ac(np.abs(r1)) - ac(np.abs(r0))) < 0.2, "but it keeps the volatility clustering"
    assert np.allclose(sp.placebo_prices(prices, np.random.default_rng(0))["close"], pp["close"]) and not np.allclose(sp.placebo_prices(prices, np.random.default_rng(1))["close"], pp["close"])
    assert len(sp.SPECS) == 81 and len({c.name for c in sp.SPECS}) == 81


def test_every_candidate_name_is_unique_per_horizon_and_counts_match_the_protocol():
    g = se.all_groups()
    by_h: dict[int, list[str]] = {}
    for sp in g.values():
        for c in sp:
            by_h.setdefault(c.h, []).append(c.name)
    assert {h: len(v) for h, v in by_h.items()} == {1: 207, 5: 36, 20: 36}, {h: len(v) for h, v in by_h.items()}
    assert all(len(v) == len(set(v)) for v in by_h.values())
    assert len(g["linear_h1"]) == 162 and len(g["hgb_h1"]) == 24 and len(g["et_h1"]) == 9
    assert sum(1 for k in g if k.startswith("nn_")) == 12 and set(se.NAMED) <= set(by_h[1])
    assert se.family_of("ridge[a100]|P|L365~s") == "ridge" and se.family_of("nn[lstm_w20_u32]|P|L365") == "nn"


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
        print("ok  ", name)
    print("all tests passed")
