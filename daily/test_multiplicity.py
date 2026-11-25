"""Tests for the E3 multiplicity experiment (run: python test_multiplicity.py; needs the arch package). Synthetic data only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multiplicity_experiment as me  # noqa: E402


def test_losses_on_tiny_arrays():
    y, yhat = np.array([0.01, -0.02]), np.array([0.00, 0.01])
    assert np.allclose(me.loss_squared_return(y, yhat), [1e-4, 9e-4])
    c, n = np.array([100.0, 100.0]), np.array([101.0, 99.0])
    assert np.allclose(me.loss_absolute_price(c, n, np.zeros(2)), [1.0, 1.0])
    assert np.allclose(me.loss_absolute_price(c, n, np.log(np.array([1.01, 0.99]))), [0.0, 0.0])
    assert list(me.loss_direction(np.array([1, 0, 1]), np.array([0.9, 0.9, 0.2]))) == [0.0, 1.0, 1.0]
    f, t = np.array([2.0, 2.0]), np.array([2.0, 4.0])
    assert np.allclose(me.qlike(t, f), np.log(f) + t / f)


def test_holm_matches_hand_values_and_is_monotone():
    adj = me.holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])                         # 3 * 0.01, then max(0.03, 2 * 0.03), max(0.06, 1 * 0.04)
    assert (me.holm(np.array([0.5, 0.6])) <= 1).all() and np.allclose(me.holm(np.array([0.2])), [0.2])


def test_hac_t_has_the_sign_of_the_mean():
    d = np.random.default_rng(0).normal(0.3, 1, 400)
    assert me.hac_t(d) > 3 and me.hac_t(-d) < -3


def _synthetic(n=500, k=12, better=None, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.chisquare(1, n)                                              # noisy squared-error-like losses
    cands = {f"m{i}": base + rng.normal(0, 0.15, n) for i in range(k)}              # the null: the same expected loss as the benchmark
    if better is not None:
        cands["good"] = base * better + rng.normal(0, 0.15, n)
    return base, pd.DataFrame(cands)


def test_spa_finds_a_truly_better_candidate_and_stays_quiet_on_noise():
    base, c = _synthetic(better=0.6, seed=1)
    r = me.family_tests(base, c, reps=600, with_sensitivity=False)
    assert r["spa_p"]["consistent"] < 0.05 and "good" in r["stepm_superior"], r["spa_p"]
    rejections = 0
    for s in range(8):                                                     # pure noise: at most a couple of false alarms in eight tries
        b, cc = _synthetic(seed=100 + s, k=15)
        rejections += me.family_tests(b, cc, reps=400, with_sensitivity=False)["spa_p"]["consistent"] < 0.05
    assert rejections <= 2, rejections


def test_mcs_drops_a_clearly_worse_model_and_keeps_equal_ones():
    rng = np.random.default_rng(3)
    n = 600
    base = rng.chisquare(1, n)
    c = pd.DataFrame({"same_a": base + rng.normal(0, 0.1, n), "same_b": base + rng.normal(0, 0.1, n), "bad": base * 2.5 + rng.normal(0, 0.1, n)})
    r = me.family_tests(base, c, reps=600, with_sensitivity=False)
    assert "bad" in r["mcs_excluded"] and "BENCHMARK" in r["mcs_included"] and "same_a" in r["mcs_included"], (r["mcs_included"], r["mcs_excluded"])


def test_family_output_contains_one_row_per_candidate_with_consistent_numbers():
    base, c = _synthetic(k=5, seed=7)
    r = me.family_tests(base, c, reps=300, with_sensitivity=True)
    assert len(r["candidates"]) == 5 and set(r["spa_consistent_p_by_block"]) == {"5", "10", "20"}
    for row in r["candidates"]:
        assert abs(row["mean_diff"] - (row["benchmark_mean_loss"] - row["mean_loss"])) < 1e-12 and row["holm_p"] >= row["p_one_sided"] - 1e-12
    assert r["expected_rejections_under_null"] == 0.25


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
