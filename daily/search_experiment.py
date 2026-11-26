"""E2 (README section 15.8): an exhaustive, honest search for a next-day gold forecaster with a positive out-of-sample R-squared.

PROTOCOL (frozen on 2026-11-26, before the first run on the real data; every choice below comes from the literature, from the existing
pipeline or from the structural hypotheses listed here, none was tuned on the data; the synthetic tests in test_search.py are the only runs
made before the freeze):

Question. Is there ANY forecaster of the next-day gold return whose out-of-sample R-squared against the rolling-mean benchmark (the number
the dashboard shows) is positive, and does that survive (a) choosing the forecaster only from earlier data and (b) the number of forecasters tried?

Data. Daily GC=F closes 2004-06-01 onwards (the Silver table plus the live pipeline's newer bars, S3) and the six cross-asset series of the macro
feature set (DXY, silver, S&P 500, VIX, 10-year yield, TIP; Yahoo, 2004-06-01 onwards). Features are the pipeline's own causal, scale-free columns
(build_daily_dataset.py): price (16), + macro (18), + calendar (6). A row is an origin t, its label is log(close[t+h] / close[t]) for h trading days.
Rows after the last date on which every series exists are dropped, so no value is ever carried forward.

Feature sets: P price only (16); M price + macro (34); C price + calendar (22); A all (40); and five small sets that each state one hypothesis:
X = same-day returns of the series that close AFTER the 13:30 ET futures settlement (non-synchronous closes: f_ret_1, DXY, silver, S&P, 10Y, TIP);
T = trend (f_ret_20, f_ret_60, MA gaps 30 and 60); R = short-term reversal (f_ret_1, 3, 5); V = volatility state; K = calendar only (weekday, NFP, FOMC, CPI).

Forecasters (every one is refit every BLOCK = 21 origins on the labelled rows known at the block's first origin, last L rows, L in {365, 1000, expanding};
the target is scaled by the training standard deviation exactly as in walk_forward.py):
  ridge (alpha 100, 1000, 10000) and elastic net (alpha 0.02 and 0.1, l1 ratio 0.5) and one-component partial least squares on every feature set;
  gradient boosting (the project's setting and a strongly regularised one) on P, M, A, X; extra trees (200 trees, leaf >= 50) on P, M, A;
  LSTM / GRU (5 seeds averaged, the walk_forward.py recipe with the return and direction heads: dropout 0.2, Dense 32, Adam 1e-3, batch 64, at most 30
  epochs, early stopping on the last 15% of the training rows) in twelve configurations: window 20, units 32, LSTM on P / A with L = 365 and 1000, on X with
  L = 1000; GRU on A (L = 1000) and P (L = 365); LSTM windows 5, 10 and 60 on P; LSTM units 16 and 64 on P (L = 365).
Shrunk version of every forecaster: F = b + lam (f - b), b the benchmark, lam the slope of (y - b) on (f - b) through the origin over all earlier origins
whose labels are known, clipped to [0, 1], and 0 until 250 such origins exist (so a useless forecaster collapses to the benchmark).
Online selectors: at every block pick the forecaster (base or shrunk) with the largest squared-error gain over the benchmark in the last 250 labelled
origins (top 1), and the average of the best five (top 5). Both use only past losses; a forecaster joins the pool once it has a full 250-origin window
(the shrunk versions therefore join about a year after the base ones).
Benchmark b: the mean of the last 365 labelled returns (the project's rolling drift). Two other yardsticks are reported for every forecaster:
R-squared against no-change (zero return) and sklearn's centred R-squared.
Multi-horizon tracks: h = 5 and h = 20 with the linear families on P, M, T, A (ridge 1000 and 10000, PLS; L = 365, 1000, expanding); secondary.
Counts: h = 1 has 207 base forecasters (162 linear, 24 boosting, 9 extra trees, 12 recurrent), 414 with the shrunk versions, plus the two online selectors;
h = 5 and h = 20 have 36 base (72 with shrunk) each.

Windows. First origin 2010-01-04. DEV = origins 2011-01-03..2019-12-31 (the forecasters and rules may be chosen on it); TEST = origins 2020-01-01 to the last
labelled origin (nothing may be chosen on it); PROJECT = 2025-01-09..2026-10-23, the dashboard's 449-day window, inside TEST; FRESH = origins after
2026-10-23 (never used before, far too short to test; descriptive only).

Rules fixed in advance (the code applies them mechanically):
  PRIMARY tests (h = 1), each one-sided at the Bonferroni level 0.05 / 2 = 0.025, on TEST:
    P-A  the single forecaster (base or shrunk) with the highest DEV R-squared against b;
    P-B  the online selector, top 1.
  A result is called POSITIVE only if its TEST R-squared against b is > 0 AND the Clark-West test of the squared-error gain over b has one-sided p < 0.025;
  anything else is reported as it is (positive but not significant, or not positive). The PROJECT numbers are reported next to it, never instead of it.
  SECONDARY: the online top-5 average; the best forecaster of each family (ridge, enet, pls, hgb, et, nn) in its base and shrunk form chosen on DEV
  (12 tests, Holm); six pre-specified named hypotheses ridge(alpha 1000, expanding) on X, T, R, V, K, A (Holm over six).
  EXPLORATORY (flagged as selected on the evaluation data): the TEST leaderboard of all forecasters and Hansen's SPA, the reality check and the Romano-Wolf
  StepM over all of them (stationary bootstrap, block 10, 10,000 replications, seed 20261126), on the squared-error gain over b (as in E3) and on the
  Clark-West-adjusted gain 2 (y - b)(f - b), because the squared-error test is conservative when a forecaster nests the benchmark.
  Per forecaster and window: R-squared (three kinds), MAE in dollars against no-change, Diebold-Mariano p (Newey-West), Clark-West t and p, IC, and a
  stationary-bootstrap 95% interval for the R-squared against b for the frozen choices. Per calendar year for the frozen choices.
Everything is reported whatever it shows, the number of forecasters tried is stated next to every claim, and no forecaster, window, benchmark or rule is
added or changed after the results are seen.

AMENDMENT (2026-11-26, written before the first evaluation was run, after a two-replicate placebo test showed that the 365-day mean is a noisy benchmark that a
forecaster can beat merely by estimating the mean better; the forecasts do not depend on it and no primary rule, selection or window changes): the report adds, for every
forecaster and window, the R-squared and Clark-West p against the EXPANDING historical mean (all labelled returns since 2004-08, the Welch-Goyal benchmark) and the
p-value of the IC (the dashboard's own "predictive power" rule is R-squared against b > 0 and IC p < 0.05). A choice is called STRICT (genuine predictive power) only if
it is positive by the rule above AND its IC p < 0.05 AND its R-squared against the expanding mean is > 0. These are extra hurdles, not relaxations.

Usage:
  python search_experiment.py prepare                       # S3 prices + Yahoo macro -> data/e2_prices.parquet, data/e2_macro.parquet
  python search_experiment.py list                          # groups and candidate counts
  python search_experiment.py forecast --group NAME         # one group of forecasters -> results/paper_e2/forecasts/NAME.npz (resumable)
  python search_experiment.py evaluate                      # all groups -> results/paper_e2/e2_summary.json and leaderboards (needs: pip install arch)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hourly"))
from build_daily_dataset import align_macro, make_calendar_features, make_features, make_macro_features  # noqa: E402
from multiplicity_experiment import hac_t, holm  # noqa: E402

DATA = HERE / "data"
PRICES, MACRO = DATA / "e2_prices.parquet", DATA / "e2_macro.parquet"
OUT = HERE / "results" / "paper_e2"
BLOCK, BENCH_LEN, MIN_HIST, SELECT_WINDOW = 21, 365, 250, 250
FIRST_ORIGIN = "2010-01-04"
DEV = ("2011-01-03", "2019-12-31")
TEST_START = "2020-01-01"
PROJECT = ("2025-01-09", "2026-10-23")
FRESH_START = "2026-10-24"
LENGTHS = {"L365": 365, "L1000": 1000, "Lexp": None}
SEEDS, EPOCHS = 5, 30
REPS, BOOT_BLOCK, SEED = 10_000, 10, 20261126
PRIMARY_LEVEL = 0.05 / 2

PRICE = ["f_ret_1", "f_ret_3", "f_ret_5", "f_ret_10", "f_ret_20", "f_ret_60", "f_vol_5", "f_vol_20", "f_vol_60",
         "f_ma_gap_7", "f_ma_gap_14", "f_ma_gap_30", "f_ma_gap_60", "f_rsi_14", "f_dow_sin", "f_dow_cos"]
MACRO_COLS = [f"f_{n}_ret_{k}" for n in ("dxy", "silver", "spx") for k in (1, 5, 20)] + [f"f_yield10y_chg_{k}" for k in (1, 5, 20)] + \
    [f"f_tip_ret_{k}" for k in (1, 5, 20)] + ["f_vix_chg_5", "f_vix_z_60", "f_gold_silver_ratio_ret_20"]
CAL = ["f_nfp_day", "f_days_since_nfp", "f_fomc_day", "f_days_since_fomc", "f_cpi_day", "f_days_since_cpi"]
SETS = {"P": PRICE, "M": PRICE + MACRO_COLS, "C": PRICE + CAL, "A": PRICE + MACRO_COLS + CAL,
        "X": ["f_ret_1", "f_dxy_ret_1", "f_silver_ret_1", "f_spx_ret_1", "f_yield10y_chg_1", "f_tip_ret_1"],
        "T": ["f_ret_20", "f_ret_60", "f_ma_gap_30", "f_ma_gap_60"], "R": ["f_ret_1", "f_ret_3", "f_ret_5"],
        "V": ["f_vol_5", "f_vol_20", "f_vol_60"], "K": ["f_dow_sin", "f_dow_cos"] + CAL}
NN_CONFIGS = [("lstm", 20, 32, "P", 1000), ("lstm", 20, 32, "A", 1000), ("lstm", 60, 32, "P", 1000), ("gru", 20, 32, "A", 1000), ("lstm", 20, 32, "X", 1000),
              ("lstm", 20, 32, "P", 365), ("lstm", 20, 32, "A", 365), ("gru", 20, 32, "P", 365), ("lstm", 5, 32, "P", 365), ("lstm", 10, 32, "P", 365),
              ("lstm", 20, 16, "P", 365), ("lstm", 20, 64, "P", 365)]  # (cell, window, units, feature set, training length)
NAMED = [f"ridge[a1000]|{s}|Lexp" for s in "XTRVKA"]
FAMILIES = ["ridge", "enet", "pls", "hgb", "et", "nn"]


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    params: tuple
    fset: str
    length: str | None
    h: int = 1


def specs_linear(h: int) -> list[Spec]:
    out: list[Spec] = []
    sets = "PMCAXTRVK" if h == 1 else "PMTA"
    ridge_alphas, enet_alphas = ((100, 1000, 10000), (0.02, 0.1)) if h == 1 else ((1000, 10000), ())
    for s in sets:
        for L in LENGTHS:
            out += [Spec(f"ridge[a{a}]|{s}|{L}", "ridge", (a,), s, L, h) for a in ridge_alphas]
            out += [Spec(f"enet[a{a}]|{s}|{L}", "enet", (a,), s, L, h) for a in enet_alphas]
            out.append(Spec(f"pls[1]|{s}|{L}", "pls", (1,), s, L, h))
    return out


def specs_trees() -> dict[str, list[Spec]]:
    hgb = [Spec(f"hgb[{p}]|{s}|{L}", "hgb", (p,), s, L) for p in ("project", "strong") for s in "PMAX" for L in LENGTHS]
    et = [Spec(f"et[200]|{s}|{L}", "et", (200,), s, L) for s in "PMA" for L in LENGTHS]
    return {"hgb_h1": hgb, "et_h1": et}


def nn_name(cfg: tuple) -> str:
    cell, W, U, s, L = cfg
    return f"nn[{cell}_w{W}_u{U}]|{s}|L{L}"


def all_groups() -> dict[str, list[Spec]]:
    g: dict[str, list[Spec]] = {"linear_h1": specs_linear(1), "linear_h5": specs_linear(5), "linear_h20": specs_linear(20), **specs_trees()}
    for cfg in NN_CONFIGS:
        g[nn_name(cfg).replace("|", "_").replace("[", "_").replace("]", "")] = [Spec(nn_name(cfg), "nn", cfg, cfg[3], f"L{cfg[4]}")]
    return g


# --------------------------------------------------------------------------------------------------------------------------- data
def prepare() -> None:
    from build_daily_dataset import fetch_macro
    from volatility_har_experiment import load_prices

    DATA.mkdir(exist_ok=True)
    p = load_prices(None)
    p.to_parquet(PRICES)
    m = fetch_macro("2004-06-01", str((pd.Timestamp.now() + pd.Timedelta(days=2)).date()), None)
    m.to_parquet(MACRO)
    print(f"prices {p['date'].min().date()}..{p['date'].max().date()} ({len(p)} rows); macro {m.index.min().date()}..{m.index.max().date()} ({len(m)} rows)")


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    X: pd.DataFrame
    close: np.ndarray
    y: dict
    cl: dict
    first: int


def build_panel(prices: pd.DataFrame, macro: pd.DataFrame, horizons=(1, 5, 20), first_origin: str = FIRST_ORIGIN) -> Panel:
    p = prices[["date", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    d = p.drop_duplicates("date").sort_values("date").set_index("date")
    d = d[d["close"] > 0]
    feats = make_features(d).join(make_macro_features(align_macro(d.index, macro), d["close"])).join(make_calendar_features(d.index))
    cap = min(macro[c].last_valid_index() for c in macro.columns)  # never carry a stale macro value forward
    feats = feats[feats.index <= cap]
    cols = sorted(set().union(*SETS.values()))
    missing = set(cols) - set(feats.columns)
    assert not missing, f"missing feature columns: {missing}"
    feats = feats[cols].dropna()
    c = d["close"]
    y = {h: np.log(c.shift(-h) / c).reindex(feats.index).to_numpy() for h in horizons}
    cl = {h: c.shift(-h).reindex(feats.index).to_numpy() for h in horizons}
    first = int(feats.index.searchsorted(pd.Timestamp(first_origin)))
    assert first >= 1100, f"only {first} rows before the first origin; the L = 1000 / window 60 fits need about 1100"
    return Panel(feats.index, feats, c.reindex(feats.index).to_numpy(), y, cl, first)


def load_panel() -> Panel:
    return build_panel(pd.read_parquet(PRICES), pd.read_parquet(MACRO))


# ---------------------------------------------------------------------------------------------------------------------- forecasters
def make_model(family: str, params: tuple):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if family == "ridge":
        est = Ridge(alpha=float(params[0]))
    elif family == "enet":
        est = ElasticNet(alpha=float(params[0]), l1_ratio=0.5, max_iter=5000)
    elif family == "pls":
        est = PLSRegression(n_components=1, scale=False)
    elif family == "hgb":
        kw = dict(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0) if params[0] == "project" else \
            dict(max_depth=2, learning_rate=0.03, max_iter=100, min_samples_leaf=100, l2_regularization=5.0)
        est = HistGradientBoostingRegressor(random_state=0, **kw)
    elif family == "et":
        est = ExtraTreesRegressor(n_estimators=int(params[0]), min_samples_leaf=50, max_features=0.5, random_state=0, n_jobs=1)
    else:
        raise ValueError(family)
    return make_pipeline(StandardScaler(), est)


def train_rows(s: int, h: int, length: str | None) -> np.ndarray:
    """Positions whose labels are known at origin s (label of row u is known at u + h <= s), last L of them."""
    hi = s - h
    n = LENGTHS[length]
    return np.arange(0 if n is None else max(0, hi - n + 1), hi + 1)


def forecast_block(spec: Spec, X: np.ndarray, y: np.ndarray, s: int, e: int) -> np.ndarray:
    tr = train_rows(s, spec.h, spec.length)
    sd = float(y[tr].std())
    m = make_model(spec.family, spec.params).fit(X[tr], y[tr] / sd)
    return np.asarray(m.predict(X[s:e])).ravel() * sd


def forecast_nn_block(spec: Spec, X: np.ndarray, y: np.ndarray, s: int, e: int) -> np.ndarray:
    from tensorflow import keras
    from train_models import build_rnn, windows

    cell, W, U = spec.params[0], spec.params[1], spec.params[2]
    tr, targets = train_rows(s, 1, spec.length), np.arange(s, e)
    sd = float(y[tr].std())
    mu, sdv = X[tr].mean(axis=0), X[tr].std(axis=0)
    sdv[sdv == 0] = 1.0
    Z = np.clip((X - mu) / sdv, -10, 10)
    lo, hi = tr.min() - W + 1, e
    assert lo >= 0
    Zw = windows(Z[lo:hi], W)
    pick = lambda pos: Zw[pos - (lo + W - 1)]  # noqa: E731
    ydir = (y > 0).astype(np.float32)
    preds = []
    for seed in range(SEEDS):
        keras.backend.clear_session()
        model = build_rnn(X.shape[1], seed, W, cell, U)
        model.fit(pick(tr), {"ret": (y[tr] / sd).astype(np.float32), "dir": ydir[tr]}, validation_split=0.15, epochs=EPOCHS, batch_size=64, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        preds.append(model.predict(pick(targets), batch_size=512, verbose=0)["ret"].ravel() * sd)
    return np.mean(preds, axis=0)


def run_group(group: str, panel: Panel, out_dir: Path, max_blocks: int | None = None, chunk: int | None = None) -> Path:
    """Fit one group block by block and write forecasts/GROUP.npz. With `chunk`, stop after that many new blocks (a checkpoint is written and the partial file is returned):
    TensorFlow leaks about 85 MB per block of recurrent fits, so the launcher calls this repeatedly in fresh short-lived processes."""
    specs = all_groups()[group]
    T = len(panel.dates)
    is_nn = group.startswith("nn_")
    F = np.full((T, len(specs)), np.nan, dtype=np.float32)
    starts = list(range(panel.first, T, BLOCK))[:max_blocks]
    fdir = out_dir / "forecasts"
    fdir.mkdir(parents=True, exist_ok=True)
    final, part = fdir / f"{group}.npz", fdir / f"{group}.partial.npz"
    done = 0
    if final.exists():
        print(f"{group}: already done", flush=True)
        return final
    if part.exists():
        z = np.load(part, allow_pickle=False)
        if int(z["T"]) == T and list(z["names"]) == [s.name for s in specs]:
            F, done = z["F"], int(z["done"])
            print(f"{group}: resuming after block {done}", flush=True)
    if is_nn:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)  # five workers on eight cores: one compute thread each avoids spinning contention
        tf.config.threading.set_inter_op_parallelism_threads(1)
    if not is_nn:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=4 if group == "hgb_h1" else 1)  # five workers share eight cores, so one BLAS thread each; the boosting group is the long pole at the end and may use four
    t0 = time.time()
    mats = {k: panel.X[SETS[k]].to_numpy(float) for k in {sp.fset for sp in specs}}
    stop = len(starts) if chunk is None else min(len(starts), done + chunk)
    for bi in range(done, stop):
        s = starts[bi]
        e = min(s + BLOCK, T)
        for k, sp in enumerate(specs):
            F[s:e, k] = (forecast_nn_block if is_nn else forecast_block)(sp, mats[sp.fset], panel.y[sp.h], s, e)
        if (bi + 1) % 10 == 0 or bi + 1 == stop:
            np.savez(part, F=F, done=bi + 1, T=T, names=np.array([s.name for s in specs]))
            print(f"{group}: block {bi + 1}/{len(starts)} ({panel.dates[s].date()}), {time.time() - t0:.0f}s", flush=True)
    if stop < len(starts):
        return part
    np.savez_compressed(final, F=F, names=np.array([s.name for s in specs]), dates=panel.dates.values.astype("datetime64[D]"), h=specs[0].h, T=T)
    part.unlink(missing_ok=True)
    return final


# ----------------------------------------------------------------------------------------------------------- causal post-processing
def block_starts(T: int, first: int) -> list[int]:
    return list(range(first, T, BLOCK))


def rolling_mean_benchmark(y: np.ndarray, h: int, starts: list[int], T: int) -> np.ndarray:
    """b[t] for the block of t: mean of the last BENCH_LEN returns whose labels are known at the block's first origin."""
    b = np.full(T, np.nan)
    for s in starts:
        hi = s - h
        b[s:min(s + BLOCK, T)] = y[max(0, hi - BENCH_LEN + 1):hi + 1].mean()
    return b


def expanding_mean_benchmark(y: np.ndarray, h: int, starts: list[int], T: int) -> np.ndarray:
    """Historical mean of Welch and Goyal: the mean of ALL returns whose labels are known at the block's first origin (constant within the block)."""
    b = np.full(T, np.nan)
    for s in starts:
        b[s:min(s + BLOCK, T)] = y[:s - h + 1].mean()
    return b


def shrink(F: np.ndarray, y: np.ndarray, b: np.ndarray, h: int, starts: list[int]) -> np.ndarray:
    """F_shrunk = b + lam (f - b); lam from the labelled origins before each block (>= MIN_HIST of them, else 0 and the row stays NaN until then)."""
    T, out = len(y), np.full(F.shape, np.nan)
    first = starts[0]
    for s in starts:
        hist = np.arange(first, s - h + 1)
        if len(hist) < MIN_HIST:
            continue
        dev, res = F[hist] - b[hist, None], (y[hist] - b[hist])[:, None]
        lam = np.clip((dev * res).sum(0) / np.maximum((dev**2).sum(0), 1e-300), 0.0, 1.0)
        e = min(s + BLOCK, T)
        out[s:e] = b[s:e, None] + lam[None, :] * (F[s:e] - b[s:e, None])
    return out


def online_select(F: np.ndarray, y: np.ndarray, b: np.ndarray, h: int, starts: list[int]) -> tuple[np.ndarray, np.ndarray, list]:
    """Top-1 and top-5 average of the candidates with the largest squared-error gain over b in the last SELECT_WINDOW labelled origins."""
    T = len(y)
    top1, top5, picks = np.full(T, np.nan), np.full(T, np.nan), []
    gain = (y - b)[:, None] ** 2 - (y[:, None] - F) ** 2
    for s in starts:
        lo, hi = s - h - SELECT_WINDOW + 1, s - h
        if lo < starts[0]:
            continue
        g = gain[lo:hi + 1]
        ok = np.isfinite(g).all(0)
        if ok.sum() < 5:
            continue
        score = np.where(ok, g.sum(0), -np.inf)
        order = np.argsort(-score)
        e = min(s + BLOCK, T)
        top1[s:e] = F[s:e, order[0]]
        top5[s:e] = b[s:e] + (F[s:e, order[:5]] - b[s:e, None]).mean(1)
        picks.append((s, int(order[0])))
    return top1, top5, picks


# ---------------------------------------------------------------------------------------------------------------------------- scoring
def window_mask(dates: pd.DatetimeIndex, lo: str | None, hi: str | None) -> np.ndarray:
    m = np.ones(len(dates), dtype=bool)
    if lo:
        m &= dates >= pd.Timestamp(lo)
    if hi:
        m &= dates <= pd.Timestamp(hi)
    return m


def ic_p_value(r: float, n: int, h: int) -> float:
    """Two-sided p of a correlation with the effective sample size n / h (the project's rule, hourly/train_models.py)."""
    if not np.isfinite(r):
        return float("nan")
    n_eff = max(n // h, 3)
    t = r * np.sqrt((n_eff - 2) / max(1 - r * r, 1e-12))
    return float(2 * (1 - student_t.cdf(abs(t), n_eff - 2)))


def score(y, f, b, close, cl, h: int, b2=None) -> dict:
    """All metrics of one forecast series on rows where y, f and b (and b2, the expanding-mean benchmark, when given) exist."""
    ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(b)
    if b2 is not None:
        ok &= np.isfinite(b2)
    y, f, b, close, cl = y[ok], f[ok], b[ok], close[ok], cl[ok]
    b2 = None if b2 is None else b2[ok]
    n = int(ok.sum())
    if n < 20:
        return {"n": n}
    sse, sse_b = float(np.sum((y - f) ** 2)), float(np.sum((y - b) ** 2))
    lag = max(5, 2 * h)
    cw = 2 * (y - b) * (f - b)
    dm = (y - b) ** 2 - (y - f) ** 2
    t_cw, t_dm = hac_t(cw, lag), hac_t(dm, lag)
    mae, mae0 = float(np.mean(np.abs(close * np.exp(f) - cl))), float(np.mean(np.abs(close - cl)))
    ic = float(spearmanr(f, y)[0]) if np.std(f) > 0 else float("nan")
    out = {"n": n, "r2_vs_b": 1 - sse / sse_b, "r2_vs_zero": 1 - sse / float(np.sum(y**2)), "r2_centred": 1 - sse / float(np.sum((y - y.mean()) ** 2)),
           "mae_usd": mae, "mae_nochange_usd": mae0, "mae_skill": 1 - mae / mae0 if mae0 > 0 else float("nan"), "ic": ic, "ic_p": ic_p_value(ic, n, h),
           "dm_t": t_dm, "dm_p_better": float(1 - norm.cdf(t_dm)), "cw_t": t_cw, "cw_p": float(1 - norm.cdf(t_cw))}
    if b2 is not None:
        out["r2_vs_expmean"] = 1 - sse / float(np.sum((y - b2) ** 2))
        out["cw_p_vs_expmean"] = float(1 - norm.cdf(hac_t(2 * (y - b2) * (f - b2), lag)))
    return out


def r2_ci(y, f, b, reps: int = 2000, seed: int = SEED) -> list[float]:
    from arch.bootstrap import StationaryBootstrap

    ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(b)
    lf, lb = ((y - f) ** 2)[ok], ((y - b) ** 2)[ok]
    bs = StationaryBootstrap(BOOT_BLOCK, lf, lb, seed=seed)
    r = bs.apply(lambda a, c: 1 - a.sum() / c.sum(), reps)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def spa_stepm(bench_loss: np.ndarray, cand_loss: pd.DataFrame, reps: int = REPS) -> dict:
    from arch.bootstrap import SPA, StepM
    from multiplicity_experiment import drop_degenerate

    cand_loss, dropped = drop_degenerate(bench_loss, cand_loss)
    if cand_loss.shape[1] == 0:
        return {"k": 0, "n": int(len(bench_loss)), "dropped_degenerate": len(dropped), "note": "every candidate equals the benchmark"}
    spa = SPA(bench_loss, cand_loss, block_size=BOOT_BLOCK, reps=reps, bootstrap="stationary", seed=SEED, studentize=True)
    spa.compute()
    rc = SPA(bench_loss, cand_loss, block_size=BOOT_BLOCK, reps=reps, bootstrap="stationary", seed=SEED, studentize=False)
    rc.compute()
    step = StepM(bench_loss, cand_loss, size=0.05, block_size=BOOT_BLOCK, reps=reps, bootstrap="stationary", seed=SEED)
    step.compute()
    return {"k": int(cand_loss.shape[1]), "n": int(len(bench_loss)), "dropped_degenerate": len(dropped), "spa_p": {k: float(v) for k, v in spa.pvalues.items()},
            "reality_check_p": float(rc.pvalues["upper"]), "stepm_superior": [str(m) for m in step.superior_models]}


def family_of(name: str) -> str:
    return name.split("[")[0].replace("~s", "")


# ------------------------------------------------------------------------------------------------------------------------- evaluation
def load_forecasts(out_dir: Path, h: int, T: int, dates: pd.DatetimeIndex) -> tuple[list[str], np.ndarray]:
    names, cols = [], []
    for f in sorted((out_dir / "forecasts").glob("*.npz")):
        if f.name.endswith(".partial.npz"):
            continue
        z = np.load(f, allow_pickle=False)
        if int(z["h"]) != h:
            continue
        assert int(z["T"]) == T and (z["dates"] == dates.values.astype("datetime64[D]")).all(), f"{f.name}: origins differ from the panel"
        names += [str(s) for s in z["names"]]
        cols.append(z["F"].astype(float))
    assert cols, f"no forecast files for h = {h}"
    return names, np.hstack(cols)


def evaluate_track(panel: Panel, h: int, out_dir: Path, reps: int = REPS) -> dict:
    T, dates, y, close, cl = len(panel.dates), panel.dates, panel.y[h], panel.close, panel.cl[h]
    starts = block_starts(T, panel.first)
    names, F = load_forecasts(out_dir, h, T, dates)
    b = rolling_mean_benchmark(y, h, starts, T)
    b2 = expanding_mean_benchmark(y, h, starts, T)
    Fs = shrink(F, y, b, h, starts)
    names_all, Fall = names + [n + "~s" for n in names], np.hstack([F, Fs])
    top1, top5, picks = online_select(Fall, y, b, h, starts)
    labelled = np.isfinite(y)
    dev = window_mask(dates, DEV[0], DEV[1]) & labelled
    first_all = np.flatnonzero(np.isfinite(Fall).all(1) & np.isfinite(top1))
    dev &= np.arange(T) >= first_all[0]  # one common DEV sample: every forecaster and the selector exist
    test = window_mask(dates, TEST_START, None) & labelled
    proj = window_mask(dates, PROJECT[0], PROJECT[1]) & labelled
    fresh = window_mask(dates, FRESH_START, None) & labelled
    W = {"dev": dev, "test": test, "project": proj, "fresh": fresh}

    def sc(f, w):
        m = W[w]
        return score(np.where(m, y, np.nan), np.where(m, f, np.nan), np.where(m, b, np.nan), close, cl, h, np.where(m, b2, np.nan))

    def card(f, ci=True):
        c = {w: sc(f, w) for w in W}
        if ci:
            for w in ("test", "project"):
                m = W[w]
                c[w]["r2_vs_b_ci95"] = r2_ci(y[m], f[m], b[m], min(reps, 2000))
        return c

    dev_r2 = np.array([score(np.where(dev, y, np.nan), np.where(dev, Fall[:, k], np.nan), np.where(dev, b, np.nan), close, cl, h)["r2_vs_b"] for k in range(Fall.shape[1])])
    best = int(np.nanargmax(dev_r2))
    fam_best = {}
    for fam in FAMILIES:
        for shr in (False, True):
            idx = [k for k, n in enumerate(names_all) if family_of(n) == fam and n.endswith("~s") == shr]
            if idx:
                k = idx[int(np.nanargmax(dev_r2[idx]))]
                fam_best[names_all[k]] = k
    named = {n: names_all.index(n) for n in NAMED if n in names_all} if h == 1 else {}
    res: dict = {"h": h, "n_base": len(names), "n_total_with_shrunk": len(names_all), "n_origins": int(T), "n_dev": int(dev.sum()), "n_test": int(test.sum()),
                 "n_project": int(proj.sum()), "n_fresh": int(fresh.sum()), "dev_start": str(dates[dev][0].date()), "test_start": str(dates[test][0].date()),
                 "test_end": str(dates[test][-1].date()), "benchmark_windows": {w: score(np.where(W[w], y, np.nan), np.where(W[w], b, np.nan), np.where(W[w], b, np.nan), close, cl, h) for w in W},
                 "reference_forecasters_vs_b": {nm_: {w: sc(f_, w) for w in W} for nm_, f_ in (("expanding mean", b2), ("zero (no change)", np.zeros(T)))}}
    prim = {"P-A dev-selected best": (names_all[best], Fall[:, best]), "P-B online selector (top 1)": ("online top 1", top1)}
    res["primary"] = {}
    for label, (nm, f) in prim.items():
        c = card(f)
        c["name"] = nm
        c["positive"] = bool(c["test"].get("r2_vs_b", -1) > 0 and c["test"].get("cw_p", 1) < PRIMARY_LEVEL)
        c["strict"] = {"protocol_positive": c["positive"], "ic_p_below_0.05": bool(c["test"].get("ic_p", 1) < 0.05), "r2_vs_expmean_above_0": bool(c["test"].get("r2_vs_expmean", -1) > 0)}
        c["strict_positive"] = bool(all(c["strict"].values()))
        res["primary"][label] = c
    res["secondary"] = {"online top-5 average": card(top5),
                        "family bests (chosen on DEV)": {n: {**card(Fall[:, k], ci=False), "dev_r2_vs_b": float(dev_r2[k])} for n, k in fam_best.items()}}
    if named:
        cw = {n: sc(Fall[:, k], "test") for n, k in named.items()}
        padj = holm(np.array([cw[n]["cw_p"] for n in cw]))
        res["secondary"]["named hypotheses (ridge 1000, expanding)"] = {n: {"test": cw[n], "project": sc(Fall[:, k], "project"), "dev": sc(Fall[:, k], "dev"), "cw_p_holm_test": float(p)}
                                                                         for (n, k), p in zip(named.items(), padj)}
    fam_p = np.array([v["test"]["cw_p"] for v in res["secondary"]["family bests (chosen on DEV)"].values()])
    for (n, v), p in zip(res["secondary"]["family bests (chosen on DEV)"].items(), holm(fam_p)):
        v["cw_p_holm_test"] = float(p)
    by_year = {}
    for lab, f in (("P-A", Fall[:, best]), ("P-B", top1)):
        by_year[lab] = {}
        for yr in sorted(set(dates[labelled].year)):
            m = (dates.year == yr) & labelled
            if yr >= 2011 and m.sum() > 20:
                sy = score(np.where(m, y, np.nan), np.where(m, f, np.nan), np.where(m, b, np.nan), close, cl, h)
                if "r2_vs_b" in sy:
                    by_year[lab][int(yr)] = round(sy["r2_vs_b"], 5)
    res["by_year_r2_vs_b"] = by_year
    # exploratory: the whole pool on TEST (selected on the evaluation data), with the multiplicity-corrected tests
    rows = []
    for k, n in enumerate(names_all):
        d_, t_, p_ = (score(np.where(W[w], y, np.nan), np.where(W[w], Fall[:, k], np.nan), np.where(W[w], b, np.nan), close, cl, h, np.where(W[w], b2, np.nan)) for w in ("dev", "test", "project"))
        rows.append({"name": n, "family": family_of(n), "shrunk": n.endswith("~s"), "dev_r2_vs_b": d_.get("r2_vs_b"), "test_r2_vs_b": t_.get("r2_vs_b"), "test_r2_vs_zero": t_.get("r2_vs_zero"),
                     "test_r2_centred": t_.get("r2_centred"), "test_cw_p": t_.get("cw_p"), "test_dm_p_better": t_.get("dm_p_better"), "test_mae_skill": t_.get("mae_skill"),
                     "test_ic": t_.get("ic"), "test_ic_p": t_.get("ic_p"), "test_r2_vs_expmean": t_.get("r2_vs_expmean"), "test_cw_p_vs_expmean": t_.get("cw_p_vs_expmean"),
                     "project_r2_vs_expmean": p_.get("r2_vs_expmean"), "project_r2_vs_b": p_.get("r2_vs_b"), "project_mae_skill": p_.get("mae_skill")})
    board = pd.DataFrame(rows).sort_values("test_r2_vs_b", ascending=False)
    board.to_csv(out_dir / f"e2_leaderboard_h{h}.csv", index=False)
    res["exploratory"] = {"n_positive_test_r2_vs_b": int((board["test_r2_vs_b"] > 0).sum()), "n_candidates": int(len(board)),
                          "n_cw_p_below_0.05": int((board["test_cw_p"] < 0.05).sum()), "expected_cw_p_below_0.05_under_null": float(0.05 * len(board)),
                          "top10_test": board.head(10).to_dict("records")}
    tm = test & np.isfinite(Fall).all(1)
    bench_loss = ((y - b) ** 2)[tm]
    cand_loss = pd.DataFrame({n: ((y - Fall[:, k]) ** 2)[tm] for k, n in enumerate(names_all)})
    res["exploratory"]["multiplicity_test_window"] = spa_stepm(bench_loss, cand_loss, reps)
    # the squared-error comparison is conservative when the candidate nests the benchmark (Clark and West 2007), so the same tests are repeated on the
    # Clark-West-adjusted gain 2 (y - b) (f - b) (benchmark loss 0, candidate loss minus that gain)
    cw_gain = pd.DataFrame({n: (2 * (y - b) * (Fall[:, k] - b))[tm] for k, n in enumerate(names_all)})
    res["exploratory"]["multiplicity_test_window_cw_adjusted"] = spa_stepm(np.zeros(int(tm.sum())), -cw_gain, reps)
    cols = {"date": dates[labelled], "y": y[labelled], "b": b[labelled], "b_expanding_mean": b2[labelled], "close": close[labelled], "next_close": cl[labelled], "dev_best": Fall[labelled, best], "online_top1": top1[labelled], "online_top5": top5[labelled]}
    pd.DataFrame(cols).to_csv(out_dir / f"e2_frozen_forecasts_h{h}.csv", index=False)
    res["dev_best_name"], res["online_picks_distinct"] = names_all[best], len({k for _, k in picks})
    return res


def evaluate(out_dir: Path, reps: int = REPS) -> None:
    panel = load_panel()
    summary = {"protocol": {"block": BLOCK, "bench_len": BENCH_LEN, "min_hist": MIN_HIST, "select_window": SELECT_WINDOW, "dev": DEV, "test_start": TEST_START, "project": PROJECT,
                            "primary_level": PRIMARY_LEVEL, "seeds_nn": SEEDS, "reps": reps, "bootstrap_block": BOOT_BLOCK, "seed": SEED},
               "data": {"first_row": str(panel.dates[0].date()), "last_row": str(panel.dates[-1].date()), "n_rows": len(panel.dates), "first_origin": str(panel.dates[panel.first].date())}, "tracks": {}}
    for h in (1, 5, 20):
        try:
            summary["tracks"][f"h{h}"] = evaluate_track(panel, h, out_dir, reps)
        except AssertionError as e:
            summary["tracks"][f"h{h}"] = {"error": str(e)}
        (out_dir / "e2_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    t = summary["tracks"].get("h1", {})
    for label, c in t.get("primary", {}).items():
        print(f"{label}: {c['name']} | TEST R2 vs b {c['test'].get('r2_vs_b', float('nan')):+.5f} CW p {c['test'].get('cw_p', float('nan')):.4f} | "
              f"PROJECT R2 vs b {c['project'].get('r2_vs_b', float('nan')):+.5f} | POSITIVE: {c['positive']} | IC p {c['test'].get('ic_p', float('nan')):.3f}, "
              f"R2 vs expanding mean {c['test'].get('r2_vs_expmean', float('nan')):+.5f} | STRICT: {c['strict_positive']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "list", "forecast", "evaluate"])
    ap.add_argument("--group")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None, help="stop after this many new blocks (resume by calling again)")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "list":
        for g, s in all_groups().items():
            print(f"{g:34s} {len(s):4d}")
        print("total base candidates:", sum(len(s) for s in all_groups().values()))
    elif a.cmd == "forecast":
        run_group(a.group, load_panel(), a.out_dir, a.max_blocks, a.chunk)
    else:
        evaluate(a.out_dir, a.reps)


if __name__ == "__main__":
    main()
