"""E6 (README section 15.8): is risk predictable when price is not? Realised-volatility forecasts against the shipped EWMA.

PROTOCOL (frozen on 2026-11-24, before the first run on any reported window; every choice below comes from the literature or from the
existing pipeline, none was tuned on the data):

Target. RV(d) = sum of squared hourly log-returns of the XAU/USD spot BID series (Dukascopy, hourly/data/raw) over the window that ends
at the futures settlement of day d. The window boundary is the one of {13:30 ET, 17:00 ET} whose summed hourly returns correlate best with
the daily GC=F log-return on the DEVELOPMENT window 2006-01-01..2007-12-31 (a measurement check on returns, not a forecasting result).
Days with fewer than 12 hourly bars have no RV. Forecasts are made at the close of day t for day t+1 and use only data up to day t.

Models (all forecast the next-day variance):
  EWMA 0.94 (daily GC=F returns, the shipped band's volatility) ......... level-matched to RV (multiplier from the training window)
  GARCH(1,1) Gaussian (daily returns, refit every block) ................. level-matched to RV
  EWMA 0.94 of RV (RiskMetrics applied to realised variance)
  HAR-RV levels  : RV(t+1) = c + b_d RV(t) + b_w mean RV(t-4..t) + b_m mean RV(t-21..t), OLS, forecast floored at 10% of the training mean
  HAR-RV logs    : the same regression for log RV, forecast exp(fit + residual variance / 2)
  HAR-RV-X       : HAR logs + dummies for the target day: NFP (first Friday), CPI release, FOMC decision, and a FOMC dummy for day t
                   (the FOMC announcement at 14:00 ET can fall on either side of the settlement boundary, so both days are included)
  LSTM-RV        : LSTM(32) -> Dropout(0.2) -> Dense(32, ReLU) -> linear, 20-day window of the HAR-RV-X inputs, target log RV, Adam 1e-3,
                   batch 64, at most 30 epochs, early stopping (patience 3, last 15% of the training rows), 3 seeds averaged
                   (same recipe as the return model in walk_forward.py)
Every fit uses the last 1000 trading days that were known at the block's first origin and is repeated every 7 origins on the primary
window (every 21 on the long window); LSTM-RV runs on the primary window only.

Windows. PRIMARY = the origins of the main walk-forward run (daily/results/wf_predictions.csv, 2025-01-09 onwards) that have RV for the
next day (hourly data end 2026-09-25); LONG = 2010-01-01 onwards with the same end. One common sample per window: rows where every model has
a forecast. Nothing is reported on the DEV window.

Scoring. Primary target RV(t+1): QLIKE = ln f + RV / f (mean; lower is better; Patton 2011), difference to EWMA 0.94 with a Diebold-Mariano
test (Bartlett HAC, 5 lags). Secondary target: the squared futures return of day t+1 (the product's own target; same as Table VIII).
Also reported: MSE of the variance, Mincer-Zarnowitz R-squared and slope, and the mean forecast over the mean target (bias).
Family of comparisons against EWMA 0.94: six models (GARCH, EWMA-RV, HAR levels, HAR logs, HAR-X, LSTM), so the Bonferroni level is 0.05 / 6.
Also pre-specified: HAR-X against HAR logs (do event days help?) and LSTM against HAR-X (does the LSTM help?).

Usage: python volatility_har_experiment.py [--dev] [--skip-lstm] [--prices FILE] [--raw-dir DIR] [--out-dir DIR]
Writes results/paper_e6/e6_summary.json and e6_primary_forecasts.csv (unless --dev)."""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm, ttest_ind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_daily_dataset import CPI_DATES, FOMC_DATES  # noqa: E402

LAM, FIT_LEN, EVERY_PRIMARY, EVERY_LONG, MIN_BARS = 0.94, 1000, 7, 21, 12
DEV = ("2006-01-01", "2007-12-31")
LONG_START = "2010-01-01"
BOUNDARIES = {"13:30 ET": 13 * 60 + 30, "17:00 ET": 17 * 60}
WINDOW, UNITS, EPOCHS, SEEDS = 20, 32, 30, 3
S3_BUCKET, S3_REGION = "gold-lstm-forecast", "ap-southeast-2"
MODELS = ["EWMA 0.94 (matched)", "GARCH(1,1) (matched)", "EWMA 0.94 of RV", "HAR-RV levels", "HAR-RV logs", "HAR-RV-X", "LSTM-RV"]
FAMILY = MODELS[1:]


# --------------------------------------------------------------------------------------------------------------------------- data
def load_prices(src: str | None) -> pd.DataFrame:
    """Daily GC=F closes: a local file with date and close, or the Silver table plus the live pipeline's newer bars from S3."""
    if src:
        p = Path(src)
        d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        d["date"] = pd.to_datetime(d["date"])
        return d[["date", "close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    import boto3

    s3 = boto3.client("s3", region_name=S3_REGION)
    get = lambda key: s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()  # noqa: E731
    silver = pd.read_parquet(io.BytesIO(get("silver/xauusd_daily_clean.parquet")), columns=["date", "close"])
    silver["date"] = pd.to_datetime(silver["date"])
    extra = pd.read_csv(io.BytesIO(get("bronze/streaming/xauusd_daily_incremental.csv")), parse_dates=["date"])[["date", "close"]]
    return pd.concat([silver, extra[extra["date"] > silver["date"].max()]]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def load_hourly_bid(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("xauusd-h1-????-??.csv"))
    frames = [pd.read_csv(f, usecols=["timestamp", "close"]) for f in files if f.stat().st_size > 0]
    h = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    h = h[h["close"] > 0].reset_index(drop=True)
    h["open_utc"] = pd.to_datetime(h["timestamp"], unit="ms", utc=True)
    return h[["open_utc", "close"]]


def day_positions(open_utc: pd.Series, daily_dates: pd.DatetimeIndex, boundary_minutes: int) -> np.ndarray:
    """Position in the daily index of the futures day each hourly bar belongs to: the bar is assigned to the first settlement at or after its close."""
    close_et = (open_utc + pd.Timedelta(hours=1)).dt.tz_convert("America/New_York")
    minutes = close_et.dt.hour * 60 + close_et.dt.minute
    label = close_et.dt.normalize().dt.tz_localize(None) + pd.to_timedelta((minutes > boundary_minutes).astype(int), unit="D")
    return np.searchsorted(daily_dates.to_numpy(), label.to_numpy())


def realised_variance(h: pd.DataFrame, daily_dates: pd.DatetimeIndex, boundary_minutes: int, min_bars: int = MIN_BARS) -> pd.DataFrame:
    """Per futures day: rv (sum of squared hourly log-returns), s (their plain sum) and n (bars); NaN where fewer than min_bars bars."""
    lr = np.log(h["close"]).diff()
    pos = day_positions(h["open_utc"], daily_dates, boundary_minutes)
    ok = lr.notna().to_numpy() & (pos < len(daily_dates))
    d = pd.DataFrame({"pos": pos[ok], "r": lr.to_numpy()[ok]})
    d["r2"] = d["r"] ** 2
    g = d.groupby("pos").agg(rv=("r2", "sum"), s=("r", "sum"), n=("r2", "size"))
    out = pd.DataFrame(np.nan, index=np.arange(len(daily_dates)), columns=["rv", "s", "n"])
    out.loc[g.index, ["rv", "s", "n"]] = g[["rv", "s", "n"]].to_numpy()
    bad = ~(out["n"] >= min_bars)
    out.loc[bad, ["rv", "s"]] = np.nan
    return out


def event_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Scheduled US releases on each futures day (known in advance): NFP = first Friday of the month, CPI and FOMC from the pipeline's date lists."""
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame({"nfp": ((d.dayofweek == 4) & (d.day <= 7)).astype(float), "cpi": d.isin(pd.to_datetime(CPI_DATES)).astype(float),
                         "fomc": d.isin(pd.to_datetime(FOMC_DATES)).astype(float)}, index=d)


# ------------------------------------------------------------------------------------------------------------------ volatility models
def ewma_path(x: np.ndarray, lam: float, init: float, square: bool) -> np.ndarray:
    """out[t] = variance forecast for day t+1 made with x up to day t (x are returns when square=True, realised variances otherwise)."""
    out, v = np.empty(len(x)), init
    for i, xi in enumerate(x):
        if np.isfinite(xi):
            v = lam * v + (1 - lam) * (xi * xi if square else xi)
        out[i] = v
    return out


def garch_variance(e: np.ndarray, om: float, a: float, b: float) -> np.ndarray:
    """v[k] = variance forecast for position k given positions before k (v[0] = sample variance), via the linear recursion v[k+1] = om + a e[k]^2 + b v[k]."""
    v0 = float(e.var())
    y = lfilter([1.0], [1.0, -b], om + a * e**2, zi=[b * v0])[0]
    return np.r_[v0, y]


def fit_garch(x: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    mu = float(x.mean()); e = x - mu; v0 = float(e.var())

    def unpack(q):
        rho = 1 / (1 + np.exp(-q[0])); u = 1 / (1 + np.exp(-q[1]))
        return v0 * (1 - rho) * np.exp(q[2]), rho * u, rho * (1 - u)

    def nll(q):
        v = garch_variance(e, *unpack(q))[:-1]
        return 0.5 * float(np.sum(np.log(v) + e**2 / v))

    best = min((minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400}) for x0 in ([2.5, -1.5, 0.0], [3.5, -2.0, 0.0])), key=lambda o: o.fun)
    return mu, unpack(best.x)


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def har_inputs(rv: np.ndarray, ev: pd.DataFrame) -> dict[str, np.ndarray]:
    """Everything a HAR model may use at origin t (row t): realised variances up to day t, and the scheduled events of day t+1 (known in advance)."""
    s = pd.Series(rv)
    d, w, m = s, s.rolling(5).mean(), s.rolling(22).mean()
    nxt = ev.shift(-1)
    return {"d": d.to_numpy(), "w": w.to_numpy(), "m": m.to_numpy(), "nfp1": nxt["nfp"].to_numpy(), "cpi1": nxt["cpi"].to_numpy(), "fomc1": nxt["fomc"].to_numpy(),
            "fomc0": ev["fomc"].to_numpy(), "y": np.r_[rv[1:], np.nan]}


def design(inp: dict[str, np.ndarray], kind: str) -> np.ndarray:
    one = np.ones(len(inp["d"]))
    if kind == "levels":
        return np.column_stack([one, inp["d"], inp["w"], inp["m"]])
    logs = [np.log(inp["d"]), np.log(inp["w"]), np.log(inp["m"])]
    if kind == "logs":
        return np.column_stack([one, *logs])
    return np.column_stack([one, *logs, inp["nfp1"], inp["cpi1"], inp["fomc1"], inp["fomc0"]])


def qlike(target: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.log(f) + target / f


def hac_p(d: np.ndarray, lag: int = 5) -> float:
    db = d.mean()
    lrv = d.var() + 2 * sum((1 - j / (lag + 1)) * np.mean((d[j:] - db) * (d[:-j] - db)) for j in range(1, lag + 1))
    return float(2 * (1 - norm.cdf(abs(db / np.sqrt(max(lrv, 1e-18) / len(d))))))


# ------------------------------------------------------------------------------------------------------------------------- the run
def lstm_block(Fm: np.ndarray, tr: np.ndarray, te: np.ndarray, ylog: np.ndarray) -> np.ndarray:
    """LSTM-RV: windows of standardised HAR-RV-X inputs -> log RV of the next day; returns variance forecasts for the rows te (NaN where a window is incomplete)."""
    from tensorflow import keras

    mu, sd = np.nanmean(Fm[tr], axis=0), np.nanstd(Fm[tr], axis=0)
    sd[sd == 0] = 1.0
    Z = np.clip((Fm - mu) / sd, -10, 10)
    win = np.lib.stride_tricks.sliding_window_view(Z, (WINDOW, Z.shape[1]))[:, 0].astype(np.float32)  # window k ends at row k + WINDOW - 1
    pick = lambda rows: win[rows - (WINDOW - 1)]  # noqa: E731
    okw = lambda rows: (rows >= WINDOW - 1) & np.array([np.isfinite(win[r - (WINDOW - 1)]).all() if r >= WINDOW - 1 else False for r in rows])  # noqa: E731
    tr = tr[okw(tr)]
    out = np.full(len(te), np.nan)
    ok_te = okw(te)
    ym, ys = ylog[tr].mean(), ylog[tr].std()
    ytr = ((ylog[tr] - ym) / ys).astype(np.float32)
    preds, fits = [], []
    for seed in range(SEEDS):
        keras.utils.set_random_seed(seed)
        inp = keras.Input(shape=(WINDOW, Z.shape[1]))
        h = keras.layers.Dense(32, activation="relu")(keras.layers.Dropout(0.2)(keras.layers.LSTM(UNITS)(inp)))
        m = keras.Model(inp, keras.layers.Dense(1)(h))
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        m.fit(pick(tr), ytr, validation_split=0.15, epochs=EPOCHS, batch_size=64, verbose=0, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        preds.append(m.predict(pick(te[ok_te]), batch_size=512, verbose=0)[:, 0]); fits.append(m.predict(pick(tr), batch_size=512, verbose=0)[:, 0])
    resid = ylog[tr] - (np.mean(fits, axis=0) * ys + ym)
    out[ok_te] = np.exp(np.mean(preds, axis=0) * ys + ym + 0.5 * resid.var())
    return out


def run_models(rows: np.ndarray, every: int, a: dict, with_lstm: bool, log=print) -> dict[str, np.ndarray]:
    """Forecasts of every model at the origins `rows` (positions in the daily index), refitted at the first origin of each block of `every` origins."""
    r, rv, inp = a["r"], a["rv"], a["inp"]
    X_lev, X_log, X_x = design(inp, "levels"), design(inp, "logs"), design(inp, "x")
    y, ylog = inp["y"], np.log(inp["y"])
    ok_lev, ok_log, ok_x = (np.isfinite(X).all(axis=1) & np.isfinite(y) for X in (X_lev, X_log, X_x))
    Fm = X_x[:, 1:]  # LSTM inputs: the HAR-RV-X regressors without the intercept
    v_ewma = a["v_ewma"]
    out = {k: np.full(len(rows), np.nan) for k in MODELS}
    out["EWMA 0.94 (shipped, unmatched)"] = v_ewma[rows]
    out["EWMA 0.94 of RV"] = a["v_ewma_rv"][rows]
    t0 = time.time()
    for k0 in range(0, len(rows), every):
        blk = np.arange(k0, min(k0 + every, len(rows))); s = int(rows[k0]); te = rows[blk]
        lo = max(0, s - FIT_LEN); idx = np.arange(lo, s)  # training rows: targets are days <= s, known at the origin
        # level-matched baselines
        have = idx[np.isfinite(y[idx])]
        out["EWMA 0.94 (matched)"][blk] = v_ewma[te] * (y[have].mean() / v_ewma[have].mean())
        x = r[max(1, lo + 1):s + 1]; x = x[np.isfinite(x)]
        mu, par = fit_garch(x[-FIT_LEN:])
        seg = r[max(1, lo + 1):te.max() + 1]; e_all = np.where(np.isfinite(seg), seg - mu, 0.0)
        v = garch_variance(e_all, *par); base = max(1, lo + 1)
        gv = lambda pos: v[pos - base + 1]  # noqa: E731  variance forecast for day pos+1 given returns through day pos
        out["GARCH(1,1) (matched)"][blk] = gv(te) * (y[have].mean() / gv(have).mean())
        # HAR family
        t_lev, t_log, t_x = idx[ok_lev[idx]], idx[ok_log[idx]], idx[ok_x[idx]]
        b = ols(X_lev[t_lev], y[t_lev]); out["HAR-RV levels"][blk] = np.maximum(X_lev[te] @ b, 0.1 * y[t_lev].mean())
        b = ols(X_log[t_log], ylog[t_log]); s2 = float(np.var(ylog[t_log] - X_log[t_log] @ b, ddof=X_log.shape[1])); out["HAR-RV logs"][blk] = np.exp(X_log[te] @ b + 0.5 * s2)
        b = ols(X_x[t_x], ylog[t_x]); s2 = float(np.var(ylog[t_x] - X_x[t_x] @ b, ddof=X_x.shape[1])); out["HAR-RV-X"][blk] = np.exp(X_x[te] @ b + 0.5 * s2)
        if with_lstm:
            out["LSTM-RV"][blk] = lstm_block(Fm, idx[ok_x[idx]], te, ylog)
            if (k0 // every) % 10 == 0:
                log(f"   block {k0 // every + 1}/{-(-len(rows) // every)}  {time.time() - t0:.0f}s")
    return out


def evaluate(rows: np.ndarray, f: dict[str, np.ndarray], a: dict) -> dict:
    """Scores on one common sample; primary target RV(t+1), secondary target the squared futures return of day t+1."""
    rvn, r2n = a["inp"]["y"][rows], np.r_[a["r"][1:], np.nan][rows] ** 2
    models = [m for m in MODELS if np.isfinite(f[m]).any()]
    common = np.isfinite(rvn) & np.isfinite(r2n) & np.all([np.isfinite(f[m]) for m in models], axis=0)
    res = {"n": int(common.sum()), "first": str(a["dates"][rows[common][0]].date()), "last": str(a["dates"][rows[common][-1]].date()), "models": {}}
    base = {tg: qlike(t[common], f["EWMA 0.94 (matched)"][common]) for tg, t in (("rv", rvn), ("r2", r2n))}
    for name in models + ["EWMA 0.94 (shipped, unmatched)"]:
        row = {}
        for tg, t in (("rv", rvn), ("r2", r2n)):
            ff, tt = f[name][common], t[common]
            q = qlike(tt, ff)
            sl = np.polyfit(ff, tt, 1)[0]
            row[tg] = {"qlike": float(q.mean()), "qlike_vs_ewma": float((q - base[tg]).mean()), "dm_p_vs_ewma": None if name == "EWMA 0.94 (matched)" else hac_p(q - base[tg]),
                       "mse_1e8": float(np.mean((ff - tt) ** 2) * 1e8), "mz_r2": float(np.corrcoef(ff, tt)[0, 1] ** 2), "mz_slope": float(sl), "bias": float(ff.mean() / tt.mean())}
        res["models"][name] = row
    pairs = {"HAR-RV-X vs HAR-RV logs": ("HAR-RV-X", "HAR-RV logs"), "LSTM-RV vs HAR-RV-X": ("LSTM-RV", "HAR-RV-X"), "HAR-RV logs vs GARCH(1,1) (matched)": ("HAR-RV logs", "GARCH(1,1) (matched)")}
    res["pairs"] = {}
    for label, (m1, m2) in pairs.items():
        if m1 in models and m2 in models:
            d = qlike(rvn[common], f[m1][common]) - qlike(rvn[common], f[m2][common])
            res["pairs"][label] = {"qlike_diff": float(d.mean()), "dm_p": hac_p(d)}
    return res


def by_year(rows: np.ndarray, f: dict[str, np.ndarray], a: dict) -> list[dict]:
    rvn = a["inp"]["y"][rows]; years = a["dates"][rows].year
    models = [m for m in MODELS if np.isfinite(f[m]).any()]
    common = np.isfinite(rvn) & np.all([np.isfinite(f[m]) for m in models], axis=0)
    out = []
    for yr in sorted(set(years[common])):
        m = common & (years == yr)
        out.append({"year": int(yr), "n": int(m.sum()), **{k: float(qlike(rvn[m], f[k][m]).mean()) for k in models}})
    return out


def event_stats(a: dict) -> dict:
    """Descriptive: realised variance on scheduled-release days against the other days, full sample (log RV, Welch t-test)."""
    rv, ev = a["rv"], a["ev"]
    flags = {"NFP day": ev["nfp"], "CPI day": ev["cpi"], "FOMC decision day": ev["fomc"], "day after FOMC": ev["fomc"].shift(1).fillna(0)}
    out, lrv = {}, np.log(rv)
    for name, fl in flags.items():
        fl = fl.to_numpy() == 1; ok = np.isfinite(lrv)
        e, o = lrv[ok & fl], lrv[ok & ~fl]
        out[name] = {"n_event_days": int(len(e)), "mean_rv_ratio": float(np.exp(e).mean() / np.exp(o).mean()), "mean_log_diff": float(e.mean() - o.mean()), "welch_t": float(ttest_ind(e, o, equal_var=False).statistic)}
    return out


def build(args) -> dict:
    px = load_prices(args.prices)
    dates = pd.DatetimeIndex(px["date"])
    r = np.r_[np.nan, np.diff(np.log(px["close"].to_numpy(float)))]
    h = load_hourly_bid(args.raw_dir)
    in_dev = (dates >= DEV[0]) & (dates <= DEV[1])
    corr = {}
    for name, mins in BOUNDARIES.items():
        rvdf = realised_variance(h, dates, mins)
        ok = in_dev & np.isfinite(rvdf["s"].to_numpy()) & np.isfinite(r)
        corr[name] = float(np.corrcoef(rvdf["s"].to_numpy()[ok], r[ok])[0, 1])
    best = max(corr, key=corr.get)
    rvdf = realised_variance(h, dates, BOUNDARIES[best])
    rv = rvdf["rv"].to_numpy()
    ev = event_flags(dates)
    first_rv = int(np.flatnonzero(np.isfinite(rv))[0])
    init = float(np.nanvar(r[1:251]))
    inp = har_inputs(rv, ev)
    return {"px": px, "dates": dates, "r": r, "rv": rv, "ev": ev, "inp": inp, "boundary": best, "boundary_corr": corr, "first_rv": first_rv, "last_rv": int(np.flatnonzero(np.isfinite(rv))[-1]),
            "v_ewma": ewma_path(r, LAM, init, True), "v_ewma_rv": ewma_path(rv, LAM, float(np.nanmean(rv[first_rv:first_rv + 22])), False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=HERE.parent / "hourly" / "data" / "raw")
    ap.add_argument("--prices", default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "paper_e6")
    ap.add_argument("--dev", action="store_true", help="debug run on the development window only: prints, writes nothing")
    ap.add_argument("--skip-lstm", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    a = build(args)
    dates = a["dates"]
    print(f"daily closes {len(dates)} ({dates[0].date()} to {dates[-1].date()}); RV from {dates[a['first_rv']].date()} to {dates[a['last_rv']].date()}; boundary {a['boundary']} (correlations {a['boundary_corr']})")
    if args.dev:
        rows = np.flatnonzero((dates >= DEV[0]) & (dates <= DEV[1]))
        f = run_models(rows, EVERY_LONG, a, with_lstm=False)
        res = evaluate(rows, f, a)
        print("DEV (debug only):", {k: round(v["rv"]["qlike"], 3) for k, v in res["models"].items()}, "n", res["n"])
        return
    p = pd.read_csv(HERE / "results" / "wf_predictions.csv", parse_dates=["date"])
    pos_all = {d: i for i, d in enumerate(dates)}
    prim = np.array([pos_all[d] for d in p["date"]])
    # consistency checks with the main run (no model comparison involved)
    assert np.allclose(px_close(a, prim), p["meta_close"].to_numpy(), atol=1e-6), "daily closes differ from the main run"
    v_sig = pd.read_csv(HERE / "results" / "volatility_forecast.csv")["sigma_ewma_pct"].to_numpy() / 100
    # the shipped EWMA ran on a dataset that lacks the embargo days at the split boundaries, so the two differ by up to 10% in the first test rows and by under 1% later
    rel = np.abs(np.sqrt(a["v_ewma"][prim]) / v_sig - 1)
    assert np.median(rel) < 1e-3 and rel[100:].max() < 1e-2, "EWMA 0.94 differs from the shipped one"
    prim = prim[(prim + 1 <= a["last_rv"]) & np.isfinite(a["inp"]["y"][prim])]
    long_rows = np.flatnonzero(dates >= LONG_START)
    long_rows = long_rows[(long_rows + 1 <= a["last_rv"])]
    out = {"protocol": {"fit_len": FIT_LEN, "every_primary": EVERY_PRIMARY, "every_long": EVERY_LONG, "lambda": LAM, "min_bars": MIN_BARS, "dev_window": DEV, "long_start": LONG_START,
                        "boundary": a["boundary"], "boundary_correlation_on_dev": a["boundary_corr"], "lstm": {"window": WINDOW, "units": UNITS, "epochs": EPOCHS, "seeds": SEEDS}, "family_bonferroni_level": 0.05 / len(FAMILY)},
           "data": {"rv_first": str(dates[a["first_rv"]].date()), "rv_last": str(dates[a["last_rv"]].date()), "rv_days": int(np.isfinite(a["rv"]).sum()),
                    "primary_origins": int(len(prim)), "long_origins": int(len(long_rows))}}
    print(f"PRIMARY {len(prim)} origins ({dates[prim[0]].date()} .. {dates[prim[-1]].date()}), LONG {len(long_rows)} origins")
    fp = run_models(prim, EVERY_PRIMARY, a, with_lstm=not args.skip_lstm)
    out["primary"] = evaluate(prim, fp, a)
    print("primary done", f"{time.time() - t0:.0f}s")
    fl = run_models(long_rows, EVERY_LONG, a, with_lstm=False)
    out["long"] = evaluate(long_rows, fl, a)
    out["long_by_year"] = by_year(long_rows, fl, a)
    out["events"] = event_stats(a)
    out["runtime_seconds"] = round(time.time() - t0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "e6_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    cols = {"date": dates[prim], "rv_next": a["inp"]["y"][prim], "r2_next": np.r_[a["r"][1:], np.nan][prim] ** 2, **{k: v for k, v in fp.items()}}
    pd.DataFrame(cols).to_csv(args.out_dir / "e6_primary_forecasts.csv", index=False)
    print("written", args.out_dir)


def px_close(a: dict, pos: np.ndarray) -> np.ndarray:
    return a["px"]["close"].to_numpy(float)[pos]


if __name__ == "__main__":
    main()
