"""E2d (README section 15.8): what is the positive R-squared of the E2 winner made of? A post-hoc mechanism check on the timing of the closes.

STATUS. Written on 2026-11-26 AFTER the E2 results were known (the winner of the frozen protocol, ridge on the "X" features, passed the STRICT test). It is DESCRIPTIVE, not
confirmatory: it does not select or change anything in E2, and everything it computes is reported whatever it shows. The question is whether the predictability is
(a) information that arrives after the COMEX gold settlement (13:30 ET) but before the other closes of the X set (S&P 500 and TIP 16:00 ET, DXY and the 10-year yield about
17:00 ET), so that it is already inside the gold market by the time the forecast is made and only looks like a forecast because the target starts at the stale settlement, or
(b) a genuinely delayed reaction.

Decomposition of the target (log return from the settlement of day t to the settlement of day t + 1): y(t) = r_post(t) + r_rest(t), where r_post is the XAU/USD spot return
between 14:00 and 17:00 ET on day t (Dukascopy hourly bid bars closing at 15:00, 16:00 and 17:00 ET; a variant adds the bar closing at 14:00, which starts 30 minutes before the
settlement) and r_rest = y - r_post is everything later. r_post has ALREADY HAPPENED when a forecast is made after the close of day t, so a fair "no change" yardstick at that
moment is the spot price at 17:00 ET, not the stale settlement.

Computed on the frozen E2 forecasts (P-A, the DEV-selected winner; the named X hypothesis ridge 1000 with an expanding window), on TEST (2020 onwards) and DEV, for origins
with hourly data (the hourly files end 2026-09-25):
  * variance shares and correlations of y, r_post, r_rest;
  * correlation (Pearson and Spearman with p) of the forecast with y, r_post and r_rest;
  * the Clark-West-type statistic 2 (target - b)(f - b) with the target y and with the target r_rest (b = the E2 rolling-mean benchmark);
  * OLS with Newey-West errors (5 lags): y on [r_post, f - b] and r_rest on [f - b]; R-squared of f against the "no change from the 17:00 spot" forecast;
  * how much of r_post the same-day macro returns (DXY, silver, S&P 500, 10-year yield, TIP) explain (contemporaneous R-squared, TEST).

Usage: python mechanism_check.py [--out results/paper_e2/e2d_mechanism.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from multiplicity_experiment import hac_t  # noqa: E402

RAW = HERE.parent / "hourly" / "data" / "raw"
OUT = HERE / "results" / "paper_e2"
LAG = 5
TEST_START = "2020-01-01"
DEV = ("2011-01-03", "2019-12-31")


def load_hourly(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("xauusd-h1-????-??.csv"))
    h = pd.concat([pd.read_csv(f, usecols=["timestamp", "close"]) for f in files if f.stat().st_size > 0], ignore_index=True)
    h = h.drop_duplicates("timestamp").sort_values("timestamp")
    h = h[h["close"] > 0].reset_index(drop=True)
    h["open_utc"] = pd.to_datetime(h["timestamp"], unit="ms", utc=True)
    return h[["open_utc", "close"]]


def post_settlement_returns(h: pd.DataFrame) -> pd.DataFrame:
    """Per New York calendar date: r_post (bars closing 15:00, 16:00, 17:00 ET) and r_post4 (also 14:00); NaN unless every bar exists and is consecutive."""
    close_et = (h["open_utc"] + pd.Timedelta(hours=1)).dt.tz_convert("America/New_York")
    d = pd.DataFrame({"date": close_et.dt.normalize().dt.tz_localize(None), "hhmm": close_et.dt.hour * 60 + close_et.dt.minute, "lr": np.log(h["close"]).diff().to_numpy(),
                      "gap": h["open_utc"].diff().dt.total_seconds().to_numpy()})
    d = d[(d["gap"] == 3600) & d["hhmm"].isin([14 * 60, 15 * 60, 16 * 60, 17 * 60])]
    piv = d.pivot_table(index="date", columns="hhmm", values="lr", aggfunc="sum")
    out = pd.DataFrame(index=piv.index)
    c3 = [c for c in (15 * 60, 16 * 60, 17 * 60)]
    out["r_post"] = piv[c3].sum(axis=1, min_count=3) if all(c in piv for c in c3) else np.nan
    out["r_post4"] = piv[[14 * 60] + c3].sum(axis=1, min_count=4) if all(c in piv for c in [14 * 60] + c3) else np.nan
    return out


def ols_hac(y: np.ndarray, X: np.ndarray, lag: int = LAG) -> dict:
    """OLS with an intercept and Newey-West (Bartlett) standard errors; X has one column per regressor."""
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for k in range(1, lag + 1):
        G = xe[k:].T @ xe[:-k]
        S += (1 - k / (lag + 1)) * (G + G.T)
    A = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(A @ S @ A))
    r2 = 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return {"beta": beta.tolist(), "se": se.tolist(), "t": (beta / se).tolist(), "r2": r2, "n": int(len(y))}


def corr_block(x: np.ndarray, y: np.ndarray, name: str) -> dict:
    r, rs = float(np.corrcoef(x, y)[0, 1]), float(spearmanr(x, y)[0])
    n = len(x)
    p = float(2 * (1 - student_t.cdf(abs(rs) * np.sqrt((n - 2) / max(1 - rs**2, 1e-12)), n - 2)))
    return {f"pearson_{name}": r, f"spearman_{name}": rs, f"spearman_p_{name}": p}


def analyse(y: np.ndarray, rpost: np.ndarray, f: np.ndarray, b: np.ndarray) -> dict:
    rrest = y - rpost
    d = f - b
    out = {"n": int(len(y)), "std_y": float(y.std()), "std_r_post": float(rpost.std()), "std_r_rest": float(rrest.std()), "var_share_r_post": float(rpost.var() / y.var()),
           "corr_y_r_post": float(np.corrcoef(y, rpost)[0, 1])}
    for nm, x in (("y", y), ("r_post", rpost), ("r_rest", rrest)):
        out.update(corr_block(f, x, nm))
    cw_y, cw_rest = 2 * (y - b) * d, 2 * (rrest - b) * d
    out["cw_t_target_y"], out["cw_p_target_y"] = hac_t(cw_y, LAG), float(1 - norm.cdf(hac_t(cw_y, LAG)))
    out["cw_t_target_r_rest"], out["cw_p_target_r_rest"] = hac_t(cw_rest, LAG), float(1 - norm.cdf(hac_t(cw_rest, LAG)))
    out["r2_vs_b_target_y"] = float(1 - np.sum((y - f) ** 2) / np.sum((y - b) ** 2))
    out["r2_vs_b_target_r_rest"] = float(1 - np.sum((rrest - f) ** 2) / np.sum((rrest - b) ** 2))
    out["r2_f_vs_no_change_from_17h_spot"] = float(1 - np.sum((y - f) ** 2) / np.sum((y - rpost) ** 2))
    out["r2_r_post_as_forecast_vs_b"] = float(1 - np.sum((y - rpost) ** 2) / np.sum((y - b) ** 2))
    out["ols_y_on_r_post_and_f"] = ols_hac(y, np.column_stack([rpost, d]))
    out["ols_y_minus_r_post_on_f"] = ols_hac(rrest, d[:, None])
    out["ols_y_on_f"] = ols_hac(y, d[:, None])
    return out


def macro_explains_r_post(prices: pd.DataFrame, macro: pd.DataFrame, rp: pd.DataFrame, lo: str, hi: str | None) -> dict:
    """Contemporaneous R-squared of r_post on the same-day macro returns (aligned as in the E2 features: the latest close at or before the gold date)."""
    from build_daily_dataset import align_macro

    d = prices[["date", "close"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.drop_duplicates("date").sort_values("date").set_index("date")
    a = align_macro(d.index, macro)
    X = pd.DataFrame({"dxy": np.log(a["dxy"]).diff(), "silver": np.log(a["silver"]).diff(), "spx": np.log(a["spx"]).diff(), "tip": np.log(a["tip"]).diff(), "yield10y": a["yield10y"].diff()})
    j = X.join(rp["r_post"], how="inner").dropna()
    j = j[j.index >= pd.Timestamp(lo)]
    if hi:
        j = j[j.index <= pd.Timestamp(hi)]
    Z = (j[["dxy", "silver", "spx", "tip", "yield10y"]] - j[["dxy", "silver", "spx", "tip", "yield10y"]].mean()) / j[["dxy", "silver", "spx", "tip", "yield10y"]].std()
    res = ols_hac(j["r_post"].to_numpy(), Z.to_numpy())
    return {"n": res["n"], "r2": res["r2"], "t_by_regressor": dict(zip(["const", "dxy", "silver", "spx", "tip", "yield10y"], res["t"])),
            "std_coef_by_regressor": dict(zip(["const", "dxy", "silver", "spx", "tip", "yield10y"], res["beta"]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "e2d_mechanism.json")
    ap.add_argument("--forecasts", type=Path, default=OUT / "forecasts" / "linear_h1.npz")
    a = ap.parse_args()
    rp = post_settlement_returns(load_hourly(RAW))
    fz = pd.read_csv(OUT / "e2_frozen_forecasts_h1.csv", parse_dates=["date"]).set_index("date")
    series = {"P-A dev-selected winner": fz["dev_best"]}
    if a.forecasts.exists():
        z = np.load(a.forecasts, allow_pickle=False)
        names = [str(s) for s in z["names"]]
        col = names.index("ridge[a1000]|X|Lexp")
        dates = pd.to_datetime(z["dates"].astype("datetime64[ns]"))
        series["named X hypothesis (ridge 1000, expanding)"] = pd.Series(z["F"][:, col].astype(float), index=dates)
    res: dict = {"hourly_last_date": str(rp.index.max().date()), "windows": {}}
    for wname, (lo, hi) in {"test": (TEST_START, None), "dev": DEV}.items():
        res["windows"][wname] = {}
        for sname, s in series.items():
            j = pd.concat([fz[["y", "b"]], rp, s.rename("f")], axis=1, join="inner").dropna(subset=["y", "b", "f", "r_post"])
            j = j[j.index >= pd.Timestamp(lo)]
            if hi:
                j = j[j.index <= pd.Timestamp(hi)]
            blk = analyse(j["y"].to_numpy(), j["r_post"].to_numpy(), j["f"].to_numpy(), j["b"].to_numpy())
            j4 = j.dropna(subset=["r_post4"])
            blk["variant_with_the_14h_bar"] = analyse(j4["y"].to_numpy(), j4["r_post4"].to_numpy(), j4["f"].to_numpy(), j4["b"].to_numpy())
            res["windows"][wname][sname] = blk
    prices, macro = pd.read_parquet(HERE / "data" / "e2_prices.parquet"), pd.read_parquet(HERE / "data" / "e2_macro.parquet")
    res["macro_explains_r_post_test"] = macro_explains_r_post(prices, macro, rp, TEST_START, None)
    a.out.write_text(json.dumps(res, indent=2, default=float))
    for w in ("test", "dev"):
        for sname, b in res["windows"][w].items():
            print(f"[{w}] {sname}: n={b['n']} | corr(f, y) {b['pearson_y']:+.3f} | corr(f, r_post) {b['pearson_r_post']:+.3f} | corr(f, r_rest) {b['pearson_r_rest']:+.3f} (Spearman p {b['spearman_p_r_rest']:.3f}) | "
                  f"CW p on y {b['cw_p_target_y']:.4f}, on r_rest {b['cw_p_target_r_rest']:.4f} | R2 vs b: y {b['r2_vs_b_target_y']:+.4f}, r_rest {b['r2_vs_b_target_r_rest']:+.4f} | "
                  f"y~r_post+f: t(f) {b['ols_y_on_r_post_and_f']['t'][2]:+.2f}")
    m = res["macro_explains_r_post_test"]
    print(f"macro same-day returns explain {m['r2']:.3f} of r_post (n={m['n']}, TEST)")


if __name__ == "__main__":
    main()
