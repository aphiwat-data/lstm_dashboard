"""Add the cross-asset model (the winner of the frozen search E2, README section 15.8) to the main results so that the dashboard can show it next to the LSTM.

What it does, on the 449 rows of the main walk-forward run (results/wf_predictions.csv):
  * adds the column pred_ret_xasset: the committed E2 forecasts of the winner (ridge, alpha 100, last 1,000 labelled days, six inputs, refit every 21 origins) for the same dates;
  * adds models.xasset to walk_forward_results.json (the same report as for the other models) and the interval and point metrics to regression_metrics.json;
  * stores the cross-asset forecast beside the LSTM's as latest_forecast_xasset (same as-of date), the page's fallback for the cross-asset model when the live pipeline cannot be read.
The LSTM columns, numbers and latest_forecast are not touched. Run it once after a refresh of the main run (the previous outputs live in results/snapshot_*); it is idempotent.

Usage: python add_crossasset_model.py [--results results] [--e2 results/paper_e2/e2_frozen_forecasts_h1.csv]   (needs data/e2_prices.parquet and data/e2_macro.parquet: python search_experiment.py prepare)"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hourly"))
import crossasset as ca  # noqa: E402
from regression_metrics import EWMA_LAMBDA, Z90, write_metrics  # noqa: E402
from train_models import price_report  # noqa: E402


def add(results: Path, e2_csv: Path, prices_path: Path, macro_path: Path) -> dict:
    raw = pd.read_csv(results / "wf_predictions.csv", dtype=str, keep_default_na=False)  # as text, so the columns of the other models are written back byte for byte
    raw = raw.drop(columns=[c for c in ("pred_ret_xasset", "p_up_xasset") if c in raw.columns])
    preds = pd.read_csv(results / "wf_predictions.csv", float_precision="round_trip")
    dates = pd.to_datetime(preds["date"])
    e2 = pd.read_csv(e2_csv, parse_dates=["date"]).set_index("date")
    m = e2.reindex(dates)
    assert m["dev_best"].notna().all(), f"E2 has no forecast for {int(m['dev_best'].isna().sum())} of the {len(preds)} rows"
    assert np.allclose(m["y"].to_numpy(float), preds["y_ret_next"].to_numpy(float), atol=1e-9) and np.allclose(m["close"].to_numpy(float), preds["meta_close"].to_numpy(float), rtol=1e-9), \
        "the E2 rows are not the same days and returns as the main run"
    preds["pred_ret_xasset"] = m["dev_best"].to_numpy(float)
    raw["pred_ret_xasset"] = [repr(float(v)) for v in preds["pred_ret_xasset"]]
    raw["p_up_xasset"] = ""
    raw.to_csv(results / "wf_predictions.csv", index=False)

    wf = json.loads((results / "walk_forward_results.json").read_text())
    ev = preds[["meta_close", "meta_next_close", "y_ret_next", "y_dir_next"]]
    wf["models"]["xasset"] = price_report(ev, 1, preds["pred_ret_xasset"].to_numpy(float), None)
    as_of = pd.Timestamp(wf["latest_forecast"]["as_of_date"])
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    close = prices.drop_duplicates("date").sort_values("date").set_index("date")["close"].loc[:as_of]
    macro = pd.read_parquet(macro_path).loc[:as_of]
    errors = preds["y_ret_next"].to_numpy(float) - preds["pred_ret_xasset"].to_numpy(float)
    fc = ca.forecast_latest(close, macro, errors)
    wf["latest_forecast_xasset"] = {"as_of_date": fc["as_of_date"], "latest_close": fc["latest_close"], "predicted_return": fc["predicted_return"], "predicted_next_close": fc["predicted_next_close"],
                             "p_up": None, "interval_90_close": fc["interval_90_close"], "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "model": ca.MODEL_NAME, "model_trained_through": fc["trained_through"],
                             "interval_method": f"Gaussian, sigma = EWMA(lambda {EWMA_LAMBDA}) of all past test errors, z = {Z90:.3f}"}
    (results / "walk_forward_results.json").write_text(json.dumps(wf, indent=2, default=float))
    write_metrics(results)
    return {"n": len(preds), "latest": wf["latest_forecast_xasset"], "metrics": json.loads((results / "regression_metrics.json").read_text())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--e2", type=Path, default=HERE / "results" / "paper_e2" / "e2_frozen_forecasts_h1.csv")
    ap.add_argument("--prices", type=Path, default=HERE / "data" / "e2_prices.parquet")
    ap.add_argument("--macro", type=Path, default=HERE / "data" / "e2_macro.parquet")
    a = ap.parse_args()
    out = add(a.results, a.e2, a.prices, a.macro)
    m, x, iv = out["metrics"]["models"], out["metrics"]["models"]["xasset"], out["metrics"]["interval"]["xasset_ewma_gaussian"]
    print(f"{out['n']} rows | cross-asset: MAE ${x['mae_usd']:.2f} (simple guess ${out['metrics']['no_change']['mae_usd']:.2f}, LSTM ${m['lstm']['mae_usd']:.2f}) | RMSE ${x['rmse_usd']:.2f} | "
          f"R2 vs rolling mean {x['r2_os_vs_rolling_mean']:+.4f} | IC {x['ic_spearman']:+.3f} (p {x['ic_p']:.3f}) | DM p {x['dm_p_vs_naive_lag24']:.3f} | 90% band coverage {iv['coverage']:.1%} (n={iv['n']})")
    print("stored latest forecast:", json.dumps(out["latest"]))


if __name__ == "__main__":
    main()
