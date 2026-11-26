"""Export what the daily Lambda needs for the cross-asset model (crossasset.py): the model's past walk-forward errors (which size the first 90% bands) and the macro history.

  serving_artifacts/crossasset_model.npz   errors (y - forecast, oldest first, the 449 test days of the main run), alpha, train_len, feature names, provenance  (committed)
  serving_artifacts/macro_history.csv      date, dxy, silver, spx, yield10y, tip: Yahoo daily closes up to the day of the export                              (NOT committed: vendor data)

The model itself is refit by the Lambda every day, so nothing is trained here. Upload: python ../automation/deploy.py code  (seeds both objects when missing) or by hand to
s3://<bucket>/model/serving/. Usage: python export_crossasset_model.py   (needs data/e2_macro.parquet: python search_experiment.py prepare, and the results written by add_crossasset_model.py)"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crossasset as ca  # noqa: E402


def export(results: Path, macro_path: Path, out_dir: Path) -> tuple[Path, Path]:
    preds = pd.read_csv(results / "wf_predictions.csv")
    errors = (preds["y_ret_next"] - preds["pred_ret_xasset"]).to_numpy(float)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "crossasset_model.npz"
    np.savez(npz, errors=errors, alpha=ca.ALPHA, train_len=ca.TRAIN_LEN, feats=np.array(ca.FEATURES), errors_through=str(preds["date"].iloc[-1])[:10],
             built_from="E2 winner (ridge, alpha 100, last 1,000 days, six inputs), README section 15.5")
    macro = pd.read_parquet(macro_path)[ca.MACRO_COLS].dropna(how="all")
    csv = out_dir / "macro_history.csv"
    macro.rename_axis("date").reset_index().to_csv(csv, index=False)
    return npz, csv


def main() -> None:
    npz, csv = export(HERE / "results", HERE / "data" / "e2_macro.parquet", HERE / "serving_artifacts")
    m = pd.read_csv(csv, parse_dates=["date"])
    z = np.load(npz)
    print(f"{npz.name}: {len(z['errors'])} errors through {z['errors_through']} | {csv.name}: {len(m)} rows {m['date'].min().date()}..{m['date'].max().date()}")


if __name__ == "__main__":
    main()
