"""Export what the daily Lambda needs for the cross-asset model (crossasset.py): the model's past walk-forward errors, which size the first 90% bands.

  serving_artifacts/crossasset_model.npz   errors (y - forecast, oldest first, the 449 test days of the main run), alpha, train_len, feature names, provenance  (committed)

The model itself is refit by the Lambda every day from data it fetches itself, so nothing is trained here. Upload: python ../automation/deploy.py code  (uploads it when missing) or by
hand to s3://<bucket>/model/serving/. Usage: python export_crossasset_model.py   (needs the results written by add_crossasset_model.py)"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crossasset as ca  # noqa: E402


def export(results: Path, out_dir: Path) -> Path:
    preds = pd.read_csv(results / "wf_predictions.csv")
    errors = (preds["y_ret_next"] - preds["pred_ret_xasset"]).to_numpy(float)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "crossasset_model.npz"
    np.savez(npz, errors=errors, alpha=ca.ALPHA, train_len=ca.TRAIN_LEN, feats=np.array(ca.FEATURES), errors_through=str(preds["date"].iloc[-1])[:10],
             built_from="E2 winner (ridge, alpha 100, last 1,000 days, six inputs), README section 15.5")
    return npz


def main() -> None:
    npz = export(HERE / "results", HERE / "serving_artifacts")
    z = np.load(npz)
    print(f"{npz.name}: {len(z['errors'])} errors through {z['errors_through']}")


if __name__ == "__main__":
    main()
