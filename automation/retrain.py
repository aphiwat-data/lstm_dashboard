"""Weekly retrain of the serving model, run by hand or from a scheduler on a machine with TensorFlow and AWS credentials (not inside Lambda).

Silver + the Lambda's streaming log (S3) -> features and labels -> final LSTM on the latest 365 labelled days -> serving_model.npz -> S3.
The next Lambda run loads the new weights (a warm container keeps the old one until it is recycled, at most a few hours).
Usage: python retrain.py [--no-upload]"""
import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "daily"))
import boto3  # noqa: E402

import lambda_function as lf  # noqa: E402  (reuses load_prices: Silver + streaming log)
from build_daily_dataset import build  # noqa: E402
from export_serving_model import train_export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    prices = lf.load_prices()
    print("prices", len(prices), prices["date"].min().date(), "->", prices["date"].max().date())
    with tempfile.TemporaryDirectory() as tmp:
        silver = Path(tmp) / "silver.parquet"
        prices.to_parquet(silver)
        df, rep, _ = build(str(silver))
        print("labelled rows", len(df), "through", df.index.max().date())
        path = train_export(df, HERE.parent / "daily" / "results" / "wf_predictions.csv", Path(tmp) / "out")
        if a.no_upload:
            print("trained, not uploaded:", path)
            return
        boto3.client("s3", region_name=lf.REGION).upload_file(str(path), lf.S3_BUCKET, lf.MODEL_KEY)
        print("uploaded ->", f"s3://{lf.S3_BUCKET}/{lf.MODEL_KEY}")


if __name__ == "__main__":
    main()
