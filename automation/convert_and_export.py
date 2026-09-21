"""
One-time export step: shrink the trained model down to exactly what the
daily Lambda needs for INFERENCE ONLY, and nothing else.

Run this once (and again after every full retrain) wherever TensorFlow is
already installed — the SageMaker Notebook Instance, or the student's Mac
(both already have `tensorflow>=2.15` per requirements.txt). It reads the
existing training artifacts from S3 and writes two new, small, ADDITIVE
files alongside them — it never modifies or deletes
lstm_model.keras/feature_scaler.pkl/target_scaler.pkl, so notebooks 05/06
and the Streamlit dashboard keep working exactly as before.

    python convert_and_export.py

Why this split exists (training stack vs. inference stack):
  - Training/evaluation (notebooks 05/06, the dashboard) need the FULL
    TensorFlow + scikit-learn stack — fine, since those run on a SageMaker
    Notebook Instance or a laptop, not in a size-constrained Lambda.
  - The daily automation Lambda (see lambda_function.py) needs to do
    exactly one thing, thousands of times cheaper than training: a single
    forward pass through an already-trained network, plus two elementwise
    (x - mean) / scale transforms. TensorFlow proper is ~500MB+ and drags
    in a training/autodiff runtime the Lambda never uses; scikit-learn's
    StandardScaler object exists only to hold two small arrays at inference
    time. Converting to TFLite + exporting scaler_params.json means the
    Lambda's own dependencies are just `tflite-runtime` (a few MB) + numpy
    + pandas — small enough to build reliably into a container image and
    fast enough that a cold start once a day (a warm container is unlikely
    to survive 24 hours between invocations anyway) adds no meaningful
    overhead (billed in ms on Lambda's free tier).
"""

import json
import pickle
import tempfile

import awswrangler as wr
import numpy as np
import tensorflow as tf

S3_BUCKET = "gold-lstm-forecast"
MODEL_PATH = f"s3://{S3_BUCKET}/gold/xauusd_daily/features"

FEATURES = ["close", "return", "ma7", "ma14", "ma30", "ma60", "volatility_7", "momentum_7"]
SEQ_LEN = 60


def main():
    tmpdir = tempfile.mkdtemp()
    print(f"Working dir: {tmpdir}")

    # --- 1. Download existing training artifacts (read-only) -------------
    for fname in ["lstm_model.keras", "feature_scaler.pkl", "target_scaler.pkl"]:
        wr.s3.download(path=f"{MODEL_PATH}/{fname}", local_file=f"{tmpdir}/{fname}")
        print(f"  Downloaded {fname}")

    model = tf.keras.models.load_model(f"{tmpdir}/lstm_model.keras")
    with open(f"{tmpdir}/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(f"{tmpdir}/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    # --- 2. Convert to TFLite ---------------------------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Default (float32) conversion — deliberately NOT applying
    # DEFAULT/dynamic-range quantization here. Quantization trades a small,
    # usually-negligible accuracy loss for a smaller/faster model; for a
    # model this small (2 LSTM layers, <50k params) the size difference is
    # irrelevant, and this project already has a documented, carefully
    # interpreted accuracy story (MAE/RMSE/directional accuracy) — silently
    # perturbing predictions via quantization is not a trade worth making
    # here. Left as a labeled, deliberate non-default, not an oversight.
    tflite_model = converter.convert()

    tflite_path = f"{tmpdir}/lstm_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite model size: {len(tflite_model) / 1024:.1f} KB "
          f"(vs. Keras .keras file on disk — compare with `ls -la` if curious)")

    # --- 3. Sanity-check the conversion introduced no meaningful drift ---
    # Build one real window from the model's own expected input shape using
    # random data drawn from a standard-normal (matches the *scaled* feature
    # distribution StandardScaler produces) purely to compare Keras vs.
    # TFLite numerically — this is NOT a real price, just a conversion check.
    rng = np.random.default_rng(42)
    dummy_X = rng.standard_normal((1, SEQ_LEN, len(FEATURES))).astype(np.float32)

    keras_pred = model.predict(dummy_X, verbose=0)[0][0]

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    print(f"  TFLite input shape : {in_details[0]['shape']} ({in_details[0]['dtype']})")
    print(f"  TFLite output shape: {out_details[0]['shape']} ({out_details[0]['dtype']})")
    interpreter.set_tensor(in_details[0]["index"], dummy_X)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(out_details[0]["index"])[0][0]

    abs_diff = abs(float(keras_pred) - float(tflite_pred))
    print(f"  Keras prediction (scaled) : {keras_pred:.6f}")
    print(f"  TFLite prediction (scaled): {tflite_pred:.6f}")
    print(f"  Absolute difference       : {abs_diff:.6e}")
    if abs_diff > 1e-4:
        raise RuntimeError(
            f"TFLite conversion drift ({abs_diff:.6e}) exceeds the 1e-4 tolerance — "
            "do not deploy this .tflite file; investigate before proceeding."
        )
    print("  OK — conversion drift within tolerance.")

    # --- 4. Export scaler params (mean_/scale_ only, no pickle/sklearn) --
    scaler_params = {
        "feature_scaler": {
            "features": FEATURES,
            "mean": feature_scaler.mean_.tolist(),
            "scale": feature_scaler.scale_.tolist(),
        },
        "target_scaler": {
            "mean": target_scaler.mean_.tolist(),
            "scale": target_scaler.scale_.tolist(),
        },
    }
    params_path = f"{tmpdir}/scaler_params.json"
    with open(params_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  Exported scaler_params.json: {scaler_params}")

    # --- 5. Upload both new artifacts (additive — nothing else touched) --
    wr.s3.upload(local_file=tflite_path, path=f"{MODEL_PATH}/lstm_model.tflite")
    wr.s3.upload(local_file=params_path, path=f"{MODEL_PATH}/scaler_params.json")
    print(f"  Uploaded -> {MODEL_PATH}/lstm_model.tflite")
    print(f"  Uploaded -> {MODEL_PATH}/scaler_params.json")
    print()
    print("Done. Existing lstm_model.keras / feature_scaler.pkl / target_scaler.pkl")
    print("were not modified — the dashboard and notebooks 05/06 are unaffected.")
    print("Re-run this script after every full retrain to refresh these two files.")


if __name__ == "__main__":
    main()
