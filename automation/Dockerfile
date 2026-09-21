# Container image for the daily inference-refresh Lambda (lambda_function.py).
#
# WHY a container image instead of a plain zip + layers:
# tflite-runtime and pandas both ship compiled C-extension code that must
# exactly match the target Lambda runtime's OS/libc/CPU architecture.
# Building FROM AWS's own Lambda base image guarantees that match — a zip
# built on a Mac (different libc, possibly arm64) is a well-known source of
# "ImportError: cannot import name '_pywrap...'" failures that only surface
# after deploying, not at build time. Container images also lift the 250MB
# unzipped ceiling a zip-based layer is capped at (up to 10GB), which
# matters if the tensorflow-cpu fallback below is ever needed.

FROM public.ecr.aws/lambda/python:3.10

COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]

# --- IF THE BUILD ABOVE FAILS at `pip install tflite-runtime` -------------
# tflite-runtime's published wheel coverage across Python versions/
# architectures has shifted over time (Google's official builds have not
# always kept pace with new Python releases). This failure happens safely
# at `docker build` time, before anything is deployed — nothing to undo.
#
# Fix: open requirements-lambda.txt, comment out the `tflite-runtime` line
# and uncomment the `tensorflow-cpu` line instead, then rebuild. No change
# to lambda_function.py is needed — it already tries `tflite_runtime` first
# and falls back to `tensorflow.lite`, which exposes the identical
# Interpreter API. The image will be considerably larger (still comfortably
# under the container image's 10GB limit) and cold starts a few hundred ms
# slower; predictions are unaffected either way. See automation/README.md,
# "If tflite-runtime won't install", for the full explanation.
