# Daily Automation — Design, Deployment, and Honest Status

This folder adds one thing to the project: a **daily-scheduled inference
refresh**, so the dashboard's next-day forecast stays current without
manually re-running notebooks. It does **not** add automatic retraining —
see "Why inference-only, not retraining" below for why that's a deliberate
choice, not a shortcut.

**Why daily, and not hourly**: the dataset is daily-granularity — one new
close per trading day, not one per hour — so the schedule was deliberately
set to match that (once/day, timed at 08:00 Thailand time) rather than
polling more often than the underlying data can possibly change. An
earlier draft of this used `rate(1 hour)`; see "Why 08:00 Thailand time
specifically" below for why daily-at-a-chosen-time is both cheaper and a
more defensible design than polling hourly "just in case."

**Status, honestly**: designed, implemented, and internally consistent with
the rest of the pipeline (formulas, paths, and shapes all cross-checked
against `notebooks/03_Feature_Engineering.ipynb`, `04_Preprocessing.ipynb`,
and `dashboard/app.py`), but **not yet deployed or run against the live AWS
account** — it was built in a cloud sandbox with no network path to AWS.
Treat this as a code-complete, ready-to-deploy package, not a claim that a
Lambda is currently running. Section "Before you trust it" below is the
checklist to actually clear before presenting this as a live, working
system rather than a design.

---

## 1. What it does, in one pass

```
EventBridge Scheduler (daily, 01:00 UTC = 08:00 Thailand time)
        |
        v
   Lambda: gold-lstm-forecast-daily-refresh
        |
        1. Read Silver (read-only) -> last known close date
        2. Fetch latest daily bar(s) from Yahoo's chart endpoint (stdlib urllib)
        3. New trading day found?
             no  -> log "no_new_data", exit (the common case)
             yes -> append to bronze/streaming/xauusd_daily_incremental.csv
                    (its OWN log -- never touches the canonical Bronze/Silver/Gold files)
                    recompute the 7 engineered features on the trailing window
                    run TFLite inference (already-trained model, no training here)
                    write gold/xauusd_daily/predictions/latest_forecast.json
```

## 2. Why 08:00 Thailand time specifically

The dataset is **daily granularity** — one new (close, feature) observation
per trading day, not continuously — so the check runs once/day rather than
polling more often than the data can possibly change (an earlier design
used `rate(1 hour)`; almost every one of those 24 daily checks would have
found nothing new, which is correct but wasteful and a weaker story to
present than a schedule chosen to match the data itself).

Picking *when* in the day matters, though: too early and the check
consistently finds nothing (a full extra day of lag before the forecast
catches up), too close to the boundary and it's a coin flip. COMEX Gold's
electronic session halts daily 5:00-6:00pm ET for maintenance, and its
official settlement price is fixed earlier, at 1:30pm ET. Converted to
Thailand time (ICT, UTC+7): the 5pm ET halt lands around 04:00 ICT the next
day, and the 1:30pm ET settlement around 00:30 ICT — so **08:00 ICT**
clears both by a 3-7 hour margin (the exact gap shifts by an hour across US
daylight saving, since Thailand doesn't observe DST but the US does). That
margin is the actual justification for this specific time, not just a
round number that happens to be a reasonable hour to check a dashboard.

`cron(0 1 * * ? *)` in `deploy.sh` (01:00 UTC = 08:00 ICT) implements this.
On weekdays it should reliably find exactly one new trading day; on
Saturday/Sunday (COMEX doesn't trade Friday 5pm ET through Sunday 6pm ET)
it correctly finds nothing and exits — see `lambda_handler()`'s
`no_new_data` branch, which was written to make that the expected, logged,
non-error outcome rather than something that looks like a bug two days a
week.

## 3. Why inference-only, not retraining

Retraining the walk-forward LSTM on this same daily schedule would repeat,
every single day, the single most expensive stage in the whole pipeline —
the same walk-forward training loop that caused CPU-credit exhaustion on
`ml.t3.medium` during development (see `docs/TROUBLESHOOTING.md`) —
regardless of whether one new day of data actually justifies a full
retrain. Doing that daily on a friend's AWS account would be a real cost
and stability risk, not a hypothetical one.

Full retraining belongs on a cadence tied to how much new data has
actually accumulated — the walk-forward loop already encodes this idea via
`RETRAIN_EVERY = 7` in `notebooks/05_LSTM_Training.ipynb`. The natural
extension, **not built here**, is a separate, much-less-frequent scheduled
job (weekly is a reasonable starting point) that re-runs the full
SageMaker training pipeline and then re-runs `convert_and_export.py` to
refresh the artifacts this Lambda reads. That's listed under "Not built
yet" below rather than implemented, to keep this addition scoped and
verifiable rather than speculative.

## 4. Why the write path is isolated (`bronze/streaming/`, not the canonical files)

The canonical `bronze/xauusd_daily/raw/`, `silver/xauusd_daily_clean.parquet`,
and `gold/xauusd_daily/features/xauusd_features.parquet` are owned by the
manual notebook pipeline, and everything downstream (training, evaluation,
the dashboard) trusts their exact schema — Silver in particular carries
columns derived from the 1W/1M timeframes that this function never
re-derives. An unattended daily Lambda and a human occasionally re-running
notebooks are two independent writers; if both touched the same files,
there's a real risk of a race or a malformed row silently corrupting a
column this function doesn't know about.

So this function **never writes** to any canonical Bronze/Silver/Gold path.
It only ever *reads* Silver, and only ever *writes* to its own
`bronze/streaming/` log plus the `predictions/` output. A full manual
notebook re-run is what "absorbs" the streaming log into the canonical
tables (see "Not built yet" — that absorption step isn't automated
either, on purpose, for now). The blast radius of a bug in this function
is therefore capped at its own log and the forecast cache — it structurally
cannot corrupt the training data or the trained model.

## 5. A real bug this surfaced (and fixed, in this function — not yet in the dashboard)

Building this exposed a genuine issue in the existing pipeline:
`03_Feature_Engineering.ipynb`'s `df.dropna()` has no `subset=`, so it drops
**any** row with a null in **any** column — including the most recent row,
whose `target` (next day's close) is always undefined at collection time.
That row's *features* (close, ma7, ma30, ...) are perfectly well-defined;
only `target` is missing, yet the whole row is dropped. The practical
effect: the persisted Gold table's last row is always **one trading day
behind** the true latest known close.

`dashboard/app.py`'s `forecast_next_day()` builds its input window from
`Gold.tail(SEQ_LEN)` — which, because of the above, ends one day short of
today. Its "next-day forecast" is therefore mechanically a prediction for a
value that is *already fully known* from data already collected (it's
sitting right there as the `target` value of Gold's last row) — not a
genuine unknown-future forecast. The model isn't given that known answer as
an input, so the number it produces isn't fabricated, but the day it's
predicting isn't actually "tomorrow" relative to the truly latest data.

This function avoids the bug structurally rather than patching the
symptom: it builds its 60-day window from **Silver's raw close series**
(never target-shifted, never dropna-filtered) plus anything newer in its
own streaming log — so the window's last row is always the *true* latest
known trading day, and the resulting prediction is a genuine
next-day-ahead forecast.

**This is not yet fixed in `dashboard/app.py`** — that file still uses the
one-day-stale `Gold.tail(SEQ_LEN)` approach. The fix there is small (build
the forecast window from Silver instead of Gold, mirroring
`compute_features()` in `lambda_function.py`), but touches a file the
dashboard demo depends on, so it's called out here as a known, understood,
deliberately-deferred issue rather than silently patched — a good "what
problem did you find and how would you fix it" talking point on its own.

## 6. Deploying it

**Prerequisites**: Docker Desktop, and AWS CLI configured with the same
credentials already set up for the dashboard (main README §10 — IAM user
`Auto`, region `ap-southeast-2`). Run this from the student's Mac or the
SageMaker Notebook Instance's terminal — anywhere with real AWS network
access, which the environment that generated this code did not have.

```bash
cd gold-lstm-forecast/automation

# One-time, wherever TensorFlow is already installed (SageMaker or the Mac):
# converts lstm_model.keras -> lstm_model.tflite and exports scaler_params.json.
# Re-run this after every full retrain.
python convert_and_export.py

# First time only: grants Auto the Lambda/ECR/EventBridge Scheduler
# permissions it doesn't currently have (see main README §4 for what Auto
# already has). Omit this flag on later re-deploys.
./deploy.sh --grant-permissions

# Later re-deploys (code changes, etc.):
./deploy.sh
```

`deploy.sh` is idempotent, prints a cost estimate and an AWS-identity check
before doing anything, and ends with a synchronous smoke-test invocation so
a broken function is caught immediately rather than a day later.

### If `tflite-runtime` won't install

`requirements-lambda.txt` and `Dockerfile` both have the fallback inline:
comment out the `tflite-runtime` line, uncomment `tensorflow-cpu`, rebuild.
`lambda_function.py` already tries `tflite_runtime.interpreter` first and
falls back to `tensorflow.lite` — no code change needed either way.

### Tearing it down

```bash
aws scheduler delete-schedule --name gold-lstm-forecast-daily --region ap-southeast-2
aws lambda delete-function --function-name gold-lstm-forecast-daily-refresh --region ap-southeast-2
aws ecr delete-repository --repository-name gold-lstm-forecast-daily-refresh --region ap-southeast-2 --force
aws iam delete-role-policy --role-name gold-lstm-forecast-lambda-exec-role --policy-name gold-lstm-forecast-lambda-least-privilege
aws iam delete-role --role-name gold-lstm-forecast-lambda-exec-role
aws iam delete-role-policy --role-name gold-lstm-forecast-scheduler-role --policy-name invoke-daily-refresh-lambda-only
aws iam delete-role --role-name gold-lstm-forecast-scheduler-role
```
Nothing above touches S3 — the streaming log and forecast JSON are left in
place (they're small; delete them manually with `aws s3 rm` if wanted).

## 7. Cost

Lambda free tier: 1M requests + 400,000 GB-seconds compute/month. This
function runs ~30 times/month (once daily) for well under a second each at
512MB — several orders of magnitude under the free tier, even more
trivially so than the hourly (~720/month) design this replaced. ECR
storage for one small image: a few cents/month at most, if that. Expected
real cost: effectively $0 — the same conclusion already reached for the
rest of this project's AWS usage, extended to the one new piece of
infrastructure this adds.

## 8. Known risks (stated up front, not discovered the hard way)

- **Yahoo's chart endpoint is unofficial and undocumented.** It can change
  shape or start blocking cloud IPs without notice — real fragility a
  static Kaggle CSV never has. Mitigation for a production version: a
  licensed market-data API with an SLA; only `fetch_recent_bars()` would
  need to change.
- **Two independent writers to the same bucket** (this Lambda, and a human
  re-running notebooks) is a known concurrency pattern with its own risks
  even with the isolated write path above — e.g. if a full notebook re-run
  happens to complete its own collection *while* the Lambda is mid-append
  to the streaming log. Low-probability at daily-vs-occasional-manual-run
  frequency, and the isolated write path means the worst case is a
  malformed streaming-log row, never a corrupted canonical file — but it's
  a real, not fully eliminated, race.
- **A missed or failed run costs a full day of staleness**, not an hour —
  the trade-off for far fewer, more meaningful invocations. `range=10d` in
  `fetch_recent_bars()` gives several days of catch-up buffer if a run
  errors out or the function is disabled for a stretch (see the code
  comment there); `automation_log.csv` is where to check for a string of
  `error` outcomes if the forecast looks stale.
- **IAM least-privilege is applied to the Lambda's own execution role, not
  to Auto's console permissions.** Auto keeps its existing broad
  `*FullAccess` policies (already an acknowledged, deliberate dev-speed
  trade-off — see main README §4) plus three more via `--grant-permissions`.
  The unattended, continuously-running piece (the Lambda's role) is scoped
  tightly; the human console user is not. That's the right place to spend
  the effort, not the only place it could be spent.

## 9. Not built yet (honest scope boundary)

- **Automatic absorption** of `bronze/streaming/xauusd_daily_incremental.csv`
  back into the canonical Bronze/Silver/Gold files — currently still a
  manual notebook re-run.
- **Scheduled full retraining** (e.g. weekly) — this function is inference-
  only by design (§2); a separate job would need to trigger a SageMaker
  Processing/Training job on a longer cadence and then re-run
  `convert_and_export.py`.
- **The `dashboard/app.py` one-day-stale forecast fix** described in §4.
- **CloudWatch alarms** on repeated `error` outcomes in
  `bronze/streaming/automation_log.csv` / the function's own CloudWatch
  Logs — right now, checking for failures is a manual log read.
