# Next-Day Gold Price Forecasting with LSTM — Project README

Undergraduate Computer Engineering senior project, Mae Fah Luang University.
This file is the single source of truth for continuing the work — proposal
context, AWS setup, pipeline internals reconciled against the actual code,
every real issue hit and fixed, current evaluation results, and everything
still pending. It merges the written thesis proposal (`Next_day_Gold_
Prices_Forcasting_With_LSTM.pdf`) with what the pipeline *actually* does
today, and calls out every place the two disagree. Read this fully before
making changes.

For the full chronological story of every AWS/code error hit and how it
was fixed (useful for "what problems did you encounter" in the defense),
see **[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)**.

---

## 0. Front Matter (from the proposal)

- **Title**: Next-Day Gold Price Forecasting with LSTM
- **Authors**: Achira Lueablae, Aphiwat Chioewvijit
- **Degree**: Bachelor of Engineering, Major Computer Engineering
- **Advisor**: Asst. Prof. Pattaramon Vuttipittayamongkol
- **Committee**: Asst. Prof. Khwunta Kirimasthong; Asst. Prof. Surapong Uttama
- **Institution**: School of Applied Digital Technology, Mae Fah Luang University

**Abstract** (as written in the proposal): This project develops an
end-to-end data workflow for predicting the next-day gold closing price
using a Long Short-Term Memory (LSTM) time-series model. Historical daily
Gold Futures (GC=F) data, including OHLCV features from 2004 to 2026, are
utilized. The system follows an ELT architecture, where raw data are
ingested, cleaned, transformed, and enhanced through feature engineering
techniques such as daily returns, moving averages, and rolling volatility.
A 60-day sliding window is applied to construct supervised learning
sequences for model training. The LSTM model is evaluated using regression
metrics including MAE and RMSE. In addition to prediction, trend and
volatility analyses are performed and presented through an interactive
dashboard.

**Keywords**: LSTM, Gold Price Prediction, Time-Series Forecasting, ELT,
Financial Analytics.

---

## 1. Repository Structure

```
gold-lstm-forecast/
├── README.md                    ← this file
├── docs/
│   └── TROUBLESHOOTING.md       ← full chronological log of every error + fix
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Preprocessing.ipynb
│   ├── 05_LSTM_Training.ipynb
│   └── 06_Model_Evaluation.ipynb
├── dashboard/
│   ├── app.py                   ← Streamlit dashboard (local, connects to S3) — work in progress, see §10
│   └── requirements.txt
├── automation/                  ← daily inference-refresh Lambda — see §11 and automation/README.md
│   ├── README.md                ← full design rationale + deploy/teardown steps
│   ├── lambda_function.py       ← the daily handler (inference-only, no training)
│   ├── convert_and_export.py    ← one-time: Keras -> TFLite + scaler param export
│   ├── Dockerfile, requirements-lambda.txt
│   ├── deploy.sh                ← idempotent: ECR + Lambda + EventBridge Scheduler
│   └── iam_*.json, scheduler_*.json  ← least-privilege policies for the Lambda's own role
└── .gitignore                   ← excludes venv/, .aws/, and pulled *.npy/*.pkl/*.keras/*.parquet artifacts
```

All 6 notebooks in `notebooks/` are the **current, corrected** versions —
every fix described in this README and in `docs/TROUBLESHOOTING.md` is
already applied to these files, not just described in prose.

---

## 2. Project Objectives, Scope & Plan (from the proposal)

### 2.1 Objectives
1. To develop an LSTM-based time-series model for predicting the next-day
   gold closing price.
2. To present results through an interactive analytical dashboard.

### 2.2 Scope (as originally written)
Predicting the next-day closing price of gold (XAU/USD) using historical
time-series data from Yahoo Finance, 2004–present. Workflow: data
collection → cleaning → feature engineering (moving averages 7/14/30/60-day,
momentum, volatility) → chronological train/test split (2004–2024 train) →
60-day sliding window → model comparison (Linear Regression, ARIMA, Naive
Baseline vs. LSTM) → evaluation via MAE/RMSE → visualization dashboard.

> See **§3 Deviations** below for exactly where the *actual* implementation
> differs from this original scope (it does, in a few specific,
> defensible places).

### 2.3 Methodology Steps (proposal §1.4, for reference / defense prep)
Topic & scope definition → data collection → data exploration → literature
review → proposal preparation & defense → data preprocessing → feature
engineering → data transformation (sliding window) → **Progress 1
presentation** → dataset splitting → initial model building → **Progress 2
presentation** → hyperparameter tuning → model evaluation → model
selection & development → **pre-project presentation** (the one this repo
is being prepared for) → final model testing & evaluation → result
analysis & visualization → documentation & report preparation → final
defense & submission.

### 2.4 Expected Results (proposal §1.6)
1. A functional end-to-end ELT pipeline (Bronze–Silver–Gold) for raw daily
   gold price data.
2. A supervised time-series dataset via sliding window + chronological
   split (leakage-preventing).
3. An LSTM model predicting next-day close, evaluated via MAE/RMSE.
4. An interactive dashboard visualizing trends, predictions, and
   performance.
5. A validated forecasting framework demonstrating LSTM's applicability to
   financial time-series prediction academically.

### 2.5 Place & Equipment (proposal §1.7)
Cloud-based (AWS S3) — no physical lab. Software stack as proposed: Python,
Jupyter/Colab, Pandas/NumPy, TensorFlow/Keras, Matplotlib, Streamlit/Gradio
for the dashboard. Data source: Yahoo Finance (XAU/USD).

---

## 3. Deviations Between the Written Proposal and the Actual Implementation

Being able to explain *why* the real system differs from the plan is part
of demonstrating understanding — these are not oversights to hide, they're
decisions made under real constraints. Every deviation found while
reconciling the proposal text against the current notebooks is listed
here.

| # | Proposal says | Actual implementation | Why |
|---|---|---|---|
| 1 | Pipeline runs on **SageMaker Studio** with an **`ml.m5.2xlarge`** instance (§3.1) | Runs on a classic **SageMaker Notebook Instance** (not Studio/Domains), on **`ml.t3.medium`** initially, then **`ml.t3.xlarge`** | Studio/Unified Studio hit an `AWS::SageMaker::Domain` "Total domains = 0" quota error on the AWS account actually used, so the team switched to the simpler classic Notebook Instance product, which doesn't touch that quota at all. Instance type was a cost-conscious starting choice (`ml.t3.medium`) that turned out too weak for the walk-forward training load (CPU-credit exhaustion — see `docs/TROUBLESHOOTING.md`), so it was upsized to `ml.t3.xlarge`, not `ml.m5.2xlarge` as originally scoped. **Action needed**: update §3.1's text and Figure 3.1 caption if it names Studio/`ml.m5.2xlarge` specifically, or add a footnote explaining the substitution. |
| 2 | Data collection covers **four** timeframes — 1D, 1W, 1M via Yahoo Finance, **plus 4H loaded from a manually pre-downloaded CSV** (2004–2025) because Yahoo Finance only serves 60 days of intraday history via the API (§3.2.2) | Current `01_Data_Collection.ipynb` collects only **1D, 1W, 1M** — no 4H at all, and §3.3's entire dot-date-format-fix / mean-resampling discussion for the 4H file no longer applies to any code path | The 4H timeframe and its associated cleaning logic were removed at some point before this document was written (confirmed absent from both `01_Data_Collection.ipynb`'s `TIMEFRAMES` config and `02_Data_Cleaning.ipynb`'s merge logic, which now only reads/merges 1D/1W/1M). The specific reason for dropping 4H isn't captured in this repo's own history — most likely a scope simplification, since 1W/1M are also collected but *also* unused downstream (only 1D actually feeds feature engineering). **Action needed**: either restore a one-line explanation ("4H was dropped because it added collection complexity without being used downstream, given only the 1D series feeds the model") to §3.2/§3.3, or simplify those sections to stop describing 4H-specific cleaning steps that no longer exist in the code. |
| 3 | Walk-forward retraining uses a **30-day sequence window**, and early stopping monitors **training loss** with patience 5 (§3.6.2) | Walk-forward retraining actually uses **`SEQ_LEN = 60`** (matching the main preprocessing window in §3.5.3, not 30), and `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` against a **chronological 15%-of-window validation split** carved out of each retraining window | The `SEQ_LEN=30` figure in §3.6.2 appears to be left over from an earlier draft — the code has consistently used 60 (matching §3.5.3) everywhere it's actually configured (`04_Preprocessing.ipynb` and `05_LSTM_Training.ipynb` both declare `SEQ_LEN = 60`). Separately, monitoring plain training loss for early stopping provides no real overfitting protection (training loss keeps improving as long as training continues), so the code was fixed to carve out the most recent 15% of each 365-day training window as a validation slice and monitor `val_loss` instead — a **more rigorous** approach than what's currently written, not a shortcut. **Action needed**: correct §3.6.2 to say 60-day window and describe the val_loss/15%-holdout early-stopping mechanism actually used. |
| 4 | §3.7 narrative (baselines, metrics, results discussion) is written and — helpfully — already **matches** the corrected pipeline's numbers, including LSTM's 51.5% directional accuracy | Same, confirmed independently from a fresh run — see §8 below | No action needed here beyond adding the actual numeric table (MAE/RMSE for all four models), which the current proposal text describes narratively but doesn't tabulate. Draft table is in §8.4. |
| 5 | Chapter 4 discussion (§4.2) compares LSTM against "ARIMA, Linear Regression, Decision Tree, and Random Forest" | Only three baselines are actually implemented: **Naive Persistence, AR(5) via Linear Regression, and Linear Regression** on the full feature set. No ARIMA, Decision Tree, or Random Forest model exists in the code. | AR(5) (an autoregressive linear model on 5 lagged closes) is sometimes loosely described elsewhere as "ARIMA-like," but it is not a fitted ARIMA(p,d,q) model — this naming slip was flagged earlier in the project and should be corrected. Decision Tree/Random Forest are named in the Chapter 4 discussion as if they were run, but they are not present in `06_Model_Evaluation.ipynb`. **Action needed**: either implement Decision Tree/Random Forest as additional baselines to match the discussion, or edit §4.2 to only discuss the baselines that were actually run (Naive, AR(5), Linear Regression) plus a general discussion of *why* ARIMA/tree-based methods are expected to underperform (which can stay as literature-grounded discussion without claiming they were empirically run in this project). |
| 6 | §3.4 Feature Engineering / §3.5 dataset sizes: Silver = 5,491 rows (Jun 2004–Apr 2026), Gold = 5,431 rows after dropping 61 rows, train = 4,887 rows, test = 544 rows | Not independently re-verified numerically in this session against a fresh run, but the *code* that produces these (60-row drop for MA60 + 1-row drop for undefined target = 61; split at `2024-12-31`) matches the proposal's description exactly | No known deviation — listed here for completeness since every other numeric claim in the proposal was checked. If the dataset has been re-collected since (new trading days appended), these exact row counts will drift slightly and should be re-printed from a fresh run before finalizing the report. |

---

## 4. AWS Account State (actual, as of now)

- **Account**: owned by a friend of the student. Account ID intentionally
  omitted from this document — this README is public on GitHub, and the
  account belongs to someone outside this project.
- **Login identity used day-to-day**: IAM user `Auto`
  (ARN `arn:aws:iam::<ACCOUNT_ID>:user/Auto`) — **not** the root user.
- **IAM policies currently attached to user `Auto`**:
  - `AmazonS3FullAccess`
  - `AmazonSageMakerFullAccess`
  - `IAMFullAccess` (added so `Auto` can manage its own access keys —
    had to be attached by someone with existing IAM permissions, since
    `Auto` could not grant this to itself)
- **Region**: `ap-southeast-2` (Asia Pacific — Sydney). All S3 and
  SageMaker resources live here.
- **S3 bucket**: `gold-lstm-forecast`
- **Compute actually used**: SageMaker classic **Notebook Instance** — see
  Deviation #1 in §3 for why this differs from the proposal's stated
  SageMaker Studio / `ml.m5.2xlarge`.
  - Instance type history: `ml.t3.medium` → CPU-credit exhaustion mid
    walk-forward training (burstable instance, sustained load) →
    `ml.t3.xlarge` for the successful full run.
  - **The Notebook Instance's own IAM execution role** is a separate
    identity from the console-login user `Auto`. It needed
    `AmazonS3FullAccess` attached directly to itself (SageMaker →
    Notebook instances → instance → "Permissions and encryption" → IAM
    role link → Add permissions).
  - **Cost reminder**: `ml.t3.xlarge` bills hourly while `InService`.
    Always **Stop** it when not actively working.

### IAM permission model — why three separate policies were needed

Three *different* identities can each independently block the pipeline:
1. IAM user `Auto` logging into the AWS Console → needed
   `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`.
2. The SageMaker **Notebook Instance's own execution role** (used
   automatically by boto3/awswrangler inside Jupyter via instance
   metadata) → needed `AmazonS3FullAccess` attached to *that role*.
3. IAM user `Auto` again, needing `IAMFullAccess` to create its own
   **Access Key** for local/programmatic use (the Streamlit dashboard).

If any future AWS `AccessDenied` error appears, check **which of these
three identities** is making the call before attaching more policies.

---

## 5. Data Architecture — Medallion (Bronze / Silver / Gold)

All paths below are exact, taken directly from each notebook's Config cell.

| Layer  | S3 path | Format | Written by |
|--------|---------|--------|------------|
| Bronze | `s3://gold-lstm-forecast/bronze/xauusd_daily/raw/` | CSV | `01_Data_Collection.ipynb` |
| Bronze (log) | `s3://gold-lstm-forecast/bronze/logs/collection_log.csv` | CSV | `01_Data_Collection.ipynb` |
| Silver | `s3://gold-lstm-forecast/silver/xauusd_daily_clean.parquet` | Parquet | `02_Data_Cleaning.ipynb` |
| Gold | `s3://gold-lstm-forecast/gold/xauusd_daily/features/xauusd_features.parquet` | Parquet | `03_Feature_Engineering.ipynb` |
| Gold (model artifacts) | `s3://gold-lstm-forecast/gold/xauusd_daily/features/` (same folder) | mixed | `04_Preprocessing.ipynb`, `05_LSTM_Training.ipynb` |

Bronze layer files: `XAU_1d_data.csv`, `XAU_1w_data.csv`,
`XAU_1Month_data.csv` (see Deviation #2 in §3 — no 4H file, contrary to
the proposal text). Only `1d` actually feeds the modeling pipeline; `1w`
and `1m` are collected and merged into Silver (forward-filled to daily
frequency) but their columns are not currently used by feature engineering
(`03_Feature_Engineering.ipynb` derives every feature from `close` alone).

Gold-layer model-artifacts folder (`gold/xauusd_daily/features/`) contains,
after notebooks 04 and 05 have both run successfully:
```
xauusd_features.parquet   (written by 03 — the feature table itself)
X_train.npy                (written by 04)
X_test.npy                 (written by 04)
y_train.npy                (written by 04)
y_test.npy                 (written by 04)
feature_scaler.pkl         (written by 04 — sklearn StandardScaler, fit on train only)
target_scaler.pkl          (written by 04 — sklearn StandardScaler, fit on train only)
test_dates.csv             (written by 04 — one date per row of y_test)
lstm_model.keras           (written by 05 — final trained Keras model)
wf_pred.npy                (written by 05 — walk-forward predictions, real price scale)
wf_actual.npy              (written by 05 — walk-forward actuals, real price scale)
```
`06_Model_Evaluation.ipynb` reads all of the above but does **not** write
anything new to S3 — it only prints metrics and saves two PNG charts
locally (`06_actual_vs_predicted.png`, `06_metrics_bar.png`). Baseline
model metrics (Naive/AR(5)/Linear Regression) are therefore **not**
persisted anywhere — they're recomputed from the Gold parquet + saved
scalers whenever needed (the Streamlit dashboard does this live).

---

## 6. Pipeline Walkthrough (`01` → `06`) — what, why, problems, fixes

Ticker: `GC=F` (COMEX Gold Futures) via `yfinance`, from `2004-06-01` to
today. CSV fallback path exists if a Yahoo Finance API call fails.

### 01_Data_Collection.ipynb

**What**: fetches 1D/1W/1M OHLCV from `yfinance`, validates each result
(all 6 OHLCV columns present, non-empty, parseable dates), uploads each
timeframe's raw CSV to Bronze, writes a metadata log (filename, row count,
date range, source, upload timestamp) to `bronze/logs/collection_log.csv`,
then reads back `XAU_1d_data.csv` to verify and produces an EDA chart.

**Why Yahoo Finance instead of a static Kaggle dataset**: the original
plan was a downloaded Kaggle CSV. A static file has a fixed end date and
needs manual re-downloading every time newer data is wanted — inconvenient
for a system meant to forecast the *current* next day. `yfinance` fetches
live data on every run, so the dataset always extends to the latest
trading date with no manual step.

**Why validate before uploading**: catching an empty DataFrame, a missing
OHLCV column, or an unparseable date *before* it lands in Bronze prevents
corrupted data from silently propagating into every downstream layer,
where it would be much harder to trace back to its source.

**Problem/risk**: intraday timeframes (like 4H) are only available for the
past 60 days via the Yahoo Finance API — this is why an earlier version of
this notebook loaded 4H from a manually pre-downloaded CSV instead of the
API. **Current state**: 4H has since been dropped entirely (see Deviation
#2, §3) — only 1D/1W/1M are collected now, all via the API with a CSV
fallback if a call fails.

### 02_Data_Cleaning.ipynb

**What**: loads the three Bronze CSVs (1D/1W/1M), normalizes column names,
fixes date parsing, resamples 1W/1M to daily frequency via forward-fill,
merges everything onto the 1D calendar, drops any remaining nulls
(concentrated at the very start of the series, before 1W/1M coverage
begins), sorts chronologically, and writes the merged result to Silver as
Parquet.

**Problem: inconsistent column names across sources.** Each raw file uses
different capitalization/naming for the same logical column (`Date` vs
`date` vs `Datetime`). Pandas treats differently-named columns as entirely
separate, so a naive merge would silently fail to align them. **Fix**: a
normalization pass lowercases and maps every column name to one of six
canonical names (`date, open, high, low, close, volume`) before anything
else happens.

**Problem: different native frequencies.** 1D has one row per trading day;
1W has one row per week; 1M has one row per month. A direct join on `date`
would leave almost every daily row with no matching weekly/monthly value.
**Fix**: 1W and 1M are each resampled to daily frequency with forward-fill
(`resample('1D').ffill()`) *before* merging — appropriate because a
weekly/monthly figure represents an aggregate condition that can reasonably
be treated as constant day-to-day within that period, until the next
data point arrives.

**Problem: missing values after merge.** Even after resampling, some early
dates still have nulls (market-closed days, or the point right before a
timeframe's coverage begins). **Fix**: forward-fill again on the merged
frame, then drop any rows still null (this only affects a handful of rows
at the very start of the series).

**Why chronological sorting matters here specifically**: the LSTM learns
by observing a *sequence* — if rows aren't in date order, the model would
learn from incorrect temporal transitions (e.g., being shown "day 500"
right after "day 3"), producing meaningless learned patterns. Sorting also
guarantees the later chronological train/test split actually assigns
earlier dates to train and later dates to test.

### 03_Feature_Engineering.ipynb

**What**: loads Silver (sorted chronologically), engineers 7 features from
`close` alone, defines the supervised-learning `target`, and writes the
result to Gold as Parquet.

Exact formulas:
- `return` = `close.pct_change()` — percentage change from the previous day.
- `ma7`, `ma14`, `ma30`, `ma60` = `close.rolling(N).mean()` for N in {7,14,30,60}.
- `volatility_7` = `return.rolling(7).std()` — 7-day rolling standard deviation of daily returns.
- `momentum_7` = `close - close.shift(7)` — price change over the past week.
- `target` = `close.shift(-1)` — next day's closing price (the label).

**Why a raw price alone isn't enough**: a single price value tells the
model nothing about whether the market is trending, decelerating, or
becoming more volatile. The engineered features give the model that
context directly, rather than forcing it to infer trend/momentum/volatility
purely from a raw price sequence.

**Why `return` specifically addresses non-stationarity**: gold's raw price
rose from roughly USD 400 (2004) to over USD 5,300 (2026) — a strong
persistent upward trend that makes the raw price series non-stationary.
Expressing changes as a percentage (`return`) removes most of this trend,
producing a series that fluctuates around zero regardless of the absolute
price level at the time.

**Why moving averages at four different windows**: 7/14-day averages
reflect short-term trend; 30/60-day averages reflect medium/long-term
trend. The relationship between the current price and each of these
averages signals whether a trend is strengthening or reversing —
information a single day's price cannot convey by itself.

**Why this stage runs before the train/test split (and why that's still
leakage-safe)**: every one of these features is a strictly backward-looking
function of already-past values (a moving average never uses a future
price). Computing them across the full chronological series before
splitting is therefore *not* leakage — the leakage risk only appears later,
at scaling (see §6, notebook 04), where fitting on future-inclusive
statistics genuinely would look ahead.

**Row removal**: 61 rows dropped via `dropna()` — 60 from the start
(insufficient history for `ma60`), 1 from the end (the final day has no
"next day" to define `target`). Gold dataset: 5,431 rows.

### 04_Preprocessing.ipynb — the leakage-critical stage

Config: `SEQ_LEN = 60`, `FEATURES = ['close','return','ma7','ma14','ma30','ma60','volatility_7','momentum_7']`, `TARGET = 'target'`.

1. **Chronological split** at `split_date = '2024-12-31'` — train = 4,887
   rows (2004–2024), test = 544 rows (2025–Apr 2026). Never random/shuffled
   — a random split would let the model train on days *after* a test day
   it's evaluated on, silently inflating apparent accuracy (data leakage).
2. **Scaling fit on train only.** `StandardScaler().fit(train[...])`, then
   `.transform()` on both train and test. This is the single most
   important anti-leakage decision in the project: fitting on the combined
   or full dataset would leak test-set statistics into training. Two
   scalers (`feature_scaler`, `target_scaler`) are persisted to S3 so the
   exact same transform can be reapplied later (dashboard inference,
   re-evaluation) without ever re-fitting on new data.
   - **Why `StandardScaler` and not `MinMaxScaler`**: gold prices in the
     test period reach USD 5,318 — far above the training set's maximum
     of roughly USD 2,700. `MinMaxScaler` maps values to a fixed [0, 1]
     range based on the *training* min/max, so any test value above the
     training max (which is exactly what happens here, dramatically) would
     be scaled to a value well outside [0, 1] — destabilizing the model at
     exactly the period that matters most for evaluation. `StandardScaler`
     (subtract mean, divide by std) has no fixed output range and degrades
     far more gracefully when test values fall outside the training
     distribution.
3. **Sliding-window sequence construction.** `create_sequences()` builds
   windows of the most recent `SEQ_LEN` days (**inclusive of day i
   itself**, i.e. `X[i-59:i+1]`) to predict `y[i]` (next-day close from day
   i). This "inclusive" indexing was a deliberately fixed off-by-one: an
   earlier version excluded day i from its own prediction window,
   discarding the freshest, most informative observation for no reason.
   Resulting shapes: `X_train (4827, 60, 8)`, `X_test (484, 60, 8)`.
4. Saves `X_train/X_test/y_train/y_test.npy`, both scalers, and
   `test_dates.csv` to S3. `test_dates` is aligned to start at the same row
   as `y_test[0]` (`test['date'].iloc[SEQ_LEN-1:]`, not `iloc[SEQ_LEN:]`)
   — a second off-by-one fix, since `y_test[0]`'s window ends at test row
   `SEQ_LEN-1`, not `SEQ_LEN`.

### 05_LSTM_Training.ipynb — model architecture & walk-forward retraining

**Architecture**: `LSTM(64, return_sequences=True) → Dropout(0.1) →
LSTM(32) → Dropout(0.1) → Dense(1)`. Adam optimizer.

**Why LSTM specifically**: its memory cells and gating mechanisms (input/
forget/output gates) let it selectively retain or discard information
across time steps — capturing long-term dependencies that a non-sequential
model (e.g. plain Linear Regression on a flattened window) cannot
represent directly.

**Why Huber loss instead of MSE**: gold prices occasionally spike several
percent in a single day around macroeconomic news. Under MSE, such large
errors are squared and dominate the loss gradient, destabilizing training.
Huber loss applies a quadratic penalty for small errors and a linear
(not squared) penalty for large ones, keeping training stable in the
presence of occasional large surprises.

**Problem: a single static train-once model performs poorly on this test
period.** The test period (2025–Apr 2026) reaches gold prices the model
never saw during training (above ~USD 2,700) — a genuine
out-of-distribution problem, not a bug. A model trained once on 2004–2024
data has no way to know what price behavior looks like above the highest
price it ever observed.

**Fix: walk-forward retraining.** Config: `SEQ_LEN = 60`, `RETRAIN_EVERY =
7`. Rather than training once, the model retrains from scratch every 7
simulated trading days, each time using only the most recent 365 days as
its training window, then predicts the next 7 days, then rolls the window
forward (the newly-realized 7 days join the window, the oldest 7 drop out)
and retrains again. This lets the model gradually absorb new, higher price
levels as they actually occur, rather than being frozen at whatever it
learned once in 2024. Each retraining step: up to 20 epochs, batch size 64.

**Fix: early stopping actually guards against overfitting.** Each 365-day
training window is further split — the most recent 15% becomes a
chronological validation slice, and `EarlyStopping(monitor='val_loss',
patience=5, restore_best_weights=True)` watches that validation loss,
not training loss (see Deviation #3, §3, for why this differs from an
earlier proposal draft that described monitoring training loss with no
validation split at all — monitoring plain training loss provides no
actual regularization signal, since it keeps improving as long as
training continues).

**Constraint that shapes how this stage must be operated**: there is **no
checkpointing** across the walk-forward loop — a kernel restart or crash
mid-run means starting over from step 1. It is also by far the most
computationally expensive stage in the whole pipeline (this is why the
instance was upsized from `ml.t3.medium` to `ml.t3.xlarge` — see
`docs/TROUBLESHOOTING.md` for the CPU-credit-exhaustion story). Never
restart the kernel on a hunch that training has stalled; first rule out
burstable-instance throttling.

Saves `lstm_model.keras`, `wf_pred.npy`, `wf_actual.npy` (already
inverse-transformed to real price scale) to S3.

### 06_Model_Evaluation.ipynb — baselines, metrics, comparison

Three baselines, each trained on the same `train_df`
(`split_date = '2024-12-31'`) and evaluated on the same `test_df` as the
LSTM, so all four models are compared on an identical window:
- **Naive Persistence** — predicts tomorrow's close = today's close. The
  floor any useful model should beat.
- **AR(5)** — Linear Regression on the previous 5 closing prices.
- **Linear Regression** — on the full 8-feature set (scaled the same way
  as the LSTM's inputs).

**Why Directional Accuracy matters as much as MAE/RMSE**: MAE/RMSE measure
error *magnitude* in price terms, but for someone actually deciding
whether to buy or sell, the practically relevant question is *direction* —
will tomorrow's price be higher or lower? A model can have excellent MAE
purely because prices don't move much day-to-day on average, while still
being no better than a coin flip at calling direction. Directional
Accuracy exposes that distinction; MAE/RMSE alone would hide it.

**Known fixed bug (AR(5) alignment)**: the AR(5) baseline's evaluation
window was previously offset one trading day earlier than every other
model's, from two related off-by-one errors in how its `Xar`/`yar` arrays
were sliced (full index derivation is in the notebook's Cell 3 comments).
Fixed without touching the leakage-free train-only fit — only the
evaluation-window alignment changed.

---

## 7. Known Risks Addressed (for the presentation / report)

- **Data leakage from random splitting** → fixed by strict chronological
  split at a fixed date.
- **Data leakage from scaling before split** → fixed by fitting both
  scalers on `train` only, never on `test` or the full dataset.
- **NaN values from rolling-window features** → dropped rather than
  imputed; imputing a moving average with, say, 0 or the series mean would
  inject a value the model could never have actually observed at that
  point in time (imputation-as-leakage).
- **Off-by-one errors in sliding-window / evaluation-window indexing** —
  three separate instances found and fixed (notebook 04's sequence
  inclusion + `test_dates` alignment, notebook 06's AR(5) window). Easy to
  introduce, easy to miss — none of them crash anything, they just
  silently shift every downstream metric by one trading day. Always
  sanity-check window alignment by print-inspecting the first and last
  `(X, y, date)` triple of any new sequence array before trusting its
  metrics.
- **Model smooths over sharp moves** — not a bug, a genuine limitation of
  any model trained predominantly on calmer historical volatility;
  documented and interpreted (§8) rather than hidden.
- **Burstable-instance CPU throttling mistaken for a crash** — mitigated
  by moving to a non-burstable/larger instance type for the full
  walk-forward run.
- **Out-of-distribution test prices** (test period reaches price levels
  never seen in training) — addressed structurally by walk-forward
  retraining (§6) and by choosing `StandardScaler` over `MinMaxScaler`
  (§6, notebook 04) specifically because it degrades more gracefully on
  out-of-range values.
- **Early stopping with no real regularization signal** (an earlier draft
  monitored training loss) — fixed by carving out a chronological 15%
  validation slice per retraining window and monitoring `val_loss` (§6,
  notebook 05).

---

## 8. Model Evaluation — Results, Interpretation, §3.7 Write-up

### 8.1 Results

| Model | MAE (USD) | RMSE (USD) | Directional Accuracy |
|---|---|---|---|
| Naive Persistence | ~51.1 | ~74.3 | ~44.7% |
| AR(5) | ~51.0 | ~74.2 | ~44.1% |
| Linear Regression | ~51.2 | ~74.1 | ~43.9% |
| **LSTM Walk-Forward** | ~641.0 | ~776.6 | **~51.5%** |

### 8.2 Interpretation

The LSTM has by far the worst absolute-error metrics (MAE/RMSE) but the
best directional accuracy, and is the only model to beat 50% (random
guess) on direction. This is a trade-off, not a contradiction: 2025–2026
saw an unusually sharp, sustained gold price rally, and "predict roughly
today's price again" is a low-error strategy precisely when today's price
is close to tomorrow's — mechanically true during a smooth trend,
regardless of whether the trend is correctly anticipated. None of the
three baselines exceed 45% directional accuracy — they are *worse than
chance* at calling direction, despite their low MAE.

The LSTM's higher absolute error reflects a genuine, documented limitation
rather than a modeling failure: trained across a long history that
includes calmer volatility regimes, it produces smoother predictions than
the actual series and systematically underestimates the magnitude of
sudden upward spikes during the 2025–2026 rally. It correctly anticipates
*that* the price will keep rising more often than the baselines or random
chance, but understates *how much*. This distinction — magnitude error
versus directional skill — is the central finding: absolute-error metrics
alone would incorrectly rank the naive baseline as "best," when it in fact
has no real predictive skill in the sense that matters for a forecasting
system.

### 8.3 What the proposal already gets right (don't rewrite this part)

The proposal's own §3.7 narrative text (already written in the current
`.docx`/PDF) already states this same interpretation correctly, including
the exact 51.5% directional-accuracy figure and the out-of-distribution
price-surge explanation. **The only thing actually missing from the
current proposal is the numeric MAE/RMSE table** (§3.7.3 describes the
comparison narratively and via Figure 3.10, but never tabulates the
numbers) — that's the one addition needed, not a rewrite.

### 8.4 Draft table + caption text to insert into §3.7.3

Ready to paste in as a table right after the existing §3.7.3 narrative
paragraph, and to use verbatim as the Figure 3.10 caption if useful:

> **Table 3.2** — Model comparison on the chronological test period
> (2025-01-01 to 2026-04-20).
>
> | Model | MAE (USD) | RMSE (USD) | Directional Accuracy |
> |---|---|---|---|
> | Naive Persistence | 51.1 | 74.3 | 44.7% |
> | AR(5) | 51.0 | 74.2 | 44.1% |
> | Linear Regression | 51.2 | 74.1 | 43.9% |
> | **LSTM (walk-forward)** | 641.0 | 776.6 | **51.5%** |

**[TODO before finalizing]**: an R² value (~0.87) appears in an earlier
draft/context of this project; it has not been recomputed against the
corrected pipeline (after the AR(5) evaluation-window fix and the
early-stopping/val_loss fix). Recompute R² directly from `wf_pred.npy` /
`wf_actual.npy` before citing any R² figure — do not reuse the old number.

---

## 9. Literature Review Summary (condensed from Chapter 2)

For quick reference / defense prep — the proposal's Chapter 2 argument in
brief:

- **ARIMA** and other classical statistical models rely on linear
  assumptions and require stationary data; they perform adequately under
  stable conditions but struggle with the nonlinear, dynamic behavior
  financial series like gold actually exhibit [2],[10].
- **Classical ML** (Linear Regression, Decision Trees, Random Forest, SVR)
  models nonlinear relationships and multiple features better than ARIMA,
  but depends heavily on hand-crafted feature engineering and doesn't
  inherently capture temporal order — it treats rows as independent unless
  time-based features are explicitly constructed [8],[11].
- **LSTM** (Hochreiter & Schmidhuber, 1997 [1]) was designed specifically
  to address the vanishing-gradient limitation of earlier recurrent
  networks, using memory cells and gates to selectively retain/discard
  information across long sequences — well suited to financial series
  where both short-term fluctuation and long-term trend matter. Prior work
  reports LSTM and hybrid CNN-LSTM/ConvLSTM architectures outperforming
  classical statistical and ML baselines on gold-price and other financial
  forecasting tasks [3],[4],[5],[12].
- **Identified gaps** this project targets: (a) many studies compare models
  without a reproducible, structured *pipeline* connecting data engineering
  to modeling; (b) simpler architectures like a single LSTM may be
  competitive when paired with proper feature engineering and leakage-safe
  preprocessing, without needing a more complex hybrid architecture; (c)
  most studies emphasize accuracy alone, under-emphasizing practical
  pipeline/visualization concerns for end users — which this project's
  ELT + dashboard structure directly addresses.

---

## 10. Streamlit Dashboard (local, ad-hoc — not a persistent server)

**STATUS: still open to further iteration**, but now a complete, polished,
English-only pass covering all four components the proposal's §3.8
specifies — not just a starting point. Keep improving it based on
feedback; nothing below should be treated as frozen.

**Design decision (this part IS fixed)**: the dashboard runs **locally on
the student's Mac**, on demand (screenshots now, live demos to the advisor
later), reading data straight from S3. It is deliberately **not** deployed
to always-on AWS compute — the real AWS cost risk in this project is
compute left running continuously, not data reads or occasional inference;
S3 storage/reads at this data volume are effectively free, and a single
local inference call costs nothing.

Per the proposal's own §3.8 (Visualization and Reporting), Streamlit was
chosen because it integrates directly with the existing Python pipeline
and can read straight from S3 without any export/format-conversion step,
keeping the whole system on one technology stack. All four components the
proposal describes are now implemented: (1) historical XAU/USD price chart
2004–present with 1Y/5Y/All range-selector buttons, (2) actual-vs-predicted
test-set comparison, (3) a dedicated 30-day rolling volatility chart, and
(4) model performance comparison (MAE/RMSE/Directional Accuracy, LSTM vs.
the three baselines) — plus a next-day forecast KPI, a prediction-error
panel, and a feature explorer that go beyond the original spec.

### Layout — organized into four tabs
- **Overview** — KPI header (latest close, next-day forecast + delta, LSTM
  directional accuracy, LSTM MAE) sits above the tabs, always visible;
  the Overview tab itself holds the full historical price chart
  (2004–present) with 1Y/5Y/All quick-zoom buttons and a range slider,
  now paired with a **Volume sub-panel** directly beneath it (shared
  x-axis, same combo-chart layout as the Model Evaluation error panel) —
  the standard price+volume view any reader will expect on a financial
  dashboard's landing tab. This was originally placed only inside
  Methodology as a "why it's excluded as a feature" diagnostic; that
  framing answers a different question (is volume a model input) than
  what the Overview tab is for (what does the raw data look like), so
  volume now appears in both places, each doing its own job — Overview
  shows it, Methodology explains why it isn't used.
- **Model Evaluation** — the Actual-vs-Predicted walk-forward chart, now
  paired with a **prediction-error sub-panel** directly beneath it (actual
  minus predicted, colored red for under-prediction / blue for
  over-prediction using the project's validated diverging pair) — this
  makes the thesis's central finding ("the model lags the 2025–2026 surge")
  visible as a pattern of red bars rather than something asserted only in
  text. Below that, the three Model Comparison bar charts (MAE/RMSE/
  Directional Accuracy) plus the numeric table and a written interpretation
  paragraph.
- **Volatility & Features** — a dedicated 30-day rolling volatility chart
  (computed on the fly for this view; the model itself trains on the
  narrower `volatility_7` feature, kept separate to match the proposal's
  own §3.8 wording), a **Daily Return Distribution histogram** (mean/std
  computed live, visualizing the non-stationarity fix described in
  Methodology — the same shape notebook 01's collection-time EDA chart
  first showed, now reproduced from the Gold-layer `return` feature rather
  than a re-derived copy), plus the feature explorer (any of the 7
  engineered features over time).
- **Methodology** — the "why," not just the "what," now opening with **why
  trading volume isn't a model feature**: a live-computed diagnostic (%
  of trading days with zero/missing volume) plus a chart, making the
  case visually rather than asserting it — this is the same volume
  sparsity notebook 01's collection-time EDA chart already showed; see
  §6 (notebook 03) discussion for why every feature is derived from
  `close` alone. Followed by chronological split, train-only scaler
  fitting, the 60-day window, walk-forward retraining, and why
  Directional Accuracy is reported alongside MAE/RMSE — written as full
  explanatory paragraphs rather than a bullet-point footer, so it reads
  well if the advisor clicks into it directly.
- A **sidebar** (always visible, no tab click needed) shows dataset range,
  split date, and model hyperparameters as quick reference during a live
  demo.

### Color palette — CVD-validated, not eyeballed
The first version reused the notebook 06 matplotlib colors (gray/salmon/
blue/green) for visual continuity, but running them through the `dataviz`
skill's validator (`validate_palette.js`) failed two checks — see
`docs/TROUBLESHOOTING.md` Phase 4 for the exact failure and the fix. The
current palette is the skill's validated 8-hue categorical order, assigned
by fixed identity and never re-cycled: **LSTM Walk-Forward = blue,
Actual = orange, Naive Persistence = aqua, AR(5) = yellow, Linear
Regression = magenta**. The prediction-error panel uses the validated
diverging blue/red pair. All combinations actually used together were
re-validated (adjacent-pair CVD ΔE and normal-vision floor both pass in
light and dark) before shipping.

### What `app.py` does
- Connects to S3 bucket `gold-lstm-forecast` (region `ap-southeast-2`) via
  `awswrangler`/`boto3`, using whatever AWS credentials are configured
  locally — no credentials are hardcoded in the file.
- Loads the Gold parquet, both scalers, `wf_pred.npy`/`wf_actual.npy`,
  `test_dates.csv`, and `lstm_model.keras` from
  `s3://gold-lstm-forecast/gold/xauusd_daily/features/`.
- **Recomputes** Naive Persistence, AR(5), and Linear Regression baselines
  live in the app (same logic as notebook 06's Cell 3), since notebook 06
  never persists these to S3.
- Loads LSTM Walk-Forward metrics directly from the saved
  `wf_pred.npy`/`wf_actual.npy`.
- Computes a **next-day forecast**: takes the most recent `SEQ_LEN=60`
  rows of the Gold feature table, scales with the persisted
  `feature_scaler`, runs `lstm_model.predict()`, inverse-transforms with
  `target_scaler`.
- Computes the **volume data-quality diagnostic** (% of rows with
  zero/missing `volume`) live from whatever is in the loaded Gold table,
  rather than a hardcoded figure — the exact percentage shifts slightly
  as more trading days get collected, so it's always recomputed, never
  quoted from a specific past run.
- All UI text is in English (an earlier draft mixed Thai captions in; the
  dashboard is meant to be shown to the advisor/committee, so it's fully
  English now, matching the proposal's own language).

### Local Mac setup (one-time)

**1. Create an AWS Access Key for IAM user `Auto`** (now that `Auto` has
`IAMFullAccess`, `Auto` can do this itself):
AWS Console → IAM → Users → `Auto` → **Security credentials** tab →
**Access keys** → **Create access key** → use case **Command Line
Interface (CLI)** → confirm → **Create access key**. Save the Access Key
ID and Secret immediately (the secret is shown only once).

**2. Configure credentials on the Mac**:
```bash
mkdir -p ~/.aws

cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id = <ACCESS_KEY_ID>
aws_secret_access_key = <SECRET_ACCESS_KEY>
EOF

cat > ~/.aws/config << 'EOF'
[default]
region = ap-southeast-2
EOF
```
boto3/awswrangler read these automatically — no keys go in `app.py`.

**3. Install dependencies** (Python 3.10+ recommended):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt`:
```
streamlit>=1.32
awswrangler>=3.7
boto3>=1.34
pandas>=2.0
numpy>=1.24,<2.0
plotly>=5.18
tensorflow>=2.15
scikit-learn>=1.3
```
Apple Silicon (M1/M2/M3/M4): current `tensorflow` on PyPI supports `arm64`
directly via `pip install tensorflow`. If that fails, fall back to
`tensorflow-macos` + `tensorflow-metal`.

**4. Run**:
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Close the process when done — nothing on
AWS keeps running as a result (S3 is storage-only and billed negligibly at
this data volume; no compute cost from running the dashboard).

---

## 11. Daily Automation Pipeline (Lambda + EventBridge Scheduler)

**Status: designed and code-complete in `automation/`, not yet deployed or
run against the live AWS account** — it was built in an environment with no
network path to AWS, so "ready to deploy" is the accurate claim, not
"currently running." Full design rationale, every file, and the deployment
steps are in **[`automation/README.md`](automation/README.md)**; this
section is the summary worth presenting.

**What it adds**: an EventBridge Scheduler rule fires once a day, at 08:00
Thailand time (01:00 UTC), and invokes a Lambda function that checks
whether a new GC=F daily close has posted, and if so, recomputes the
engineered features and produces a fresh next-day forecast — without
waiting for the next manual notebook run. 08:00 ICT isn't an arbitrary
round number: COMEX Gold halts trading daily at 5:00pm ET and fixes its
official settlement at 1:30pm ET, both of which land in the small hours of
the Thailand morning, so 08:00 ICT clears them with a multi-hour buffer —
see `automation/README.md` §2 for the full time-zone reasoning.

**Why daily, not hourly** (an earlier design used `rate(1 hour)`): the
dataset is daily-granularity — one new observation per trading day — so
the schedule was changed to match that exactly instead of polling 24x more
often than the data can change. This also makes **what it deliberately
does *not* add — automatic retraining** — an easier case to make: retraining
the walk-forward LSTM on this same daily schedule would mean re-running the
single most expensive stage in the pipeline (the same walk-forward loop
that exhausted CPU credit on `ml.t3.medium` during development — see
`docs/TROUBLESHOOTING.md`) every single day, regardless of whether one new
day of data actually justifies a full retrain — wasted compute with
minimal information gain, and a real cost/stability risk on a friend's AWS
account, not a hypothetical one. Automating *only* the part of the
pipeline that actually benefits from this tighter, data-matched scheduling
(inference) and leaving retraining on a slower, data-driven cadence —
mirroring the `RETRAIN_EVERY = 7` logic notebook 05 already uses — is the
deliberate design decision here, not a scope cut.

**Why the write path is isolated.** The Lambda never writes to the
canonical `bronze/`, `silver/`, or `gold/.../xauusd_features.parquet`
files the manual notebooks and dashboard depend on — an unattended daily
job and an occasional manual notebook re-run are two independent writers,
and letting both touch the same files risks a race or a malformed column
neither writer fully owns. Instead it reads Silver read-only and writes
only to its own append-only `bronze/streaming/` log plus a
`gold/.../predictions/latest_forecast.json` cache. A bug in this function
can corrupt only its own log, never the training data or the trained
model.

**A real bug this work surfaced.** `03_Feature_Engineering.ipynb`'s
`dropna()` drops the most recent row's *features* along with its
(always-undefined-at-collection-time) `target`, which means the persisted
Gold table's last row is always one trading day behind the true latest
close — so `dashboard/app.py`'s current "Next-Day Forecast" KPI is actually
predicting a value already knowable from already-collected data, not a
genuine unknown future. The Lambda avoids this structurally by building its
forecast window from Silver's un-shifted close series instead of Gold. The
same fix has **not** yet been ported back into `dashboard/app.py` — flagged
as a known, understood, deliberately-deferred follow-up rather than
silently patched; see `automation/README.md` §5 for the full explanation.

**Cost**: Lambda's free tier covers roughly 720 invocations/month at this
function's size by several orders of magnitude — expected real cost is
effectively $0, the same conclusion already reached for the rest of this
project's AWS usage.

**Deploying it** (from the student's Mac or the SageMaker terminal — needs
real AWS network access, which is why it wasn't done from the environment
that wrote this code): `python automation/convert_and_export.py` once,
then `automation/deploy.sh --grant-permissions`. Full steps, the IAM
least-privilege policy applied to the Lambda's own execution role
(deliberately tighter than Auto's own `*FullAccess` policies — see §4 —
because this role runs unattended and continuously, which is exactly where
that discipline matters most), and teardown instructions are all in
`automation/README.md`.

---

## 12. Pending / Not Yet Done

1. **Update the thesis proposal Word document** (source `.docx`, not
   accessible from this session — PDF-only):
   - Fix the Figure 3.9/3.10 caption swap noted earlier in the project.
   - Insert the numeric Table 3.2 from §8.4 above (the narrative text is
     already correct — this is purely an addition).
   - Correct §3.1 to reflect the actual compute used (Deviation #1, §3),
     or add a footnote explaining the substitution.
   - Correct §3.2/§3.3 to stop describing 4H-specific collection/cleaning
     steps that no longer exist in the code (Deviation #2, §3).
   - Correct §3.6.2's sequence-window figure (30-day → 60-day) and describe
     the actual `val_loss`/15%-holdout early-stopping mechanism
     (Deviation #3, §3).
   - Reconcile §4.2's discussion of ARIMA/Decision Tree/Random Forest with
     what was actually implemented (Deviation #5, §3) — either implement
     those baselines or adjust the text to describe them as
     literature-grounded discussion rather than empirical results from
     this project.
   - Fix the "ARIMA" terminology slip elsewhere in Chapter 4 (AR(5) is a
     linear-regression autoregressive model, not a fitted ARIMA(p,d,q)).
2. **Recompute R²** for the corrected LSTM walk-forward predictions (§8.4
   TODO) before citing any R² figure — the ~0.87 figure in circulation
   predates the AR(5)/early-stopping fixes and hasn't been re-verified.
3. **Re-verify the exact row counts** in §2.2/§6 (5,491 Silver rows, 5,431
   Gold rows, 4,887/544 train/test) against a fresh run if the dataset has
   grown since (new trading days keep appending via `yfinance`).
4. **Verify the pre-existing EC2/RDS recurring-charge concern** flagged
   earlier in the project is actually resolved — check AWS Billing / Cost
   Explorer for anything still accruing charges outside the SageMaker
   Notebook Instance and S3.
5. **Stop the SageMaker Notebook Instance** (`ml.t3.xlarge`) whenever not
   actively in use — it bills hourly while `InService`.
6. Optional/future: persist the baseline model metrics from notebook 06 to
   S3 (currently only recomputed on the fly) if a saved-artifact record is
   ever needed outside a live session.
7. **Keep iterating on the Streamlit dashboard** (§10) — current layout is
   a first pass; extend toward the proposal's full §3.8 spec (historical
   price chart, dedicated volatility chart) and beyond, based on feedback.
8. **Actually deploy and verify the daily automation** (§11) against the
   live AWS account — run `convert_and_export.py` + `deploy.sh`, confirm a
   real invocation writes a correct `latest_forecast.json`, then let it run
   for at least one real trading-day rollover before demoing it as "live"
   rather than "designed and ready."
9. **Port the Silver-based (not Gold-based) forecast-window fix** found
   while building §11 into `dashboard/app.py`'s `forecast_next_day()`, so
   the dashboard's own Next-Day Forecast KPI stops being one day stale.
10. Optional/future, once §11 is live and trusted: automate *absorbing*
    `bronze/streaming/xauusd_daily_incremental.csv` back into the canonical
    Bronze/Silver/Gold files, and add a second, much-less-frequent
    (e.g. weekly) scheduled job that re-runs full walk-forward retraining
    — see `automation/README.md` §9 for why this is scoped out for now.

---

## 13. References (from the proposal)

[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural
Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.
[2] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time
Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2015.
[3] A. Mohapatra, A. McGinity, and A. John, "Gold price prediction using
machine learning: A comparative study of LSTM and traditional models,"
*Procedia Computer Science*, vol. 218, pp. 1385–1396, 2023.
[4] T. Kim and H. Kim, "Forecasting stock prices with a feature fusion
LSTM-CNN model using different representations of the same data," *PLOS
ONE*, vol. 14, no. 2, p. e0212320, Feb. 2019.
[5] P. Lenz, N. Andres, and M. Kiel, "LSTM-based forecasting models for
financial time-series prediction," in *Proc. Int. Conf. Financial
Technology and Data Science*, Clausius Press, 2024, pp. 112–118.
[6] F. Chollet, *Deep Learning with Python*, 2nd ed. Manning Publications,
2021.
[7] M. Abadi et al., "TensorFlow: A system for large-scale machine
learning," in *Proc. 12th USENIX Symp. OSDI*, 2016, pp. 265–283.
[8] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J.
Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
[9] W. McKinney, "Data structures for statistical computing in Python," in
*Proc. 9th Python Sci. Conf.*, 2010, pp. 51–56.
[10] A. Primananda and S. Isa, "Gold price prediction using ARIMA and
LSTM models," *Sinkron: Jurnal dan Penelitian Teknik Informatika*, vol. 7,
no. 3, pp. 1258–1262, 2023.
[11] H. Kilimci, "Ensemble regression-based gold price (XAU/USD)
prediction," *Journal of Emerging Computer Technologies (JECT)*, vol. 2,
no. 2, pp. 1–10, 2022.
[12] M. A. Alkhodair et al., "Gold price prediction by a CNN-BiLSTM model
along with automatic parameter tuning," *PLOS ONE*, vol. 19, no. 2, p.
e0298426, 2024.

---

## 14. Putting This on GitHub

This folder is a ready-to-init repo. From inside `gold-lstm-forecast/`:

```bash
git init
git add .
git commit -m "Initial commit: full pipeline (01-06), dashboard, docs"
```

Then create an empty repository on GitHub (via the website, or `gh repo
create` if the GitHub CLI is installed and authenticated), and:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

Notes:
- The `.gitignore` already excludes `venv/`, `.aws/` credentials, and the
  pulled `*.npy`/`*.pkl`/`*.keras`/`*.parquet` model/data artifacts — those
  live in S3 (bucket `gold-lstm-forecast`), not in git. Do **not** remove
  those exclusions and commit real AWS credentials or large binary model
  files into the repo.
- If the repo will be public, double-check no AWS Access Key ID/Secret,
  account ID, or bucket name that should stay private is left in any
  notebook cell's printed output before pushing (the notebooks currently
  only print the bucket *name*, not credentials — but always re-check
  outputs after any new run, since printed debug output is an easy way for
  a secret to end up committed by accident).
- The thesis proposal `.docx`/PDF is intentionally **not** included in this
  repo (it's the write-up, not the code/data artifact this repo tracks).
  Add it under a `docs/proposal/` folder if version-controlling it
  alongside the code is wanted.
