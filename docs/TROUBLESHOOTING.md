# Development Log & Troubleshooting Journal

A chronological record of every real issue hit while building this project on
AWS, why it happened, and exactly how it was fixed. Kept as its own document
because this is the part a professor asking "what problems did you run into
and how did you solve them" actually wants to see — and the part most likely
to get lost if it only lives in chat history.

Phases below roughly match three working sessions (early setup, full
pipeline debugging, final IAM/dashboard polish).

---

## Phase 1 — AWS Account Setup

### Issue: SageMaker Studio / Domain quota error
Attempting to use SageMaker Studio (Unified Studio / DataZone-based) hit an
`AWS::SageMaker::Domain` "Total domains = 0" quota error on the account.

**Fix**: abandoned the Studio/Domain path entirely and used classic
**SageMaker Notebook Instances** instead — a plain EC2-backed Jupyter
environment that doesn't touch the Domain quota at all. This is the
approach the whole project is built on; there is no Studio/Domain usage
anywhere in the final pipeline.

### Issue: `AccessDeniedException` on `datazone:ListDomains` for IAM user "Auto"
When logging into the AWS Console as the non-root IAM user `Auto` (who has
no attached policies by default) and opening the SageMaker service page,
the console tried to list Unified Studio domains in the background and
threw `AccessDeniedException: ... datazone:ListDomains ...`.

**Diagnosis**: this error is a red herring — it has nothing to do with
Notebook Instances specifically. It fires because the SageMaker landing
page always tries to enumerate Studio domains regardless of which SageMaker
feature you're about to use.

**Fix**: attached `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`
directly to IAM user `Auto` (IAM → Users → Auto → Permissions → Add
permissions → Attach policies directly). The `datazone` error itself
became irrelevant once the actual Notebook Instances workflow was used.

### Issue: Notebook Instance not appearing after creation
After creating a Notebook Instance, it didn't show up in the console list
for the person checking.

**Diagnosis checklist used**: region mismatch (console set to a different
region than the resource was created in), instance still in `Pending`
status, stale IAM permission cache, or looking at the wrong console page
(SageMaker Studio's instance list vs. the classic "Notebook instances"
page under SageMaker).

**Resolution**: resolved on the user's end (most likely a region or page
mismatch) — later screenshots showed the instance running normally.

---

## Phase 2 — Full Pipeline Debugging (the bulk of the real errors)

### Issue: `AccessDenied` on `PutObject` inside Jupyter (`wr.s3.to_csv`)
Once notebooks started running inside the Notebook Instance, `awswrangler`
S3 writes failed with `AccessDenied` on `PutObject` — even though IAM user
`Auto` already had `AmazonS3FullAccess`.

**Root cause (two separate things, both true at once)**:
1. The notebooks' `S3_BUCKET` config constant was set to a placeholder
   bucket name (`mfu-gold-lstm-forecast-2026`) that didn't match the real
   bucket that had actually been created (`gold-lstm-forecast`).
2. Separately — and this is the part that's easy to miss — **the
   Notebook Instance itself runs under its own IAM execution role**,
   completely distinct from the IAM user (`Auto`) used to log into the
   AWS Console. boto3/awswrangler inside Jupyter authenticate via that
   role's instance metadata, not via the console login. Giving `Auto`
   more permissions does nothing for code running inside the notebook.

**Fix**:
- Bulk `sed` replace across all 6 notebooks: every `mfu-gold-lstm-forecast-2026`
  → `gold-lstm-forecast` (the real bucket name, confirmed from the S3
  console screenshot).
- Found the Notebook Instance's own role (SageMaker → Notebook instances →
  click the instance → "Permissions and encryption" → IAM role link) and
  attached `AmazonS3FullAccess` **to that role directly** — a separate
  action from anything done to IAM user `Auto`.
- Kernel restart + full re-run after both fixes.

This one cost the most back-and-forth in the whole project because the
symptom (`AccessDenied on PutObject`) looked identical both before and
after the bucket-name fix, since the *actual* blocker (the execution
role's missing policy) hadn't been touched yet. Lesson: an `AccessDenied`
error from inside a notebook should always be checked against the
**Notebook Instance's own role**, not just the console-login user.

### Issue: `ModuleNotFoundError: No module named 'matplotlib'`
Hit at the EDA cell in `01_Data_Collection.ipynb`. Contrary to the usual
expectation that SageMaker's standard `conda_python3` kernel ships with
matplotlib preinstalled, this environment's image didn't have it.

**Fix**: immediate — `!pip install matplotlib --quiet` in a new cell above
the failing one, to unblock the running session. Systemic — added
`matplotlib` to the Cell 0 pip-install line in every notebook that
actually imports it (`01`, `03`, `06`).

**Follow-up hardening**: since the environment already surprised us once
on an assumed-preinstalled package, proactively added `scikit-learn` to
the Cell 0 installs in `04` and `06` as well (both use
`sklearn.preprocessing.StandardScaler` / `sklearn.linear_model.LinearRegression`)
rather than waiting to hit the same class of error a second time.

### Issue: `NameError: name 'df' is not defined`
Hit in `04_Preprocessing.ipynb` at a cell with execution count `[2]` — a
tell that cells had been run out of order (Cell 0 imports and Cell 2 data
load were skipped, and a later cell was run directly).

**Fix**: not a code bug — a workflow habit fix. Always run notebooks top
to bottom from Cell 0, or use Jupyter's **Run → Run All Cells** menu
action instead of clicking individual cells out of sequence.

### Issue: severe slowdown during LSTM walk-forward training
On `ml.t3.medium`, the walk-forward retraining loop in
`05_LSTM_Training.ipynb` ran normally for the first ~35 minutes (through
step 47), then produced no new step output for 7+ minutes despite the
kernel still showing as busy/running (not crashed, not erroring).

**Diagnosis**: `ml.t3.medium` is a *burstable* instance type — it earns
CPU credits over time and can spend them for short bursts above its
baseline, but a sustained ~35+ minute CPU-bound TensorFlow training job
exhausts that credit balance, after which the instance is throttled back
to a much lower sustained CPU rate. This produces exactly this symptom:
no crash, no error, just a dramatic and confusing slowdown.

**Constraint that shaped the fix**: the walk-forward loop has **no
checkpointing** — it retrains a full model from scratch every 7 simulated
days across the entire test period. Restarting the kernel mid-run means
losing all progress and starting over from step 1. This is why the advice
during the actual slowdown was explicitly **not** to restart the kernel,
only to wait or, as a last resort, move to a larger/non-burstable
instance type.

**Fix**: moved to `ml.t3.xlarge` (larger, and with a proportionally larger
baseline/burst CPU allowance) for the full successful run. If this
project is extended later, adding checkpointing to the walk-forward loop
(saving intermediate model/results state every N steps) would remove this
fragility entirely — currently flagged as a nice-to-have, not done.

### Issue: AR(5) baseline evaluation window misaligned by one trading day
Found during a detailed index-by-index re-audit of `06_Model_Evaluation.ipynb`
(not caught in an earlier pass) — the AR(5) baseline's predictions were
being compared against actuals that were shifted one trading day earlier
than every other model's (Naive, Linear Regression, LSTM) evaluation
window, throughout the entire test period.

**Root cause**: two related off-by-one errors in how the `Xar`/`yar`
arrays were constructed and sliced:
1. `Xar`'s `column_stack` used `close_all[i:len(close_all)-N+i]`, which
   excludes the dataset's very last possible window — so `Xar` was always
   one row short of reaching the final target.
2. The prediction slice started at `Xar[sp:]` / `yar[sp:]` instead of
   `Xar[sp+1:]` / `yar[sp+1:]` — `yar[sp]` is actually the boundary value
   (test day 0's own close, not a forecast), since `Xar[sp]`'s last input
   day is the final training day. Every other model (Naive/LR/LSTM)
   starts its forecast one day *into* the test period, so AR(5) needs to
   start there too.

Both bugs pushed AR(5)'s window one day earlier than intended,
consistently, for the whole test set — the AR(5) metrics were internally
self-consistent (not `NaN`/broken), just silently answering a slightly
different, easier-by-coincidence question than the other three models.

**Fix**: changed the column_stack upper bound to
`close_all[i:len(close_all)-N+i+1]` and the slice starts to `sp+1`. See
the inline code comments in `06_Model_Evaluation.ipynb` Cell 3 for the
full index derivation — kept in the notebook itself since this is exactly
the kind of bug worth being able to re-derive from scratch later.

### (Related, proactive) Dormant off-by-one in `04_Preprocessing.ipynb`
While auditing the AR(5) bug above, noticed `test_dates = test['date'].iloc[SEQ_LEN:]`
in notebook 04 used the wrong offset relative to `create_sequences()`'s
own indexing (`y_test[0]` corresponds to test row `SEQ_LEN-1`, not
`SEQ_LEN`). This particular instance was **currently harmless** — its only
downstream use in notebook 06 is a right-aligned trim
(`test_dates[-len(wf_pred):]`) that happens to not care about the
off-by-one — but it was fixed anyway (`iloc[SEQ_LEN-1:]`) for consistency
and to avoid it becoming a real bug the next time `test_dates` is used
differently.

---

## Phase 3 — IAM Refinement & Local Dashboard Setup

### Issue: `AccessDenied` on `iam:ListUsers` for IAM user "Auto"
While trying to view the IAM Users list in the console (to create a
personal Access Key for local/programmatic use), user `Auto` — who only
had `AmazonS3FullAccess` and `AmazonSageMakerFullAccess` — hit
`AccessDenied` on the `iam:ListUsers` action.

**Fix**: had the account owner (who has existing IAM permissions) attach
`IAMFullAccess` to user `Auto`. This is a case where the user genuinely
could not fix this themselves — granting IAM permissions to an identity
that doesn't have any IAM permissions yet is a chicken-and-egg problem
that has to be broken from outside (root, or another already-privileged
identity).

**Consolidated final IAM state for user `Auto`** (all three attached,
no further policies expected to be needed for this project):
- `AmazonS3FullAccess`
- `AmazonSageMakerFullAccess`
- `IAMFullAccess`

### Task: local Streamlit dashboard, connected to S3, no persistent AWS compute
Design question raised: should the dashboard run as an always-on service
(e.g., on SageMaker or EC2), or can "using the cloud" be satisfied more
cheaply? Resolved by recognizing that **the cost risk in this project is
compute left running continuously, not data reads or occasional
inference** — S3 storage/reads at this data volume are effectively free,
and a single local inference call costs nothing.

**Decision**: run the Streamlit dashboard **locally on the student's
Mac**, on demand, reading data from S3 via a personal IAM Access Key
(created using the `IAMFullAccess` permission above). See the main
`README.md` §10 for exact setup steps. Nothing about this required an
AWS-side fix — it's a design decision, not a bug — but it's recorded here
because it was reached only after ruling out the more expensive default
(always-on hosted compute).

---

## Phase 4 — Running the Dashboard Locally

### Issue: `streamlit run app.py` → "File does not exist: app.py"
Ran from the wrong working directory — `app.py` lives in `dashboard/`, not
at the repo root, and a fresh Terminal window always starts back at the
home directory regardless of where a previous window's `cd` left off.

**Fix**: `cd` into `dashboard/` first (or run `streamlit run dashboard/app.py`
from the repo root). The most reliable way to get the right path without
retyping it is dragging the `dashboard` folder from Finder onto the
Terminal window right after typing `cd ` — Terminal fills in the full
absolute path automatically.

### Issue: `ModuleNotFoundError: No module named 'awswrangler'` after Streamlit actually launches
Streamlit itself started fine and served the page — the failure happened
*inside* `app.py`, at `import awswrangler as wr`.

**Root cause**: the `pip install -r requirements.txt` step and the
`streamlit run` step used two different Python interpreters — e.g.
packages were installed with a virtual environment active in one Terminal
window, then `streamlit` was invoked in a fresh window/session where that
environment was never activated, so it fell back to a system Python that
never got the packages.

**Fix**: pin both commands to the exact same interpreter with `python3 -m`,
which sidesteps whatever `streamlit`/`pip` resolve to on `$PATH`:
```bash
cd dashboard
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```
If using a virtual environment instead, remember it must be re-activated
(`source venv/bin/activate`) in **every new Terminal window** before
running Streamlit — activation does not persist across windows/sessions,
only within the one shell it was run in. A `(venv)` prefix on the prompt
confirms it's active.

### Follow-up: dashboard color palette failed CVD-safety validation
The original bar/line colors (gray, salmon-pink, blue, green — carried
over from the notebook 06 matplotlib figures for visual continuity) were
run through the project's color-blindness/contrast validator
(`dataviz` skill's `validate_palette.js`) while rewriting the dashboard
for the final presentation, and failed: the gray had zero chroma (reads as
a validator error even though "neutral baseline" was the intent), and the
salmon-pink/green pairing sits close enough together for red-green color
blindness (protanopia/deuteranopia) to be a real risk in a 4-series chart.

**Fix**: switched to the skill's validated 8-hue categorical order (a blue/
orange/aqua/yellow/magenta/green/violet/red sequence chosen so adjacent
hues clear a minimum perceptual-distance threshold for both normal and
color-blind vision). Assigned by fixed identity — LSTM Walk-Forward = blue,
Actual = orange, Naive = aqua, AR(5) = yellow, Linear Regression = magenta
— never re-cycled across charts, and re-validated with the script before
shipping (see `dashboard/app.py`'s `COLORS` dict for the exact hex values
and the reasoning comment above it).
