# Next-Day Gold Price Forecasting with LSTM — Project README

Undergraduate Computer Engineering senior project, Mae Fah Luang University.
This file is the single source of truth for continuing the work — proposal
context, AWS setup, pipeline internals reconciled against the actual code,
every real issue hit and fixed, current evaluation results, and everything
still pending. It merges the written thesis proposal (`Next_day_Gold_
Prices_Forcasting_With_LSTM.pdf`) with what the pipeline *actually* does
today, and calls out every place the two disagree. Read this fully before
making changes. If you are picking the project up, start with the
**[Handoff](#handoff-where-things-stand-2026-10-27)** block below.

For the full chronological story of every AWS/code error hit and how it
was fixed (useful for "what problems did you encounter" in the defense),
see **[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)**.

> **Update (September 2026): the evaluation results in §8.1–§8.4 are superseded.** On review, the original
> pipeline's very large LSTM error (MAE ≈ \$640) was a scale problem, not a market effect, and its "best
> directional accuracy" did not survive comparison with the trivial "always up" rule. Read the
> **[Project Journey](#project-journey-what-we-tried-what-broke-what-we-learned)** first; the corrected numbers are in §8.0.

---

## Handoff: where things stand (2026-10-27)

Read this block first if you are picking the project up (a new contributor or another coding agent); everything below it is the
detailed record.

**Links.** Code: https://github.com/aphiwat-data/lstm_dashboard (public, branch `main`). Live dashboard:
https://lstmdashboard-v6sere4oaulpge5dj9bq5p.streamlit.app (Streamlit Community Cloud redeploys from `main` within about a minute
of a push). The thesis documents (Word/PDF, including Chapter 3) are kept outside this repo.

**The result.** On the 449-day walk-forward test window (2025-01-09 to 2026-10-23) the LSTM is level with a no-change forecast, not
better:

| Check | Result |
|---|---|
| Price error | MAE \$46.83 vs \$46.63 for "tomorrow = today" (Diebold-Mariano p = 0.38); 1.15% of the price |
| Fit on returns | R² -0.014 (a constant zero-return forecast scores -0.003); IC +0.006 (p = 0.90) |
| Direction (secondary) | 51.9% vs always-up 55.2% (p = 0.155), AUC 0.463; the model says "up" on about 88% of days |
| Extra features | Macro, NFP, FOMC+CPI and COT (daily and hourly) are all null; best daily variant FOMC+CPI 57.5%, AUC 0.520, p = 0.37 |
| The ~70% target | Not reachable on this evidence; ~70% sign accuracy on daily returns would need a predicted-vs-actual correlation near 0.59 |

So far the write-up is a well-instrumented negative result. **The project is open to new approaches**: the old proposal and its ~70% target are superseded
and the advisor allows trying anything. The feature searches in Phase 9 were null (evidence, not a prohibition); judge new ideas with the
regression-first metrics and the same causal protocol.

**Metric policy (advisor, 2026-10-27).** Directional accuracy is a classification metric, not the yardstick for a price forecast. Lead
with price error (MAE, RMSE, % error, R², IC) against the "tomorrow = today" baseline and keep up/down accuracy only as a labelled
secondary check (it is how the proposal phrased the ~70% target, and it exposes the model's majority-class behaviour). The dashboard
already does this, and so does Chapter 3 (price accuracy first, direction as a secondary table).

**Regression-first metric rewrite (2026-10-27/29, done).** `daily/regression_metrics.py` (tests in `daily/test_regression_metrics.py`,
called by `walk_forward.py`) computes from `wf_predictions.csv`: MAE, RMSE, MAPE, relative MAE and Diebold-Mariano p (lags 0 and 24; LSTM
p = 0.38-0.41 either way, Ridge 0.023-0.052, so borderline), R², out-of-sample R² vs. the rolling mean (LSTM -0.011), IC, and interval
coverage, width and interval score from errors known before each forecast; output is `regression_metrics.json` in every results folder.
*Finding:* the old "90% likely range" was calibrated on the whole test window, so "covers 90% of test days" was true by construction; scored
honestly it covered 85.9%. The shipped band is now Gaussian with sigma = EWMA(lambda 0.94, fixed in advance) of past errors: 89.5% coverage
(n = 389), as wide as a no-change band ($223 vs. $221), so the LSTM does not narrow uncertainty. The dashboard, README §8.0 and Chapter 3
(Table 3.2 price accuracy, 3.3 R²/IC, 3.4 secondary direction check, 3.5 feature sets, 3.6 horizons; interval paragraph added) now follow it;
the Chapter 3 backup before this change is `Chapter3_before_regression_first_backup.docx`.

**Interval-method comparison (2026-10-29, `daily/interval_experiment.py`, `daily/lstm_quantile.py`, results in `daily/results/interval_comparison.json`).**
Same 389 scored test days, every band uses only earlier information; target coverage 90%. EWMA Gaussian (the shipped band) covers 89.5%
(width \$223, interval score 317, Kupiec p = 0.73); GARCH(1,1) Gaussian 85.6% (\$203, 322, p = 0.006) and with filtered historical simulation
85.3% (p = 0.004); empirical quantiles of all past errors 85.9% (p = 0.010); an LSTM trained with pinball loss to output the 5%/95%
quantiles directly (same walk-forward protocol, 2 seeds) only 63.5% (\$129, score 457). No method beats EWMA, so the dashboard band is unchanged. Caveat: the
EWMA decay was fixed in advance, but the alternatives were compared on the same test window, so 89.5% is slightly optimistic.

**Multi-day price forecasts (2026-10-29, `daily/walk_forward_horizon.py`, results in `daily/results/horizon_h5/` and `horizon_h20/`).** Same
protocol as the daily run (retrain every 7 days on the latest 365 labelled days, LSTM(32), 3 seeds), forecasting the price h trading days ahead
and judged against "price in h days = price today". Labels overlap, so p-values use n_eff = n / h and a HAC Diebold-Mariano variance.

| Horizon (days) | n / n_eff | No-change MAE (\$) | LSTM MAE (\$) | LSTM vs no-change | DM p | R² out-of-sample | IC (p) |
|---|---|---|---|---|---|---|---|
| 5 | 444 / 88 | 109.19 | 109.50 | -0.3% | 0.43 | -0.040 | -0.018 (0.87) |
| 20 | 429 / 21 | 219.40 | 216.95 | +1.1% | 0.59 | +0.034 | +0.085 (0.72) |

No significant skill at either horizon. At 20 days the LSTM is 1.1% better in MAE with a positive out-of-sample R² (+0.034), but with only
21 independent windows nothing is distinguishable from zero (DM p = 0.59, IC p = 0.72); Ridge and HistGB are worse than no-change at both
horizons. Training rows stop h days before each block because a label is only known h days later.

**Does VIX improve the 90% interval? (2026-10-29, `daily/interval_vix.py`, `daily/results/interval_vix.json`).** Scale model sigma = exp(a + b log EWMA + c log VIX),
fitted by Gaussian quasi-ML on the last 1000 days known at each refit (every 7 days), same 389 scored days. Shipped EWMA band: coverage 89.5%, score 317.
Refit without VIX: 87.1% (score 322); with same-day VIX: 86.1% (322); with yesterday's VIX: 86.4% (322); VIX alone: 77.1% (359). VIX gets a positive
coefficient (c about 0.2) but adds nothing out of sample (score 322 with and without it; no difference from EWMA is significant), and the
Gaussian fit under-covers because returns have fat tails. The shipped EWMA band is unchanged.

**Evaluation protocol (daily).** `daily/walk_forward.py` retrains every 7 trading days on the trailing 365 days: input window 20 days,
LSTM(32) with two heads (next-day log-return and direction), 3 seeds averaged, scalers re-fit inside every window; the forecast price
is `close * exp(predicted log-return)`. Baselines on identical rows: rolling drift (= always up), ridge, HistGB regressor and
classifier, momentum rule. Statistics: binomial test against the always-up rate, Wilson 95% CIs, AUC, Diebold-Mariano against
no-change, Spearman IC, R² on returns, and a 90% interval from the LSTM residuals. Features come from
`daily/build_daily_dataset.py` and are causal and scale-free by construction (`daily/test_no_lookahead_daily.py` checks it).

**How to run things** (the S3 silver table and the dashboard's original-pipeline views need AWS credentials; pass `--silver <local
parquet>` to keep the daily scripts off S3):

```bash
python3 -m streamlit run dashboard/app.py          # dashboard on http://localhost:8501 (repo root)
cd daily
python3 test_no_lookahead_daily.py                 # causality tests
python3 build_daily_dataset.py --out-dir data/processed
python3 walk_forward.py --data data/processed/xauusd_d1_features.parquet --latest data/processed/xauusd_d1_latest_features.parquet --out-dir results
python3 horizon_always_up.py                       # the Phase 8 horizon table
```

The feature-search folders are the same walk-forward run on a differently built dataset (one `--out-dir` per variant):

| Results folder (`daily/results/`) | `build_daily_dataset.py` flags |
|---|---|
| `with_macro/` | `--macro` |
| `with_macro_calendar/` | `--macro --calendar --calendar-events nfp` |
| `with_macro_calendar_full/` | `--macro --calendar` |
| `with_cot/` | `--macro --calendar --cot` |

Hourly: `hourly/run_full_pipeline.sh` (needs Node for `dukascopy-node`; run `npm install` in `hourly/` first), then
`hourly/train_models.py` (usage in its docstring).

**Refreshing the results** (§12 item 12). (1) Extend the silver data: the S3 table is still at 2026-09-21, and a local parquet with the
same columns can be passed to `--silver`. (2) Delete or repoint `daily/data/macro_cache.parquet` and `cot_cache.parquet` first:
`fetch_macro` and `fetch_cot` return a cache as-is without checking its end date, so a stale cache silently forward-fills old values.
(3) Re-run the main walk-forward **and all four variants on the same rows**, keep the previous outputs in a dated
`daily/results/snapshot_*` folder, and update the README numbers, Chapter 3 Tables 3.2-3.5 and the dashboard together (the
dashboard's feature-search table reads the `with_*` folders by itself).

**Working agreements and traps.**
- The repo is public: never commit AWS keys, the AWS account ID or any credentials (Streamlit Cloud keeps them in the app's Secrets).
  The student names and IDs in the dashboard sidebar are public on purpose (the owner's request).
- Dashboard code: native Streamlit elements only (no `unsafe_allow_html`), `width="stretch"`, escape dollar signs in markdown as
  `\$`, and use `st.container(horizontal=True)` instead of `st.columns` for anything that must survive the open sidebar. Cloud
  installs a newer Streamlit than local development used (1.66 vs 1.56), so check every layout change at 1280, 1024, 900, 760 and
  390 px and on the live URL after the push (§10, lessons).
- The owner pushes small commits straight to `main`. Every push redeploys the public app, so a broken `app.py` is public at once.
- Report status (2026-10-29): Chapter 3 is final. `Report_Abstract_Ch1_Ch4.docx` (next to it, outside the repo) holds the revised Abstract, Chapter 1 sections 1.2/1.3/1.6,
  Chapter 4 (conclusion and discussion) and the additional references. Chapter 2 (literature review) is the proposal's original text and has not been revised.
- Chapter 3 tables were updated programmatically from the results JSONs with python-docx (each table cell is a single run: set
  `run.text` to keep the formatting). Keep a backup next to the file and compare every number with the README.

---

## Project Journey: what we tried, what broke, what we learned

The advisor asked how the work actually went, including the wrong turns. This section is that honest timeline.
Earlier AWS/notebook errors are logged in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md); later phases are
also visible in the commit history.

### At a glance

| Question | Answer today |
|---|---|
| Does the LSTM beat a trivial baseline at next-day direction? | **No.** Daily: 51.9% vs "always up" 55.2% (test 2025-01-09 to 2026-10-23, n=449). |
| Does hourly data help (1, 4 and 24 bars ahead)? | **No.** Accuracy about 50–54% vs "always up" 51.5% / 52.9% / 55.1%; AUC 0.50–0.52. |
| Is the advisor's ~70% directional-accuracy expectation reachable? | **Not on this evidence, not by combining models, and not by adding macro/cross-asset inputs.** Every model tried (ridge, gradient boosting, LSTM/GRU) sits at AUC 0.50-0.52 on price history alone; adding DXY/silver/S&P 500/VIX/yields/TIP (Phase 9) still lands at AUC ~0.50. Ensembling or feeding in more inputs cannot manufacture a signal that none of them has. "Always up" alone reaches 55% at 1 day, 62.0% at 20 days and 75.6% at 60 days on the test period (Phase 8) purely because gold trended up, so a ~70% figure needs a stated horizon, baseline and coverage before it means anything. Worth discussing with the advisor (where does 70% come from, and how is it measured?). |
| Do other horizons, interval methods or VIX change that? | **No.** 5 and 20 days ahead: no significant gain over no-change (p = 0.43, 0.59). The shipped 90% band (EWMA, 89.5% causal coverage) beat GARCH (85.6%), an LSTM quantile model (63.5%) and a VIX model (86.1-86.4%) (Phase 10). |
| Is the price forecast usable? | It matches a no-change forecast (daily MAE \$46.83 vs \$46.63). Read it as "about today's close, within the interval". |
| What was the big bug? | A price-level target and price-level inputs with train-only scalers: once gold left the training range in 2025-26, MAE reached ≈ \$640. Fixed with a return target and scale-free features. |

### Phase 1: build the pipeline (§5-§7)
Medallion data lake on S3 (Bronze/Silver/Gold), six notebooks on SageMaker, a walk-forward LSTM predicting the next-day
close *price*, three baselines, a Streamlit dashboard and a designed (not yet deployed) daily Lambda. The problems met on
the way (IAM permissions, instance credits, leakage, evaluation-window mistakes) are in `docs/TROUBLESHOOTING.md`.

### Phase 2: publish and deploy (September 2026)
- Pushed the repo to GitHub (public). The AWS account belongs to a friend, so its account ID is deliberately not
  written in any tracked file.
- Streamlit Community Cloud first failed with "Error installing requirements": the platform used Python 3.14, which has
  no TensorFlow wheel. Adding `runtime.txt` and `.python-version` had no effect in our case; what worked was deleting the
  app and re-creating it with Python 3.11 selected in the deploy dialog.
- Next came `NoCredentialsError` (the dashboard reads S3): fixed by adding AWS credentials in the app's Secrets, never in git.
- A slip worth remembering: the IAM user `Auto` was deleted (the user page's Delete button removes the whole user, not one
  access key) and re-created, and a secret access key is shown only once. Read destructive buttons twice.

### Phase 3: dashboard extensions
Added, without removing anything: a model-freshness indicator, an empirical 90% forecast range, CSV downloads, a
date-range metrics picker and a panel comparing with the Lambda's daily forecast. Lesson from the picker: a 4-day range
showed 100% directional accuracy, which is only three comparisons, i.e. small-sample luck, not leakage.

### Phase 4: attempt to deploy the automation (still open)
`automation/deploy.sh` attached three managed policies to `Auto` (Lambda, ECR and EventBridge Scheduler full access) and
created the ECR repository `gold-lstm-forecast-daily-refresh`, then the Docker build failed with I/O errors because the
Mac's disk was 98% full (249 MB free); after freeing space Docker Desktop's engine still did not become healthy after a
restart. The Lambda and schedule were never created. The policies and the ECR repository still exist, so either finish the deployment or clean them up (§12).

### Phase 5: advisor feedback: hourly data and a ~70% target (hourly/)
- Data: Dukascopy XAU/USD hourly bars, 2004 to 2026 (free, spot price). Two data traps: the downloader silently saved
  *partial* files when the feed answered HTTP 429 (rate limit), which we caught by comparing row counts and replaced with a
  verified monthly downloader plus coverage checks in the builder; and about a third of the rows were market-closed bars
  with zero volume (36% in the sample month we inspected), which are dropped.
- Pipeline: features use only past bars, labels look 1, 4 or 24 bars ahead, chronological split with an embargo. Tests edit
  future prices and assert that past features do not change, and a mutation check shows the tests fail when a look-ahead
  feature is injected.
- Rules fixed before running: compare with the "always up" rate (gold trends up, so it is 51.5% / 52.9% / 55.1% at 1 / 4 /
  24 bars, not 50%); confidence intervals use n / horizon because neighbouring labels overlap; a label-shuffle control.
- Models: drift, momentum rule, ridge, gradient boosting (regressor and classifier) and a two-head GRU. Result (test 2025+, bid prices):

| Horizon | Always up | Best model accuracy | AUC range | n_eff |
|---|---|---|---|---|
| 1 bar | 51.5% | 51.4% (gru_ensemble) | 0.517–0.518 | 10,197 |
| 4 bars | 52.9% | 51.5% (hist_gb_reg) | 0.495–0.500 | 2,548 |
| 24 bars | 55.1% | 53.7% (hist_gb_reg) | 0.497–0.520 | 424 |

### Phase 6: chasing an apparent 60% (bid-ask artifact)
Gradient boosting reached 56-60% on its most confident 5-10% of predictions, concentrated in the hours around the daily
market break. The spread widens 2-3x there (about 2.4 bp normally, 4.8-5.3 bp at 20-21 UTC) and a bid-only price falls as
it widens, so the "signal" is not tradable. Downloading ask prices and using mid = (bid + ask) / 2: the top-5% edge over
"always up" went from +6.7 pt (bid) to +0.2 pt (mid) on the test split, while validation was mixed, so we report it as
"largely an artifact", not proven. Re-running every model on mid prices gave the same conclusion:

| Horizon | Always up | Best model accuracy | AUC range | n_eff |
|---|---|---|---|---|
| 1 bar | 51.7% | 51.6% (hist_gb_reg) | 0.514–0.516 | 9,780 |
| 4 bars | 53.2% | 51.1% (hist_gb_reg) | 0.495–0.499 | 2,444 |
| 24 bars | 55.8% | 54.2% (hist_gb_clf) | 0.510–0.520 | 406 |

We also had to retract our own earlier statement that this edge "replicates on validation and test". (Four months of ask data
are missing because of rate limiting, so bid-vs-mid comparisons use the bars present in both feeds.)

**Addendum: hourly cross-asset features, mirroring Phase 9's daily test.** The infrastructure for this was
already written (`AUX_TAGS` in `hourly/build_hourly_dataset.py`) but never exercised — no raw data had ever
been downloaded. Added silver, EUR/USD and S&P 500 (Dukascopy codes `xagusd`/`eurusd`/`usa500idxusd`, already
hinted at in the downloader's own usage examples) plus a US Dollar Index proxy (`dollaridxusd`, newly added
to `AUX_TAGS`). Two real constraints, unlike anything hit on the daily side: Dukascopy's S&P 500 series only
starts 2011-09 and its Dollar Index only starts 2017-12 (both well after gold's 2004 start), and no VIX or
bond-yield equivalent exists on Dukascopy at all — those two are simply not obtainable this way. Full
download (bid+ask XAUUSD, bid-only for the aux instruments, 2004-present, mid-price series rebuilt from
bid+ask) took about 40 minutes.

Result, same 1/4/24-bar horizons as Phase 5, mid prices: **null, matching Phase 9's daily conclusion almost
exactly.** Comparing IC (Spearman) with vs. without the aux features at 1 bar barely moves — HistGB regressor
0.0246→0.0244 (p 0.015→0.016), GRU ensemble 0.0279→0.0275 (p 0.006→0.007) — and no model's direction accuracy
clears always-up at any horizon (all p > 0.05 for outperforming; several score significantly *below*
always-up instead, the same pattern seen throughout). Full results: `hourly/results/with_aux_mid/`.

**Separate observation, not caused by the aux features (present already in the pre-existing 1-bar baseline
committed before this addendum), worth flagging rather than acting on:** at the 1-bar horizon specifically,
HistGB regressor and the GRU both show an IC that clears p < 0.05 (unlike anything seen on daily data across
every variant in Phase 8/9), while direction accuracy still does not beat always-up (p = 0.38-0.76). This
does not reproduce at 4 or 24 bars. Given how many models/horizons/feature-sets have now been tested across
this whole investigation, a handful of p-values in the 0.006-0.04 range are within what multiple testing
alone would produce by chance — this is noted as a curiosity for a future, dedicated look (with a proper
multiple-comparisons correction), not reported as a finding.

### Phase 7: fixing the daily pipeline properly (daily/)
- The earlier "fixed" claim for the daily MAE was wrong: it only held for the last three weeks of the test period; over the
  full period the LSTM's MAE was still ≈ \$639 against ≈ \$51 for trivial baselines.
- **Root cause:** notebooks 03-05 predict `close.shift(-1)` (a price level) from price-level inputs (close, MA7/14/30/60, USD
  momentum), scaled with statistics from the training period. In 2025-26 gold traded far above that range, so inputs and
  targets left what the scalers and the model had seen. Walk-forward retraining adapted the weights but not the fixed scalers.
- **Data audit:** 6.2% of daily rows had inconsistent OHLC (almost all in 2004-2011) and the Silver `_1w`/`_1m` columns hold
  the period's final values, i.e. future information on mid-period days (the original model did not use them).
- **Fix:** every feature is a return, ratio or oscillator computed from `close`; the target is the next-day log-return
  converted back to a price; scalers are re-fit in every walk-forward window. A test multiplies all prices by a constant and
  asserts no feature changes.
- **Also wrong before:** the old "directional accuracy" compared day-to-day changes of the predictions with those of the
  actual prices; the standard definition compares the sign of the predicted return with the sign of the actual return.
- Result (walk-forward, retrain every 7 days on 365 days, test 2025-01-09 to 2026-10-23, n=449; the earlier 424-day run is kept in
  `daily/results/snapshot_2026-09-18/`):

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | MAE vs no-change (%) | DM p |
|---|---|---|---|---|---|
| No-change (naive) | 46.63 | 68.94 | 1.14 | – | – |
| Rolling drift | 46.36 | 69.07 | 1.14 | -0.6 | 0.941 |
| Ridge | 47.87 | 70.35 | 1.17 | +2.7 | 0.023 |
| HistGB regressor | 50.68 | 75.05 | 1.24 | +8.7 | <0.001 |
| **LSTM (walk-forward)** | 46.83 | 69.45 | 1.15 | +0.4 | 0.385 |

Secondary, classification-style check (direction), all six models:

| Model | Direction accuracy | vs always-up (pt) | p vs always-up | AUC |
|---|---|---|---|---|
| Rolling drift (= always up) | 55.2% | +0.0 | 1.000 | – |
| Ridge | 49.0% | -6.2 | 0.009 | – |
| HistGB regressor | 50.3% | -4.9 | 0.041 | – |
| HistGB classifier | 52.6% | -2.7 | 0.255 | 0.498 |
| **LSTM (walk-forward)** | 51.9% | -3.3 | 0.155 | 0.463 |
| Momentum rule | 45.2% | -10.0 | 0.000 | – |

  The LSTM's price error is at the no-change level (Diebold-Mariano p = 0.38) and its direction accuracy
  (51.9%, 95% CI 47.3-56.5%) is not distinguishable from
  always-up. Six static-training variants (window 20 or 60 days, 1- and 5-day horizons, training from 2012) agree; see `daily/results/`.
- The dashboard now leads with this corrected evaluation; the original views are kept in an expander for the record. While
  checking the page in a browser we also found that pairs of dollar signs in markdown text are rendered as LaTeX (garbled
  captions), a bug no automated test had caught.

**Addendum: regression-native metrics (R², IC), not just direction/MAE.** Directional accuracy is a
classification-style view of what is fundamentally a regression problem, so both R² (coefficient of
determination) and the Information Coefficient (Spearman rank correlation between predicted and actual
return, `ic_spearman`/`ic_p` — already computed by `daily/walk_forward.py`, just not previously surfaced in
this doc) were checked on the same identical test period:

| Model | R² (returns) | IC (Spearman) | IC p-value |
|---|---|---|---|
| Predict-zero-return baseline | -0.003 | – | – |
| Rolling drift | -0.003 | -0.027 | 0.561 |
| Ridge | -0.045 | -0.103 | 0.029 |
| HistGB regressor | -0.186 | -0.034 | 0.473 |
| **LSTM** | **-0.014** | **0.006** | **0.896** |

Every model's R² is negative (R² < 0 means doing worse than simply predicting the test-period mean return), and the
three learned models are also below the predict-zero baseline; only the rolling drift, which is essentially a constant, is
level with it. No IC points the useful way: the LSTM's is +0.006 (p = 0.90), and Ridge's is -0.103 (p = 0.029), nominally
significant but with the *wrong sign* (its predictions were weakly inversely related to the outcome), which with four models
tested would not survive a Bonferroni correction (threshold 0.0125) and could only be exploited with hindsight. HistGB
regressor's R² is the worst by far (-0.186), consistent with Phase 9's suspicion that adding degrees of freedom to these
models mostly adds overfitting, not skill. This confirms the direction-accuracy finding from an entirely different metric
family: it isn't an artifact of choosing accuracy/AUC as the yardstick.

R² is now computed by the code, not by hand: `price_report()` in `hourly/train_models.py` (shared by `daily/walk_forward.py`)
writes `r2_returns` per model and `naive_r2_returns` (the predict-zero baseline) to every results JSON, and the dashboard shows it.

**Refresh (2026-10-27).** The daily pipeline was re-run on newly collected data: 25 more trading days (through 2026-10-26;
S3 silver itself was not modified, new GC=F bars were appended locally after checking that 22 overlapping days matched the
silver table exactly), so the test window grew from 424 to 449 days (2025-01-09 to 2026-10-23). **Every daily number in
this README, the dashboard and the thesis tables is now from the 449-day window, including all four feature-search
variants (`with_macro*`, `with_cot`), which were re-run on the same rows as the main run.** The previous 424-day outputs
(main run and all four variants) are kept in `daily/results/snapshot_2026-09-18/`. The conclusion did not change: LSTM
direction accuracy 51.9% vs. always-up 55.2% (p = 0.155, AUC 0.463), MAE \$46.83 vs. \$46.63 for a no-change forecast
(Diebold-Mariano p = 0.38), R² -0.014 (predict-zero -0.003), IC +0.006 (p = 0.90); the feature-search LSTM AUCs moved by
+0.004 to +0.011 and kept their order. The static-training variants (`model_results_lstm_w*`) were not re-run.

### Phase 8: should models be combined to reach 70% accuracy?

Asked directly after Phase 7. Answer: no combination of what we have gets there, and 70% itself needs to be
pinned down before it is a meaningful target.

- **Ensembling does not create signal.** Every model tried across the daily and hourly work (ridge, HistGB
  regressor/classifier, LSTM/GRU) independently lands at AUC 0.50-0.52. Blending them can reduce variance
  between runs, not add predictive power that none of the inputs has.
- **"Always up" alone climbs with the horizon, since gold trended up over the sample:**

| Horizon (trading days) | Always-up, TEST 2025-01-09..2026-10-23 | Always-up, all years 2004-2026 | Test n_eff |
|---|---|---|---|
| 1 | 55.1% | 52.7% | 448 |
| 5 | 58.8% | 55.9% | 88 |
| 20 | 62.0% | 56.3% | 21 |
| 60 | 75.6% | 62.9% | 6 |
| 120 | 71.1% | 68.9% | 2 |

  (Produced by `daily/horizon_always_up.py`: the share of test days whose h-day-ahead return is positive, using only price pairs
  inside the test window, so n = N - h and n_eff = n // h.)

  A reported "76% accuracy at 60 days" needs no model at all, and at that horizon the test period only
  contains about 6 non-overlapping windows, too few to conclude anything either way.
- **Three ways a ~70% figure shows up that do not mean the model has skill:** (1) a long horizon in a trending
  market, as above; (2) reporting accuracy only on the model's most confident predictions without stating the
  coverage (Phase 6's 56-60% figure, later traced to a spread artifact); (3) a leak or an evaluation mistake —
  the likeliest explanation for a ~70% figure seen elsewhere on daily-or-shorter gold data.
- **What might move AUC, untested here:** inputs outside price history — economic-calendar events (FOMC, CPI,
  NFP), the US dollar index, bond yields, COT positioning — and hourly cross-asset features (§12). Expected
  effect size is small; there is no basis in what we have tried to expect it to reach 70%.

### Phase 9: testing cross-asset/macro features (daily) — still null

Phase 8 named the untested lever ("inputs outside price history... expected effect size is small; no basis
to expect it to reach 70%") and left it open. Tested it directly: added six cross-asset/macro series to the
daily pipeline via `yfinance` — US Dollar Index (`DX-Y.NYB`), silver (`SI=F`), S&P 500 (`^GSPC`), VIX
(`^VIX`), the 10-year Treasury yield (`^TNX`), and the TIP ETF (`TIP`, an inverse real-yield proxy) — each
turned into 18 new return/change/z-score features (never raw levels, same convention as the price-only
features), as-of aligned onto gold's own trading calendar so a macro market's holiday can only look
backward, never leak a future value. Causality and scale-invariance are covered by two new tests in
`daily/test_no_lookahead_daily.py` (`test_macro_alignment_and_features_are_causal`,
`test_macro_features_are_scale_free`), same style as the existing price-feature tests. Code:
`daily/build_daily_dataset.py --macro`; full before/after results in `daily/results/` vs.
`daily/results/with_macro/`.

Re-ran the identical walk-forward protocol (retrain every 7 days on the last 365, same test window
2025-01-09 to 2026-10-23, n=449) on the exact same rows, features-added-only:

| Model | Acc (price-only → +macro) | AUC (price-only → +macro) | p vs always-up (price-only → +macro) |
|---|---|---|---|
| LSTM | 51.9% → **55.2%** | 0.463 → 0.510 | 0.155 → **1.000** |
| HistGB classifier | 52.6% → 49.9% | 0.498 → 0.488 | 0.255 → 0.026 |
| Ridge | 49.0% → 48.6% | – | 0.009 → 0.005 |

The LSTM's accuracy moved from 3.3 points below the 55.2% always-up rate to exactly level with it, which looks
like an improvement until the mechanism is checked directly (Lessons #4 below): its predicted-`p_up`
distribution shows it still predicts "up" on 82-91% of test days whatever inputs it is given (price-only 88%,
+macro 86%), and the test period itself is 55.2% up days — so a model that mostly says "up" scores well on
accuracy in a trending window by construction, independent of any real signal. It never gets *above* the
always-up rate (a p-value of 1.000 means identical, not better). AUC (which doesn't reward guessing the
majority class) moved from 0.463 to 0.510, both within noise of chance (standard error about 0.03 at n=449).
HistGB and ridge got worse — the HistGB classifier is now significantly below always-up (p = 0.026), plausible
overfitting from 34 features against a 365-day retraining window. **Conclusion: no evidence of real skill gained from macro/cross-asset features
either.** Combined with Phase 8, every lever tried on this dataset — more model types, ensembling, and now
non-price inputs — lands at the same AUC ~0.50 ceiling.

**Addendum 3 (COT positioning added, final)**: also added CFTC Commitment of Traders data — gold's weekly
net non-commercial (speculator) position as a share of open interest, plus its 52-week z-score and 4-/12-week
changes — sourced live from CFTC's own Socrata API (`publicreporting.cftc.gov`, dataset `6dca-aqww`, no
scraping or bot-blocking involved), 1,238 weekly records back to 2003. The report covers positions "as of"
Tuesday but isn't published until the following Friday, so the feature's availability date is shifted +3 days
before alignment — the one new causality risk this feature introduced, now covered by its own test. Result:
LSTM AUC **dropped** to 0.498 (from 0.520 without COT), accuracy 54.6%, p vs. always-up = 0.776 — adding a
5th feature source made things slightly worse, not better, consistent with added dimensionality diluting the
365-day training window rather than contributing real signal. Full results in `daily/results/with_cot/`.

**This closes the feature-search line of investigation.** Five attempts — ensembling (Phase 8), cross-asset/macro,
NFP, FOMC+CPI, and COT positioning (Phase 9; nested feature sets on one shared test window, so not independent
replications) — all land in the same LSTM AUC 0.46-0.52 band with no p-value ever clearing 0.05. One pattern is worth
stating plainly rather than leaving to the table: on the 449-day window the LSTM's point estimates rise as feature
blocks are added (AUC 0.463 → 0.510 → 0.514 → 0.520; rank correlation IC +0.006 → +0.043 → +0.055 → +0.070) before COT
gives part of it back (AUC 0.498). That is what a small real effect would look like, and also what noise on a short,
shared test window can look like: no step is distinguishable from zero (the best IC, +0.070, has p = 0.14 and is the best
of five variants, so a Bonferroni-adjusted p is about 0.7), and even a real rank correlation of 0.07 would lift sign
accuracy by only about two points (0.5 + arcsin(0.07)/π ≈ 52.2% on a symmetric target). Reaching ~70% on daily returns
would need a correlation near 0.59 (R² about 0.35), against the R² of about -0.01 measured here. The result is reported
as a well-instrumented negative result rather than pursued further.

**Addendum 1**: also added an NFP (US Non-Farm Payrolls) release-day calendar feature (`--calendar` flag,
`f_nfp_day`/`f_days_since_nfp` in `daily/build_daily_dataset.py`) on top of the six macro series above — a
deterministic "first Friday of the month" rule, since `bls.gov` blocks automated access and no other source
of the exact historical CPI/FOMC release-date list could be verified reliably enough to trust in a research
dataset (at that point those two were untested for that reason, not by choice; Addendum 2 below adds them). Result: still null (LSTM AUC 0.514,
accuracy 55.0%, p = 0.924; HistGB classifier AUC 0.489) — full results in `daily/results/with_macro_calendar/`.

**Addendum 2 (FOMC + CPI added)**: revisited the CPI/FOMC gap by sourcing exact historical dates directly
from `federalreserve.gov`'s per-year historical-materials pages (2004-2020) and `fomccalendars.htm`
(2021-2027) for 204 FOMC decision days, and `bls.gov/bls/news-release/cpi.htm` for 271 CPI release dates
(2004-2026) — both fetched and hand-verified, not guessed. Added as `f_fomc_day`/`f_days_since_fomc` and
`f_cpi_day`/`f_days_since_cpi`, same causal `_days_since` helper as NFP (event dates are fixed/known in
advance, so no look-ahead risk). This is the closest-to-significant result of every experiment so far — LSTM
AUC 0.520, accuracy 57.5% (above the 55.2% always-up rate), MAE $46.38 (a rare positive skill vs. no-change,
+0.0055) — **but still not significant**: p vs. always-up = 0.367, and checking the mechanism (Lessons #4)
shows the same pattern as every prior run: the LSTM predicts "up" on 89% of days regardless of which
features it's given, inside the 82-91% range of every other version. The point estimate has moved across the
five feature-set variants (51.9% → 55.2% → 55.0% → 57.5% → 54.6% with COT) while AUC stayed in a tight 0.46-0.52 band
and no p-value has cleared 0.05 — consistent with the same underlying null (the small, steady upward drift in AUC and IC
is discussed in the closing paragraph above). (The "predicted down" subset alone shows a higher hit rate, 60% of 50 days, but at n=50 this is exactly
the small-sample, look-at-the-confident-subset trap Phase 6's bid-ask artifact already burned this project
on once — not reported as a finding.) Full results in `daily/results/with_macro_calendar_full/`.

### Phase 10: dashboard redesign, regression-first metrics and price-forecast experiments (October 27-29, 2026)

What was done, in order, with the commits that carry it (`git log` has the full diffs).

**10.1 Data refresh and one results window (27 Oct; `8b27ddc`, `b4da450`).** 25 new trading days were appended locally to the silver table (S3 was not
modified; 22 overlapping days were checked against it), growing the test window from 424 to 449 days (2025-01-09 to 2026-10-23). R² and IC were
moved into the code (`price_report`), the main run was refreshed, and **all four feature-search variants were re-run on exactly the same rows**
(macro, +NFP, +FOMC/CPI, +COT) so every daily number compares like with like. The 424-day outputs are kept in `daily/results/snapshot_2026-09-18/`.
Result unchanged: no variant beats always-up (best FOMC+CPI 57.5%, AUC 0.520, p = 0.37; COT pulls AUC back to 0.498); LSTM AUCs moved by +0.004 to
+0.011. README Phase 9, the thesis tables and the dashboard's "Also tested, still null" table (it reads the `with_*` folders) were updated together.

**10.2 Dashboard: forecast-first redesign and the bugs found on the way (27 Oct; `bbc4173`, `66096ba`, `60570a6`, `18a8499`, `5be730a`).**
- The first redesign drew the forecast card with raw HTML/CSS. It broke twice in practice (a stale browser tab showed the markup as text after a
  redeploy; the price wrapped at laptop widths). Replaced by **native Streamlit elements only**; `width="stretch"` replaces the deprecated
  `use_container_width`; `streamlit>=1.56` floor because Cloud installs a newer release (1.66) than local development used (1.56).
- A second bug was found only by testing with the sidebar open at about 800-1100 px: the three-column hero crushed the price into `$4,045` / `.61`,
  because `st.columns` stacks by *viewport* width while the open sidebar takes ~300 px. The hero and the metric cards now sit in wrapping
  `st.container(horizontal=True)` blocks, which wrap by *content* width. Verified on Streamlit 1.56 and 1.66.1 at 1280, 1024, 900, 800 and 375 px.
- Sidebar now opens with the project credits (MFU and School of Applied Digital Technology logos side by side, title, degree, school, authors with
  student IDs, advisor, committee). Logos are served at full resolution (`st.image(width=...)` would downscale the file and look soft on Retina).
- Tab order: Track record, Data explorer, How reliable?, Method.
- Advisor feedback (accuracy is a classification metric): the "Up/down calls correct" card was replaced by typical miss as % of price (MAPE); "How
  reliable?" now leads with price error, R², IC and a typical-miss-per-model chart, with up/down accuracy in a labelled secondary expander; delta
  chips are coloured by the statistical verdict (green/red only when p < 0.05, orange/blue when within noise).

**10.3 Metric audit and the regression-first rewrite (27-29 Oct; `3854e49`, `d515d16`, `05f8992`).** Auditing what each metric actually measures found:
1. The model table mixed regression and classification models under a "Direction accuracy" headline. Fixed: Table 3.2 is now price accuracy for the
   five models that forecast a price (MAE, RMSE, MAPE, MAE vs. no-change, Diebold-Mariano), direction/AUC moved to a secondary table with all six models.
2. **The "90% likely range" was circular.** Its quantiles came from the whole test window's errors, so "the actual close fell inside on 90% of test
   days" was true by construction. Scored causally (errors known before each forecast only) it covered **85.9%**.
3. The Diebold-Mariano lag (24) was inherited from the hourly pipeline. Checked: the LSTM-vs-no-change conclusion does not depend on it
   (p = 0.38-0.41 for lags 0 to 24); Ridge is borderline (p = 0.023-0.052).
4. R² (sklearn definition) is centred on the test-period mean; added out-of-sample R² against the no-change and rolling-mean forecasts (Campbell and
   Thompson 2008). LSTM: -0.011 against the rolling mean.
New code: `daily/regression_metrics.py` (point metrics, out-of-sample R², DM at lags 0 and 24, IC, causal interval coverage / width / interval
score), `daily/test_regression_metrics.py` (6 tests that can fail, incl. a test that bands use only past errors), output `regression_metrics.json` in
every results folder, called by `walk_forward.py`. The shipped band is now Gaussian with sigma = EWMA(lambda 0.94, RiskMetrics default, fixed in
advance) of past errors: **89.5% coverage (n = 389)**, width $223. The dashboard caption, "How reliable?" and the metrics table use these numbers.

**10.4 Price-forecast experiments (29 Oct).** The project is a *price* forecast, so the remaining ideas were tested on price and its interval:

| Experiment (script, results) | Result |
|---|---|
| 90% interval method comparison (`interval_experiment.py`, `lstm_quantile.py`; `interval_comparison.json`) | EWMA 89.5% (Kupiec p 0.73) beats empirical quantiles 85.9%, GARCH(1,1) 85.6% / FHS 85.3%, and an LSTM trained with pinball loss (63.5%, far too narrow). The LSTM band is as wide as a no-change band ($223 vs. $221): no sharpness gain |
| Price 5 and 20 trading days ahead (`walk_forward_horizon.py`; `horizon_h5/`, `horizon_h20/`) | 5 days: MAE $109.50 vs. $109.19 (p = 0.43). 20 days: $216.95 vs. $219.40 (+1.1%, p = 0.59, only 21 independent windows). Ridge and HistGB worse than no-change at both |
| VIX in the interval model (`interval_vix.py`; `interval_vix.json`) | Refit without VIX 87.1%, with same-day VIX 86.1%, with yesterday's VIX 86.4%, same interval score (322); VIX alone 77.1%. VIX has a positive coefficient (~0.2) but adds nothing out of sample |

**10.5 Writing and decisions.** Chapter 3 of the thesis was restructured regression-first (Tables 3.2-3.6, interval paragraph, multi-day paragraph,
VIX/alternatives sentences) and stays on the 449-day window; the final-report objective was reworded (next-day closing-price forecast judged by
regression metrics against the no-change benchmark with a calibrated 90% interval; **no directional-accuracy target**, thesis §3.7.4 explains why).
The old proposal is superseded and the advisor allows trying anything, so nothing in the Journey is a prohibition. `Report_Abstract_Ch1_Ch4.docx`
(outside the repo) holds the revised Abstract, Chapter 1 sections 1.2/1.3/1.6, Chapter 4 and additional references; Chapter 2 is untouched.
Each Word edit kept a dated backup beside `Chapter3.docx`.

**10.6 Repository housekeeping.** README Handoff block at the top; `AGENTS.md` with the project rules for any coding agent;
`daily/horizon_always_up.py` reproduces the Phase 8 horizon table; README repo tree, "where things live", §10 (dashboard), §12 and §14 brought up to date.

**Scoreboard of everything tried for next-day price forecasting (all on the 449-day window unless noted).** Nothing beats "tomorrow = today" on the
point forecast: LSTM, ridge, gradient boosting, ensembling, macro/cross-asset inputs, NFP/FOMC/CPI calendar, COT positioning, hourly cross-asset
features (own window), 5- and 20-day horizons. What does work is the evaluation and the calibrated interval (EWMA band, 89.5% coverage).

### Lessons learned
1. Predict stationary quantities (returns) from scale-free inputs; a price-level model breaks when prices leave the training range.
2. Every metric needs its trivial baseline: "always up" for direction (not 50%), "no change" for price. Report intervals, and
   use n / horizon when labels overlap.
3. Tests must be able to fail: inject a leak on purpose and check the test catches it.
4. A too-good result is a bug until its mechanism is explained (here: bid-ask spread around the daily break;
   also Phase 9's accuracy-without-AUC gain, explained by the model mostly predicting the majority class).
5. Verify inputs, not just models: silent partial downloads, invalid OHLC rows, columns that contain the future.
6. "Fixed" means re-checking the full period, not the latest weeks.
7. Look at the rendered page: some bugs are invisible to headless tests (raw HTML shown as text; a price crushed by an open sidebar).
8. A coverage or accuracy claim must be scored with information available *before* each forecast. "Covers 90% of test days" was true by construction
   when the band was built from the test window's own errors.
9. Keep one results window for everything (README, dashboard, thesis). A refresh re-runs the main run and every variant on the same rows.
10. Judge a forecast with the metric family that matches its type: price forecasts by price error against the no-change benchmark; direction
    accuracy is a secondary classification check.
11. Check that a conclusion survives the arbitrary choices in the test (e.g. the Diebold-Mariano lag), and when several alternatives are compared on
    one test window, say so: the winner's score is slightly optimistic.
12. Test the evaluator, not only the model: the metrics module has tests that fail on look-ahead and on wrong formulas.

### Where things live
| Path | What |
|---|---|
| `notebooks/`, `automation/` | Original daily pipeline (notebooks 01-06) and the Lambda automation, which is designed but not deployed and still runs the old price-level model |
| `daily/` | Corrected daily dataset builder, causality tests, walk-forward evaluation, regression-first metrics (`regression_metrics.py`), multi-day walk-forward (`walk_forward_horizon.py`) and the interval experiments (`interval_experiment.py`, `lstm_quantile.py`, `interval_vix.py`). `results/` (the main price-only run) feeds the dashboard; `results/with_macro/`, `with_macro_calendar/`, `with_macro_calendar_full/` and `with_cot/` are the Phase 9 feature-search runs on the same rows; `results/snapshot_2026-09-18/` keeps the earlier 424-day outputs of all five |
| `hourly/` | Hourly experiments: downloader, builder, tests, baselines, models, bid-vs-mid check and the cross-asset test (`results/with_aux_mid/`); `results/summary_hourly_findings.png` is the one-page summary |
| `dashboard/app.py` | Streamlit app (deployed on Streamlit Community Cloud from `main`); `dashboard/assets/` holds the MFU and ADT logos and `.streamlit/config.toml` the theme |
| `docs/TROUBLESHOOTING.md` | Chronological error log for the original pipeline |

### Status and next steps
See the **Handoff** block at the top and §12 for the tracked open items. In short: the modelling work is finished and closed as a
negative result (every lever tried, including macro, NFP, FOMC/CPI and COT features on daily and hourly data, lands at AUC ~0.50,
so the honest write-up is a well-instrumented negative result rather than a 70% claim). What remains is writing and housekeeping:
the thesis write-up (Chapter 3 is on the 449-day window; a regression-first restructure is optional), the Lambda decision (finish it
around the return-target model, or tear down the partial AWS changes), and refreshing the results when new data arrives (§12 item 12).

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
lstm_dashboard/
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
│   ├── app.py                   ← Streamlit dashboard (deployed on Streamlit Community Cloud; reads S3 + daily/results) — see §10
│   ├── assets/                  ← mfu_logo.png, adt_logo.png (sidebar header)
│   ├── requirements.txt         ← sets a streamlit>=1.56 floor (Cloud installs the newest release)
│   └── .python-version          ← 3.11
├── .streamlit/config.toml       ← theme colour and minimal toolbar
├── .claude/launch.json          ← preview-server config for Claude Code sessions (optional)
├── .devcontainer/               ← dev container definition
├── .python-version              ← 3.11
├── automation/                  ← daily inference-refresh Lambda — see §11 and automation/README.md
│   ├── README.md                ← full design rationale + deploy/teardown steps
│   ├── lambda_function.py       ← the daily handler (inference-only, no training)
│   ├── convert_and_export.py    ← one-time: Keras -> TFLite + scaler param export
│   ├── Dockerfile, requirements-lambda.txt
│   ├── deploy.sh                ← idempotent: ECR + Lambda + EventBridge Scheduler
│   └── iam_*.json, scheduler_*.json  ← least-privilege policies for the Lambda's own role
├── daily/                       ← corrected daily pipeline (Sep 2026) — see Journey and §8.0
│   ├── build_daily_dataset.py   ← scale-free features + log-return targets from the Silver table; --macro / --calendar / --cot add the Phase 9 feature sets
│   ├── test_no_lookahead_daily.py
│   ├── walk_forward.py          ← walk-forward LSTM (retrain every 7 days, 365-day window) + baselines + tomorrow's forecast
│   ├── horizon_always_up.py     ← always-up accuracy by horizon (Phase 8 table)
│   ├── regression_metrics.py, test_regression_metrics.py   ← price error, out-of-sample R², causal interval coverage (Phase 10)
│   ├── walk_forward_horizon.py  ← price 5 / 20 trading days ahead vs. no-change (results/horizon_h5, horizon_h20)
│   ├── interval_experiment.py, lstm_quantile.py, interval_vix.py   ← 90% interval method comparison and the VIX test
│   ├── requirements.txt
│   └── results/                 ← main run (449-day window): walk_forward_results.json, wf_predictions.csv, latest_forecast.json (read by the dashboard)
│       ├── with_macro/, with_macro_calendar/, with_macro_calendar_full/, with_cot/   ← Phase 9 feature-search runs on the same rows
│       ├── snapshot_2026-09-18/ ← the earlier 424-day outputs of all five runs
│       └── model_results_lstm_w*.json   ← static-training variants (not re-run in the October refresh)
├── hourly/                      ← hourly XAU/USD experiments (Dukascopy) — see Journey, phases 5-6
│   ├── download_dukascopy.py, build_hourly_dataset.py, test_no_lookahead.py
│   ├── baseline.py, train_models.py, spread_artifact_check.py, make_summary_figure.py
│   ├── run_full_pipeline.sh, package.json   ← download (bid, ask, aux instruments) and build; needs Node for dukascopy-node
│   └── results/                 ← metrics JSON + summary_hourly_findings.png; with_aux_mid/ is the hourly cross-asset test
└── .gitignore                   ← excludes venv/, .aws/, daily/data/, hourly/data/ and pulled *.npy/*.pkl/*.keras/*.parquet artifacts
```

All 6 notebooks in `notebooks/` are the **current, corrected** versions —
every fix described in this README and in `docs/TROUBLESHOOTING.md` is
already applied to these files, not just described in prose.

---

## 2. Project Objectives, Scope & Plan (from the proposal)

### 2.1 Objectives

> **Superseded wording (2026-10-29):** the final report's objective is a next-day gold closing-price forecast judged by regression metrics against the
> no-change benchmark, with a calibrated 90% interval; no directional-accuracy target (see §12 item 14). The text below is the original proposal.

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

> **Update (Sep 2026):** the price-level features and the price target defined here (`close`, MA7/14/30/60, USD momentum,
> `target = close.shift(-1)`) are the root cause of the scale problem described in the Journey. The corrected, scale-free
> version is `daily/build_daily_dataset.py`. The `_1w`/`_1m` columns carried in Silver contain future information for
> mid-period days and must not be used as features.

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

> **Update (Sep 2026):** predicting a price level with scalers fixed on the training period is what made the LSTM's error
> explode in 2025-26 (MAE ≈ \$640). The corrected walk-forward (same retrain-every-7-days / 365-day scheme, return target,
> scalers re-fit per window) is `daily/walk_forward.py`.

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

> **Superseded (September 2026).** §8.1-§8.4 describe the original price-level pipeline. Its LSTM error (MAE ≈ \$640) came
> from a scale problem and its "best directional accuracy" was measured with a non-standard metric and without the
> always-up baseline. Use §8.0 for current numbers; the original text is kept for the record.

### 8.0 Corrected results (walk-forward on returns, September 2026)

Same walk-forward scheme as notebook 05 (retrain every 7 trading days on the latest 365 days), but predicting the
next-day log-return from scale-free features with scalers re-fit in every window (`daily/walk_forward.py`). Test period
2025-01-09 to 2026-10-23 (449 days); all models are scored on the same days. Gold rose on
55.2% of these days, so "always up" scores 55.2%.

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | MAE vs no-change (%) | DM p |
|---|---|---|---|---|---|
| No-change (naive) | 46.63 | 68.94 | 1.14 | – | – |
| Rolling drift | 46.36 | 69.07 | 1.14 | -0.6 | 0.941 |
| Ridge | 47.87 | 70.35 | 1.17 | +2.7 | 0.023 |
| HistGB regressor | 50.68 | 75.05 | 1.24 | +8.7 | <0.001 |
| **LSTM (walk-forward)** | 46.83 | 69.45 | 1.15 | +0.4 | 0.385 |

Secondary, classification-style check (direction), all six models:

| Model | Direction accuracy | vs always-up (pt) | p vs always-up | AUC |
|---|---|---|---|---|
| Rolling drift (= always up) | 55.2% | +0.0 | 1.000 | – |
| Ridge | 49.0% | -6.2 | 0.009 | – |
| HistGB regressor | 50.3% | -4.9 | 0.041 | – |
| HistGB classifier | 52.6% | -2.7 | 0.255 | 0.498 |
| **LSTM (walk-forward)** | 51.9% | -3.3 | 0.155 | 0.463 |
| Momentum rule | 45.2% | -10.0 | 0.000 | – |

**Interpretation.** The corrected LSTM's price error is essentially the no-change level and its direction accuracy is not
distinguishable from always-up (95% CI 47.3-56.5%, AUC 0.46). With 449 test days the interval is about ±5 points, so small
edges cannot be detected either way. Hourly experiments (Journey, phases 5-6) reach the same conclusion. The earlier "central
finding" (worst MAE but best directional accuracy) was produced by the price-level scale problem and a non-standard
direction metric. Raw results: `daily/results/`, `hourly/results/`.

### 8.1 Original results (superseded)

| Model | MAE (USD) | RMSE (USD) | Directional Accuracy |
|---|---|---|---|
| Naive Persistence | ~51.1 | ~74.3 | ~44.7% |
| AR(5) | ~51.0 | ~74.2 | ~44.1% |
| Linear Regression | ~51.2 | ~74.1 | ~43.9% |
| **LSTM Walk-Forward** | ~641.0 | ~776.6 | **~51.5%** |

### 8.2 Interpretation (original, superseded)

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

### 8.3 What the proposal already gets right (original, superseded: the narrative must now change)

The proposal's own §3.7 narrative text (already written in the current
`.docx`/PDF) already states this same interpretation correctly, including
the exact 51.5% directional-accuracy figure and the out-of-distribution
price-surge explanation. **The only thing actually missing from the
current proposal is the numeric MAE/RMSE table** (§3.7.3 describes the
comparison narratively and via Figure 3.10, but never tabulates the
numbers) — that's the one addition needed, not a rewrite.

### 8.4 Draft table + caption text to insert into §3.7.3 (original: do NOT paste, use §8.0)

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

## 10. Streamlit Dashboard (deployed on Streamlit Community Cloud)

> **Update (Sep 2026):** the dashboard is also deployed on Streamlit Community Cloud (Python 3.11; AWS credentials in the
> app's Secrets, never in git). The KPIs and the Model Evaluation tab now lead with the corrected evaluation read from
> `daily/results/`; the original views live in an expander ("Original pipeline"). The layout description below covers the
> original design and the views kept in that expander.
>
> **Update (Oct 2026): forecast-first redesign.** The next-day forecast is now the largest element on the page (big price,
> change vs. last close, 90% likely range, and a recent-price chart with the forecast and its range drawn on it). Tabs are
> now **Track record** (past forecasts vs. actual), **Data explorer** (price history, volatility, features), **How reliable?**
> (plain-language verdict from price error vs. "tomorrow = today", R² and IC, typical miss per model, the full metrics table in
> an expander, and the up/down check kept as a clearly labelled secondary view) and **Method**. After advisor feedback that
> directional accuracy is a classification metric rather than the yardstick for a price forecast, the headline metrics under the
> forecast are now the typical daily miss in dollars and as a share of the price (chips are green or red only when the
> difference from "tomorrow = today" is statistically clear, orange or blue when it is within noise); the proposal's ~70%
> direction target is still addressed, in the secondary view and in thesis §3.7.4. The sidebar opens with the project credits (MFU and School of Applied Digital Technology logos, authors
> with student IDs, advisor and examining committee), then data freshness and a "Model and data details" expander. The tab-by-tab description below
> predates this and is kept for the record. Streamlit theme and toolbar settings live in `.streamlit/config.toml`.
>
> Lessons from the first deployed version of this redesign: (1) it drew the forecast card with raw HTML/CSS, which broke
> twice in practice (a 4rem price wrapped onto three lines at ordinary laptop widths, and one open browser tab showed the
> markup as plain text after a redeploy), so the page now uses **only native Streamlit elements**; (2) Cloud installs the
> newest Streamlit (1.66 at the time) while local development was on 1.56, so `dashboard/requirements.txt` now sets a floor
> (`streamlit>=1.56`) and every layout change should be checked at 1280, 1024, 900, 760 and 390 px wide *and* on the
> deployed URL; (3) `use_container_width` had been deprecated (removal date passed), replaced by `width="stretch"`;
> (4) the three-column forecast card still crushed the price (`$4,045` / `.61`) at about 800–1100 px with the sidebar open,
> because Streamlit stacks columns by *viewport* width (below 640 px) while the open sidebar takes ~300 px of it. The card now
> uses wrapping horizontal containers (`st.container(horizontal=True)`), which wrap by *content* width, so the range and
> last close drop below the price instead of squeezing it.

**STATUS: live and redesigned (October 2026).** Further iteration is welcome (nothing here is frozen), but the structure in the next
subsection is what a reader sees today.

### Current dashboard (October 2026)
- **Hosting.** Streamlit Community Cloud, app path `dashboard/app.py`, Python 3.11, redeployed automatically from `main`. AWS credentials
  live in the app's Secrets (never in git). For local runs use `python3 -m streamlit run dashboard/app.py` (needs AWS credentials for
  the S3 reads).
- **Sidebar (credits first).** MFU and ADT logos side by side, project title, degree and school, authors with student IDs, advisor and
  committee, then data freshness and a "Model and data details" expander.
- **Header (the forecast is the product).** Next-day forecast price (the largest element), change vs. last close, 90% likely range and
  last close, a 1M/3M/6M/1Y chart with the forecast and its range drawn on it, then three cards: typical daily miss (MAE in dollars),
  typical miss as a share of the price (MAPE) and the data-through date. Chips are green or red only when the difference from
  "tomorrow = today" is statistically clear (p < 0.05), orange or blue when it is within noise.
- **Tabs, in order.** *Track record* (forecast vs. actual and the daily miss); *Data explorer* (price history with volume, 30-day
  volatility, return distribution, feature explorer); *How reliable?* (verdict, price error, R² and IC, typical miss per model vs.
  "tomorrow = today", a secondary up/down expander, a full-metrics expander, and the original pipeline kept for the record);
  *Method* (bottom line, what changed after review, the "Also tested, still null" table read from `daily/results/with_*`, and the
  original methodology notes).
- **Data it reads.** `daily/results/walk_forward_results.json`, `wf_predictions.csv` and `latest_forecast.json` (cache keyed on file
  mtime, so a refresh shows up without a restart), the four feature-search folders, and the S3 Gold/model artifacts for the original
  pipeline views.
- **Rules that came from real bugs:** native Streamlit elements only; `width="stretch"` instead of `use_container_width`; escape
  dollar signs in markdown as `\$`; wrapping containers instead of `st.columns` where the open sidebar would squeeze content; check
  every layout change at several widths and on the live URL (see the lessons above).

**Hosting decision.** Streamlit Community Cloud is free. The real AWS cost risk in this project is compute left running continuously
(not S3 reads or occasional inference), so the dashboard is deliberately not deployed to always-on AWS compute.

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

### Layout of the original design (superseded October 2026; kept for the record)
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

1. **Thesis write-up.** *Status (Oct 2026): the advisor allowed rewriting, so the old proposal is superseded and a new report is
   being written. `Chapter3.docx` already has Table 3.2 replaced with the corrected results, §3.6.2's window fixed, and §3.7.4 with
   Tables 3.3-3.5 (R²/IC, feature sets, horizons) added, all on the 449-day window; backups sit next to the file. The checklist
   below is the original list for the old proposal; apply what is still relevant to the new report.*
   - Fix the Figure 3.9/3.10 caption swap noted earlier in the project.
   - Replace the interpretation and Table 3.2 with the corrected results in
     §8.0 (the original §8.4 table and the "worst MAE but best direction"
     narrative are superseded: they came from the scale problem).
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
2. ~~Do not cite the old R² (~0.87)~~ **Done** — computed on returns from
   `daily/results/wf_predictions.csv`. It predates both the AR(5)/early-stopping
   fixes and the scale correction (Journey, phase 7), and is superseded: the
   corrected LSTM's R² is **-0.014** (worse than predicting zero return every
   day, R² = -0.003). See Journey, Phase 7 addendum for the full table
   (R² and IC for every model).
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
7. ~~Keep iterating on the Streamlit dashboard~~ **Done for now (Oct 2026):** the forecast-first redesign, credits sidebar,
   regression-first metrics and responsive layout are live (§10). Further changes are optional and should follow the rules in §10.
8. **Actually deploy and verify the daily automation** (§11) against the
   live AWS account — run `convert_and_export.py` + `deploy.sh`, confirm a
   real invocation writes a correct `latest_forecast.json`, then let it run
   for at least one real trading-day rollover before demoing it as "live"
   rather than "designed and ready." **Status (Sep 2026):** the attempt stopped at
   the Docker build (disk full); the ECR repository and three extra managed
   policies on `Auto` already exist (Journey, phase 4). The Lambda still runs
   the original price-level model and must be rebuilt around the return-target
   model before it is worth deploying.
9. ~~Port the Silver-based forecast-window fix into the dashboard's
   Next-Day Forecast.~~ **Done (Sep 2026):** the corrected daily forecast
   (`daily/walk_forward.py`) builds its features from the Silver table, so it
   includes the newest day, and the KPI reads `daily/results/latest_forecast.json`.
10. Optional/future, once §11 is live and trusted: automate *absorbing*
    `bronze/streaming/xauusd_daily_incremental.csv` back into the canonical
    Bronze/Silver/Gold files, and add a second, much-less-frequent
    (e.g. weekly) scheduled job that re-runs full walk-forward retraining
    — see `automation/README.md` §9 for why this is scoped out for now.

11. **Clean up or finish the partial AWS deployment** (Journey, phase 4): the ECR
    repository `gold-lstm-forecast-daily-refresh` and the policies
    `AWSLambda_FullAccess`, `AmazonEC2ContainerRegistryFullAccess`,
    `AmazonEventBridgeSchedulerFullAccess` attached to `Auto` are standing changes
    on the friend's account; detach/delete them if the automation is dropped.
12. **Refresh the corrected daily results** after new data arrives:
    `python daily/build_daily_dataset.py` then `python daily/walk_forward.py ...`
    (commands in the script docstrings), commit `daily/results/`. **Last done 2026-10-27** (data through 2026-10-26,
    see Journey, Phase 7 refresh note). The S3 silver table is still at 2026-09-21: re-run notebooks 01-02 (or the Lambda
    path) to bring it forward, or pass a locally extended silver parquet to `--silver`. When refreshing, re-run the main walk-forward
    **and all four feature-search variants on the same rows**, and update the README, Chapter 3 and the dashboard together; delete or
    repoint `daily/data/macro_cache.parquet` and `cot_cache.parquet` first, because a stale cache is reused as-is (see Handoff).
13. **Daily-pipeline cross-asset + calendar + COT features: done (Phase 9, all addenda), null** —
    DXY, silver, S&P 500, VIX, 10Y yield, TIP, NFP, FOMC (204 dates), CPI (271 dates) and CFTC Commitment of
    Traders positioning all added to the daily model; best single result is LSTM AUC 0.520 (macro+NFP+FOMC+CPI,
    without COT), p vs. always-up = 0.367 — closer than every other variant but never significant; adding COT
    on top made it worse (AUC 0.498). This particular feature search found nothing; other ideas remain open. The hourly version of
    cross-asset features (silver, EURUSD, S&P 500, DXY as hourly bars) was run too and is also null (hourly
    addendum in the Journey); re-download the four missing ask months (2014-04, 2015-07, 2016-08, 2026-09) once
    Dukascopy's rate limit allows.
14. ~~Agree with the advisor what "~70% accuracy" refers to~~ **Resolved (2026-10-29, decided by the project owner):** the ~70% direction
    figure was an early expectation, not a written objective, and is not adopted. Objective wording for the final report: *a next-day gold
    closing-price forecast, judged by regression metrics (MAE, RMSE, MAPE, Diebold-Mariano) against the no-change benchmark, with a 90% forecast
    interval whose coverage is close to nominal.* Direction accuracy stays a secondary check (Chapter 3, Table 3.4); thesis §3.7.4 explains why
    (always-up already scores 55.2%, and ~70% sign accuracy on daily returns would need a correlation near 0.59, an R² near 0.35).
15. ~~Restructure Chapter 3 regression-first~~ **Done (2026-10-29):** see the Handoff block.
16. **Keep results, README, thesis and dashboard on one window.** Every daily number is on 2025-01-09 to 2026-10-23 (n = 449). A refresh
    regenerates the main run and all four variants together, updates the README (Journey, §8.0, Phase 8 and 9), Chapter 3
    Tables 3.2-3.5 and keeps the previous outputs in a dated `snapshot_*` folder.
17. **Privacy.** The repo is public: no AWS keys or account ID in any file; the student names and IDs shown in the dashboard are there
    by the owner's choice.

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

## 14. Repository, hosting and publishing

The code lives at https://github.com/aphiwat-data/lstm_dashboard (public). `main` is the deployed branch: a push redeploys the
Streamlit app within about a minute, so look at the live URL after every push (the deployed Streamlit version is newer than the one
used locally). New clone: `git clone https://github.com/aphiwat-data/lstm_dashboard.git`, then `pip install -r dashboard/requirements.txt`
(and `daily/requirements.txt` for the daily scripts).

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
