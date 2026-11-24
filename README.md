# Next-Day Gold Price Forecasting with LSTM — Project README

Undergraduate Computer Engineering senior project, Mae Fah Luang University.
This file is the single source of truth for continuing the work — proposal
context, AWS setup, pipeline internals reconciled against the actual code,
every real issue hit and fixed, current evaluation results, and everything
still pending. It merges the written thesis proposal (`Next_day_Gold_
Prices_Forcasting_With_LSTM.pdf`) with what the pipeline *actually* does
today, and calls out every place the two disagree. Read this fully before
making changes. If you are picking the project up, start with the
**[Handoff](#handoff-where-things-stand-2026-11-17)** block below.

For the full chronological story of every AWS/code error hit and how it
was fixed (useful for "what problems did you encounter" in the defense),
see **[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)**.

> **Update (September 2026): the evaluation results in §8.1–§8.4 are superseded.** On review, the original
> pipeline's very large LSTM error (MAE ≈ \$640) was a scale problem, not a market effect, and its "best
> directional accuracy" did not survive comparison with the trivial "always up" rule. Read the
> **[Project Journey](#project-journey-what-we-tried-what-broke-what-we-learned)** first; the corrected numbers are in §8.0.

**Contents.**
1. [Handoff](#handoff-where-things-stand-2026-11-17): the current result, the live pipeline, how to run and refresh things, traps and open items.
2. [Project Journey](#project-journey-what-we-tried-what-broke-what-we-learned): Phases 1-13 (the dashboard's final form is Phase 13 plus its appendix [Content removed from the dashboard](#content-removed-from-the-dashboard-november-17-2026)),
   then [lessons learned](#lessons-learned), [where things live](#where-things-live) and [status and next steps](#status-and-next-steps) (including the publication plan).
3. Reference sections: §0 front matter, [§1 repository structure](#1-repository-structure), §2-§3 proposal and deviations, [§4 AWS state](#4-aws-account-state-actual-as-of-now), §5-§6 data architecture and notebooks, §7 risks,
   [§8 results](#8-model-evaluation--results-interpretation-37-write-up), §9 literature, [§10 dashboard](#10-streamlit-dashboard-deployed-on-streamlit-community-cloud), [§11 daily automation](#11-daily-automation-pipeline-lambda--eventbridge-scheduler),
   [§12 pending](#12-pending--not-yet-done), §13 references, §14 repository and hosting, [§15 research dossier for the IEEE paper](#15-research-dossier-for-the-ieee-paper).

---

## Handoff: where things stand (2026-11-17)

Read this block first if you are picking the project up (a new contributor or another coding agent); everything below it is the
detailed record.

**Links.** Code: https://github.com/aphiwat-data/lstm_dashboard (public, branch `main`). Live dashboard:
https://lstmdashboard-v6sere4oaulpge5dj9bq5p.streamlit.app (Streamlit Community Cloud redeploys from `main` within about a minute
of a push). The thesis documents (Word/PDF, including Chapter 3) are kept outside this repo.

**The result.** On the 449-day walk-forward test window (2025-01-09 to 2026-10-23) the LSTM is level with a no-change forecast, not
better:

| Check | Result |
|---|---|
| Price error | MAE \$46.83 vs \$46.63 for "tomorrow = today", the *simple guess* of repeating the last close (Diebold-Mariano p = 0.38); 1.15% of the price |
| Fit on returns | R² -0.014 (a constant zero-return forecast scores -0.003); IC +0.006 (p = 0.90) |
| Direction (secondary) | 51.9% vs always-up 55.2% (p = 0.155), AUC 0.463; the model says "up" on about 88% of days |
| 90% range | Reliable: the real close stayed inside it on 348 of 389 scored test days (89.5%, goal 90%; Kupiec p = 0.73). It is as wide as a no-change range (\$223 vs \$221), so it tracks market volatility, not model skill |
| Live record (2026-11-17) | 11 real forecasts scored since 2026-10-28: MAE \$48.49 vs \$48.21 for the simple guess; 9 of 11 inside the 90% range. Far too few days to conclude anything; it grows by one trading day each morning |
| Risk (E6, 2026-11-24) | Volatility is predictable and its R² is positive: Mincer-Zarnowitz R² 0.19 for the shipped EWMA and 0.29 for HAR-RV with release days (2010-2026); HAR-RV beats the EWMA (QLIKE -0.054, p < 0.001; not significant on 2025-26, p = 0.10); an LSTM does not help. The return R² (-0.011) is unchanged (§15.5) |
| Extra features | Macro, NFP, FOMC+CPI and COT (daily and hourly) are all null; best daily variant FOMC+CPI 57.5%, AUC 0.520, p = 0.37 |
| The ~70% target | Not reachable on this evidence; ~70% sign accuracy on daily returns would need a predicted-vs-actual correlation near 0.59 |

So far the write-up is a well-instrumented negative result. **The project is open to new approaches**: the old proposal and its ~70% target are superseded
and the advisor allows trying anything. The feature searches in Phase 9 were null (evidence, not a prohibition); judge new ideas with the
regression-first metrics and the same causal protocol.

**Metric policy (advisor, 2026-10-27).** Directional accuracy is a classification metric, not the yardstick for a price forecast. Lead
with price error (MAE, RMSE, % error, R², IC) against the "tomorrow = today" baseline and keep up/down accuracy only as a labelled
secondary check (it is how the proposal phrased the ~70% target, and it exposes the model's majority-class behaviour). The dashboard
leads with price error and, since 17 Nov, no longer shows direction accuracy at all (its former side check is kept under "Content removed from the dashboard" in Phase 13); Chapter 3 did the same (price accuracy first, direction as a secondary table).

**Dashboard (final form, 2026-11-17; the owner closed the dashboard work that day).** `dashboard/app.py` is a 40-line entry point; the page lives in `dashboard/ui/` (`data.py` cached loaders, `header.py`, `sidebar.py`, `cards.py` metric cards,
one `tab_*.py` per section, `theme.py` colours and the shared chart style, `logic.py` the price verdict and the chip colour). It does not import TensorFlow (requirements: streamlit, boto3, pandas, numpy, plotly, pyarrow), so a page
loads in about a second. Every number in the prose is read from the result files, so a results refresh needs no dashboard edit. Theme and font are in `.streamlit/config.toml` (one light theme, on purpose).
The page, top to bottom: a **header** (the title with a freshness badge; three lines under it: what the model forecasts, that live forecasts are saved before the day and scored against the real close, and the senior-project disclaimer; the forecast card with
the forecast day, the forecast price, its change against the last close and the 90% range with its *range reliability*; the chart "Actual vs Forecast" with a 1M / 3M / 6M / 1Y filter; the scorecard "Accuracy · 449 backtest days" with MAE, RMSE, error % and R²,
each card showing only a label and a number), then a **navigation bar of three sections** in one row (nothing is shown until one is clicked; click the active one to hide it): *Forecast vs. actual* (the backtest with its window filter at the right, then the live record),
*Actual market data* (price history, volatility, daily changes, the model's inputs) and *How reliable?* (the evidence rows, average miss by model, range reliability by method). The sidebar holds the credits. The details are in §10 ("Current dashboard").
Step by step the owner removed the briefing, notes, chips, verdict boxes, the side-check and statistics expanders, many tooltips and the whole *About the project* section; everything removed is kept in "Content removed from the dashboard" (Phase 13)
and `AGENTS.md` lists the rules so that it stays removed.

**Live pipeline (since 2026-10-29; Phase 11, §11).** An EventBridge schedule (`cron(0 3 * * ? *)`, 03:00 UTC = 10:00 Thailand, ENABLED) invokes the Lambda `gold-lstm-forecast-daily-refresh` (python3.12, 1 GB, about 1.5 s per run) in `ap-southeast-2`.
Each morning it appends the newest settled Yahoo bar to `bronze/streaming/xauusd_daily_incremental.csv`, runs data-quality checks, makes the forecast with a numpy LSTM and its 90% EWMA range (`gold/xauusd_daily/predictions/latest_forecast.json`)
and scores every earlier forecast whose next close is now known (`forecast_log.csv`, the live record). Status on 2026-11-17: the 10:00 run worked (latest forecast, made from the 2026-11-16 close: \$3,858.67, range \$3,776.27 to \$3,942.87);
12 forecasts are logged, 11 scored, 9 of 11 inside the range, MAE \$48.49 against \$48.21 for the simple guess. **Open:** (1) the serving model (`model/serving/serving_model.npz`) was last written on 2026-10-29, so the weekly
`python3 automation/retrain.py` (run by hand; needs TensorFlow and AWS credentials) is two retrains behind; (2) `forecast_log.csv` is an ordinary S3 object and the Lambda replaces the row of the same as-of date on a re-run, so the live record is not tamper-proof
(for a paper add S3 versioning with Object Lock, or a daily git commit of the log; the *Live record* tooltip on the page still says nothing can be edited afterwards); (3) no CloudWatch alarms or SNS topic (the IAM user `Auto` lacks the permission and the owner chose not to widen it);
(4) `AmazonEC2ContainerRegistryFullAccess` is still attached to `Auto` although nothing uses it. See §12 items 18-22.

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

**How to run things** (the S3 silver table and the dashboard's live forecast and price history need AWS credentials, and the dashboard falls back to the repository
files when S3 cannot be read; pass `--silver <local parquet>` to keep the daily scripts off S3):

```bash
python3 -m streamlit run dashboard/app.py          # dashboard on http://localhost:8501 (repo root)
cd daily
python3 test_no_lookahead_daily.py                 # causality tests
python3 build_daily_dataset.py --out-dir data/processed
python3 walk_forward.py --data data/processed/xauusd_d1_features.parquet --latest data/processed/xauusd_d1_latest_features.parquet --out-dir results
python3 horizon_always_up.py                       # the Phase 8 horizon table
python3 test_regression_metrics.py                 # metric tests; test_serving.py (numpy LSTM equals Keras) and ../automation/test_lambda_function.py run the same way
cd ../automation && python3 deploy.py plan         # the daily Lambda and its schedule: plan / apply / teardown (idempotent)
python3 retrain.py                                 # weekly: retrain the serving model and upload it (needs TensorFlow and AWS credentials)
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

**Refreshing the results** (§12 item 12; last done 2026-10-27 with data through 2026-10-26). (1) Extend the silver data: the S3 Silver table is still at 2026-09-21 (the Lambda keeps newer bars in `bronze/streaming/xauusd_daily_incremental.csv` and nothing absorbs them into Silver yet),
so build a local parquet with the same columns and pass it to `--silver`. (2) Delete or repoint `daily/data/macro_cache.parquet` and `cot_cache.parquet` first: `fetch_macro` and `fetch_cot` return a cache as-is without checking its end date, so a stale cache silently
forward-fills old values. (3) Re-run the main walk-forward **and all four variants on the same rows**, keep the previous outputs in a dated `daily/results/snapshot_*` folder (only `snapshot_2026-09-18/` exists so far; copy the current outputs into a new one before overwriting)
and update the README numbers. The dashboard reads the result files by itself (cache keyed on file mtime) and its prose is generated, so it needs no edit; the README tables are typed by hand. The Word files are no longer maintained. A long-window run for a paper belongs in its own
results folder, so that the dashboard's 449-day window stays consistent.

**Paper (IEEE; the next step since 2026-11-17).** The research dossier is §15: what the evidence supports and does not support, every number with its source file, the method in reproducible detail, threats to validity, extra experiments (E1-E12), a checked bibliography with BibTeX (39 references, DOIs verified 2026-11-17), an IEEE venue shortlist (JCSSE 2027 second round on 22 Jan 2027, hosted by the owner's school; IEEE Access, rolling), a section-by-section writing plan and the rules for working with Claude (§15.13). Read §15.0 and §15.13 before drafting anything.

**Working agreements and traps.**
- The repo is public: never commit AWS keys, the AWS account ID or any credentials (Streamlit Cloud keeps them in the app's Secrets).
  The student names and IDs in the dashboard sidebar are public on purpose (the owner's request).
- Dashboard code: native Streamlit elements only (no `unsafe_allow_html`), `width="stretch"`, escape dollar signs in markdown as
  `\$`, and use `st.container(horizontal=True)` instead of `st.columns` for anything that must survive the open sidebar. Cloud
  installs a newer Streamlit than local development used (1.66 vs 1.56), so check every layout change at 1280, 1024, 900, 760 and
  390 px and on the live URL after the push (§10, lessons).
- The owner pushes small commits straight to `main`. Every push redeploys the public app, so a broken `app.py` is public at once.
- Checking a second Streamlit version means proving which version ran: print `streamlit.__version__` first (a broken scratch venv silently ran 1.56 for days). Streamlit 1.66's `AppTest.from_file` resolves a relative path against the calling file, so give it the absolute path of
  `dashboard/app.py`; prime `st.session_state["section"]` to test each navigation section.
- The owner decides every visible element of the page and writes the rules down in `AGENTS.md`: never delete a chart or section unless the owner asks for that exact removal, and put removed content into this README (see "Content removed from the dashboard").
- The paper will be rewritten from scratch by the owner (2026-10-29): the Word files (`Chapter3.docx`, `Report_Abstract_Ch1_Ch4.docx`, kept outside the repo) are no longer maintained. Record every result and decision in this README only.

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
| Is the 90% range trustworthy? | **Yes.** The real close stayed inside it on 89.5% of 389 scored test days (goal 90%, Kupiec p = 0.73) and on 9 of 11 live days so far. It is as wide as a no-change range, so it reflects market volatility, not model skill. |
| What does the live record say? | 11 real forecasts scored (2026-10-28 to 2026-11-16): MAE \$48.49 vs \$48.21 for the simple guess, 9 of 11 inside the range. Far too few days to conclude anything; what it shows is that the pipeline runs unattended. |
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

### Phase 4: attempt to deploy the automation (finished in Phase 11)
`automation/deploy.sh` attached three managed policies to `Auto` (Lambda, ECR and EventBridge Scheduler full access) and
created the ECR repository `gold-lstm-forecast-daily-refresh`, then the Docker build failed with I/O errors because the
Mac's disk was 98% full (249 MB free); after freeing space Docker Desktop's engine still did not become healthy after a
restart. The Lambda and schedule were never created. Resolved in Phase 11: the Lambda was rebuilt around the return-target model and deployed on 2026-10-29, and the empty ECR repository was deleted; the broad managed policies on `Auto` remain (§12 item 11).

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
- Data explorer has a "Volatility: forecast vs. realised" section (30 Oct): expected daily move (1 sigma), annualised, the previous VIX close, a chart of actual daily moves against the
  90% half-width the model expected, and the QLIKE table of volatility models (reads `daily/results/volatility_*`).
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
| Volatility forecasting itself (`volatility_experiment.py`; `volatility_comparison.json`, `volatility_forecast.csv`) | Scored with QLIKE (Patton 2011) and a DM test against EWMA(0.94) on 449 days: EWMA 0.94 -7.356 is best; EWMA 0.97 -7.325, GARCH(1,1) -7.288 (p = 0.07), EWMA + VIX -7.292, VIX scaled -6.686, EWMA + slower EWMA + VIX -7.267. Empirical standardised quantiles instead of Gaussian z fix coverage (89-90%) but widen the band and worsen the interval score. No volatility model beats the simple EWMA; the VIX does not enter the forecast |

**10.5 Writing and decisions.** Chapter 3 of the thesis was restructured regression-first (Tables 3.2-3.6, interval paragraph, multi-day paragraph,
VIX/alternatives sentences) and stays on the 449-day window; the final-report objective was reworded (next-day closing-price forecast judged by
regression metrics against the no-change benchmark with a calibrated 90% interval; **no directional-accuracy target**, thesis §3.7.4 explains why).
The old proposal is superseded and the advisor allows trying anything, so nothing in the Journey is a prohibition. `Report_Abstract_Ch1_Ch4.docx`
(outside the repo) holds the revised Abstract, Chapter 1 sections 1.2/1.3/1.6, Chapter 4 and additional references; Chapter 2 is untouched.
Each Word edit kept a dated backup beside `Chapter3.docx`.

**10.6 Repository housekeeping.** README Handoff block at the top; `AGENTS.md` with the project rules for any coding agent;
`daily/horizon_always_up.py` reproduces the Phase 8 horizon table; README repo tree, "where things live", §10 (dashboard), §12 and §14 brought up to date.

**10.7 Small items closed (29 Oct).** README row counts re-verified read-only from S3 (Silver 5,606 rows to 2026-09-21; Gold 5,546 rows to 2026-09-18; the
proposal's 4,887/544 split is not reproducible, new counts are in section 6 and section 12 item 3); the old thesis checklist (section 12 item 1) marked superseded;
three of the four missing hourly ask months downloaded (see section 12 item 13). Deliberately not done: the S3 silver table was not refreshed (it is the friend's AWS account;
this belongs with the Lambda decision), and the static-training variants (`model_results_lstm_w*`) were not re-run (they are an older side experiment, flagged in Phase 7).

**Scoreboard of everything tried for next-day price forecasting (all on the 449-day window unless noted).** Nothing beats "tomorrow = today" on the
point forecast: LSTM, ridge, gradient boosting, ensembling, macro/cross-asset inputs, NFP/FOMC/CPI calendar, COT positioning, hourly cross-asset
features (own window), 5- and 20-day horizons. What does work is the evaluation and the calibrated interval (EWMA band, 89.5% coverage).

### Phase 11: the daily pipeline on AWS and a live track record (October 29, 2026)

The old Lambda design (price-level model exported to TFLite, container image) was dropped: it ran the model whose scale error Phase 7 had found, never
retrained, and its Docker build had failed. Replaced by a pipeline split the way a real data team would split it (ingestion / training / serving):

```
EventBridge Scheduler (cron 03:00 UTC = 10:00 Thailand, daily; it was 01:00 UTC until 2026-11-04)
   -> Lambda gold-lstm-forecast-daily-refresh (python3.12 zip + AWS SDK for pandas layer, 1 GB, ~1.5 s per run)
        1. fetch settled daily bars (Yahoo chart endpoint, stdlib urllib; range widens automatically after an outage)
        2. append new bars to bronze/streaming/xauusd_daily_incremental.csv (idempotent, keyed by date; Silver is never rewritten)
        3. data-quality checks -> gold/xauusd_daily/predictions/dq_report.json (critical failure: no forecast, run fails, metric Failure=1)
        4. features -> LSTM forward pass in plain numpy -> forecast + 90% EWMA band -> latest_forecast.json
        5. upsert forecast_log.csv: every earlier forecast is scored once its next close is known (the live track record)
        6. CloudWatch metrics (embedded metric format: NewRows, DQWarnings, Failure, ForecastsScored, RunSeconds) and a run log
   weekly, by hand or any scheduler with TensorFlow: automation/retrain.py  (Silver + stream log -> final LSTM -> s3://.../model/serving/serving_model.npz)
   dashboard: Method tab, "Live pipeline" expander (latest forecast, last run age, live track record); replaced by the header badge and the *Live record* section in Phases 12-13
```

Files: `daily/serving.py` (latest features, numpy LSTM, EWMA band), `daily/export_serving_model.py` (train the final model and export weights, 100 KB),
`daily/serving_artifacts/` (seed model + Keras reference outputs), `automation/lambda_function.py` (v2; the old one is `lambda_function_v1_price_level.py.old`),
`automation/deploy.py` (idempotent boto3: `plan` / `apply` / `teardown`), `automation/retrain.py`, `automation/iam_permissions_policy.json` (least privilege).

Verification: `daily/test_serving.py` (numpy forward pass matches Keras to about 1e-8; the band equals the evaluation band; features from a 200-row tail equal those from the
full history) and `automation/test_lambda_function.py` (end-to-end against an in-memory S3: idempotent replays, scoring of the previous forecast, critical data-quality failure
publishes nothing, unsettled bars dropped, model reloaded when the object changes). On AWS: the first run back-filled 27 missing bars (22 Sep to 28 Oct) and the prices matched the local
yfinance data on all 25 overlapping days (max difference 0.0).

Problems found while deploying (all fixed): the first test run on AWS failed because without `s3:ListBucket` S3 answers AccessDenied instead of NoSuchKey for a log that does not exist yet (the
least-privilege policy now grants ListBucket on the bucket, object access stays prefix-limited); `deploy.py` re-uploaded the seed model over the retrained one (it now only seeds a missing object);
a warm Lambda container kept the old weights (it now checks the object's ETag); an all-empty CSV column was read back as float and broke the second day's scoring (caught by the tests); at 08:00 Thailand
time a new COMEX session has started and Yahoo shows a partial bar for it (only bars whose session ended, 17:00 New York, are accepted).

Dashboard hero (29 Oct): the header forecast, range, sidebar "data through" line, volatility section and the history chart now use the Lambda's `latest_forecast.json` whenever its as-of date is newer than the forecast stored with the last walk-forward run (`daily/results/latest_forecast.json`); closes after that run are filled from the Lambda's streaming log, and the page falls back to the repository files if S3 cannot be read. The track record and metric cards still come from the walk-forward results.

Honest status: deployed and invoked manually on 2026-10-29; the schedule is ENABLED and its first unattended run is 2026-10-30 01:00 UTC. The live track record starts with the forecast for 2026-10-28 and
only fills in as days pass. Not done: CloudWatch alarms and the SNS topic (the IAM user `Auto` lacks SNS/CloudWatch permissions and the permissions were not widened); the empty ECR repo from September was deleted on request (2026-10-29). Cost: Lambda (a few seconds a day) and S3 are far inside the free tier. Operating it: `python3 automation/deploy.py apply` (safe to repeat),
`python3 automation/retrain.py` weekly, `python3 automation/deploy.py teardown` removes everything except the S3 data.

Schedule change (2026-11-04): the dashboard's live record had stopped moving because the 01:00 UTC run (08:00 Thailand) often found Yahoo's newest settled bar missing (the forecast lagged 1 to 3 days; the cause of the late bar was not proven). Two manual invocations after the COMEX close brought `latest_forecast.json` to as-of 2026-11-03 (forecast 3947.09, 90% range 3868.38 to 4027.40) and the live log to 3 scored forecasts (2 inside the range, 1 outside) plus one pending. The EventBridge schedule is now `cron(0 3 * * ? *)` UTC = 10:00 Thailand (changed in place with `aws scheduler update-schedule`; `deploy.py` and the dashboard wording match). Check after the next two mornings that `forecast_log.csv` gains one row per trading day.

### Phase 12: a readable, professional dashboard (October 30, 2026)

Request: make the UI much easier to read and as professional as possible. The old `app.py` had grown to 1055 lines, imported TensorFlow and awswrangler, and loaded the
Keras model and the S3 Gold table at start-up (~30 s cold start) to feed views of the superseded price-level pipeline. It was rebuilt rather than patched.

- **Structure.** `dashboard/app.py` (30 lines) wires `dashboard/ui/`: `data.py` (cached loaders: results from `daily/results`, the live forecast, track record and price history from S3),
  `logic.py` (the price verdict and the chip colour), `theme.py` (colours, `usd()`, one shared Plotly style), `header.py`, `sidebar.py`, `tab_track.py`, `tab_market.py` and `tab_reliability.py` (`tab_about.py` was removed on 17 Nov). The original-pipeline views (60-day price-level LSTM, AR(5) baselines, Gold-table feature explorer) were removed from the page; this README is their record.
  `requirements.txt` dropped tensorflow, awswrangler and scikit-learn and gained pyarrow. A warm page paints in about a second; segmented controls are `st.fragment`s, so changing a
  chart window re-runs only that chart.
- **Design system, native theming only.** `.streamlit/config.toml` sets the Inter font, rounder corners, a heading scale where `h1` is the forecast price, and one light theme with
  semantic colours (green/orange/red/blue chips); see the light-theme bullet below. Charts keep the project's colour-blind-safe hues (orange = actual price, blue = the model, grey = "tomorrow = today").
- **Information design.** Always visible: title and freshness badge, then the forecast card (price, change chip, 90% range, 1M/3M/6M/1Y chart with the forecast and its range bar), then
  three trust cards (typical error, range accuracy, beats "tomorrow = today"?) and a "How to read this" popover. Tabs: *Track record* (the live record first, then the 449-day backtest),
  *Data explorer* (price and volume with the test-period marker, volatility forecast vs. realised with a QLIKE comparison, 30-day volatility, daily changes, the model's own inputs),
  *How reliable?* (verdict callout, a three-row scorecard, price-error and range-accuracy charts, the secondary direction check and the full statistics table) and *About* (what the
  project is, the five pipeline steps, a "what we tried" chart of every experiment against "tomorrow = today" read from the result files, the method in plain words, a glossary).
- **Honesty kept.** The page leads with price error against the baseline; red/green chips only when p < 0.05; the "What we tried" chart shows Ridge and gradient boosting as clearly
  worse and everything else as within noise. Captions are generated from the data (verdict wording, which bars are better or worse), so they cannot drift from the numbers.
- **Layout findings (Streamlit 1.56 and 1.66.1, Plotly 6.8 and 7.1, checked at 390, 760, 900, 1024, 1280 and 1440 px).** In a horizontal container a `width="stretch"`
  child gets `flex: 1 1 128px` with `min-width: 16px`. `st.metric` cards are protected by their own text and wrap cleanly (3 across, then 2+1, then a column), but text-heavy cards and
  charts were squeezed instead of wrapping (900 px: three 130 px cards, two 180 px charts with rotated labels). So metric cards use stretch, the evidence is a scorecard of rows (number
  left, explanation right, wrapping under the number on a phone) and the two bar charts are full width. Plotly measures text once and caches it, so a web font that loads a moment
  later clipped legend entries; charts therefore use the system UI font (`CHART_FONT`). Streamlit remembers the sidebar state in `localStorage` (`stSidebarCollapsed-`): clear it when
  testing narrow widths, otherwise an open drawer hides the page.
- **Light theme only (30 Oct, later the same day).** The advisor prefers light mode. With both `[theme.light]` and `[theme.dark]` defined, Streamlit follows each visitor's system setting (an
  OS in dark mode got the dark page) and has no option to default to light. The config is now a single `[theme]` with `base = "light"` (sidebar colours in `[theme.sidebar]`). In the frontend
  a lone custom theme replaces Streamlit's presets and is applied whatever the OS or a remembered "Dark" preference says; verified on 1.56 and 1.66.1 with the OS emulated as dark and
  `localStorage` `stActiveTheme-/-v2` set to `"Dark"`. The cost is that viewers have no dark option. A running local server keeps the old theme after `config.toml` is edited: restart it.
- **Less text (advisor feedback, 30 Oct).** The advisor found the page too wordy for a general reader. Text visible without opening anything fell from about 1,300 to about 390 words
  (counted with Streamlit's `AppTest` on the same data, expanders and popovers excluded). Section and chart captions moved into `help=` tooltips (the small ? next to a heading, a metric or a chart
  title), each scorecard row keeps one short line with the detail in its metric tooltip, the "honest result" box on *About* was dropped because the verdict already sits on *How reliable?*,
  the five method expanders became one, and the pipeline steps are a few words each. Rule: a heading may carry a tooltip, a caption is at most one short line, and anything longer goes in a
  tooltip, popover or expander. Tooltips open on hover; tapping them on a phone was not tested (the owner said mobile does not matter).
- **Smaller text and a one-screen sidebar (30 Oct, owner feedback).** On the owner's screen (Safari on a 14-inch MacBook, viewport about 1512 x 857 px) the text looked too big and the sidebar
  needed scrolling (1,184 px of content in an 857 px window). `theme.baseFontSize = 14` scales every rem-based size down 12.5% (headings, metric values and spacing follow), and the forecast chart is
  280 px tall, so the title, the forecast, the chart and the three trust cards now fit the first screen. The sidebar holds only the credits: smaller logos, the school lines condensed to three, each
  credit block is one element with small bold names and the student IDs on their own lines, the status moved into the header badge's tooltip and the source link lives on *About*. It measures about
  555 px (Streamlit 1.56) and 578 px (1.66.1) below a 67 px header. The owner then found it a little small, so the sidebar is now a fixed 320 px wide (`st.set_page_config(initial_sidebar_state=320)`;
  1.66 would otherwise size it in rem, 263 px at a 14 px base, while 1.56 keeps 300 px), the logos are bigger, the title is a heading, and the names and student IDs are body size. It measures about 595 px:
  plenty of room in an 857 px window, a few pixels of scroll in a 650 px one. A local preview server must be restarted after editing `config.toml` or a `ui/` module (a plain reload showed stale code several times).
- **Plain wording, no overlapping chart text (30 Oct, owner feedback).** One name per idea everywhere: *baseline* ("tomorrow = today", also called the naive baseline), *this model* (the LSTM shown in the
  header), *typical error*, *range hit rate*. Vague labels were replaced ("Used here", "No clear edge", "Range accuracy", "half-width"); the backtest reads "Actual vs. forecast price" with the legend "Actual" / "Forecast" and "Actual − Forecast ($)" for the
  gap bars; the three cards read "Average miss per day", "Price stayed in the range" and "Better than a simple guess?" (the baseline is named in the tooltip), and the volatility cards "Typical daily move", "Yearly pace" and "Fear gauge (VIX)"; model and input labels are spelled out ("LSTM + other markets", "Ridge (linear model)", "Follow last move"). Bar values now sit in the labels under the bars and reference lines are drawn behind the bars, so no text collides; every chart is checked with a script that compares the bounding boxes of all
  its text. A first attempt also deleted the statistics, model comparisons, volatility and input charts to suit "general users"; the owner objected (too many charts gone), so that commit was reverted and all
  charts are back. Do not remove charts without being asked.
- **Verification.** No exceptions on any tab at the widths above; no horizontal overflow (script-checked per tab); Streamlit 1.66.1 in a fresh venv installed from the new
  `requirements.txt` rendered identically; live URL checked after the push.

### Phase 13: page restructure, clearer tabs and a fuller scorecard (November 9-17, 2026)

Owner request (nine points, plus "make a checklist"): a short briefing in the header; the forecast day much more prominent; the 90% range and its hit rate merged into one block and the
hit rate renamed; metrics shown with their technical names (MAE, RMSE, ...) and a new home for the reliability numbers; tab names that say what is actual and what is predicted;
the backtest first; actual values beside the forecasts in the "Price history and forecast" chart; visualisations on top and the explanations separated.

- **Header briefing (removed).** The owner asked for a 2-3 line briefing under the title on 9 Nov (what the model does, that every forecast is saved before the day happens and scored against the real close, and that it is
  judged against "tomorrow = today"). On 14 Nov the owner had its first sentence ("An LSTM neural network forecasts the next trading day's gold price (XAU/USD) and a 90% range") removed, and then the two remaining lines as well.
  The header is now just the title and the status badge; the instrument name (XAU/USD) is no longer on the main page. Do not add a briefing back unless asked.
- **Forecast day (placement revised twice).** The first version put a calendar tile (FORECAST FOR / MONDAY / 9 / NOVEMBER 2026) to the left of the price; the owner found it cluttered there. The second put "Forecast for
  Monday 9 November 2026" (calendar icon, the date in blue) in the top-right corner of the card, opposite a card caption. Three placements were compared in a scratch app for that step (heading top-right, a big blue date above
  the range box, a small badge top-right). On 14 Nov the owner asked for the line to sit right above the forecast number, so it is now the first line of the price block ("Forecast for Monday 16 November 2026", then the
  price and the change chip) and the range box stays on the right; the card has no top row any more. The line "vs. the last close $3,887.60 · Fri 13 Nov" under the change chip was removed the same day (owner request); the last
  close it is measured from is in the chip's tooltip. "goal 90%" is also gone from the live-record card's chip; it remains as the chip of the range-reliability row on *How reliable?*, where the chip carries the green/orange verdict.
- **Range and its reliability are one block.** "RANGE" (first "LIKELY RANGE (90%)"; the owner had "LIKELY" and "(90%)" removed on 14 Nov, the tooltip still says 9 days out of 10) with the range and, under it, a badge
  "Range reliability 89.5%" (first with "goal 90%", also removed). The `?` next to "RANGE" was removed too (owner, 14 Nov); its sentence ("the range is where the close is expected to land on 9 days out of 10") moved
  into the hover tooltip of the green reliability badge, together with 348 of 389 test days and the live count. Streamlit draws no `?` for a markdown line that is only a badge, so that tooltip shows on hovering the badge itself. The old "Price stayed in the range" card is gone. One name from now on: **range reliability** (it replaces "range hit rate" and "price stayed in the range" on every tab, chart title and the glossary).
- **Scorecard.** "Accuracy · 449 backtest days" with four metric cards that carry their technical names: MAE (average miss), RMSE, error % (MAPE) and out-of-sample R². (First version: each card also had a chip "simple guess $46.63" and a one-line note; both were removed on 14 Nov, see "Cards stripped".)
  The verdict was a coloured pill beside the title ("Not clearly better than a simple guess", green or red only when p < 0.05; a fifth card had wrapped alone onto a second row at 1280 px, hence the pill); the owner had the pill removed
  on 14 Nov, see "Pill removed".
  Rank correlation (IC) stays on *How reliable?*. The detailed evidence for the numbers stays on that tab (its rows use the same names).
- **One-line notes under the accuracy metrics (owner request, same day).** Each accuracy card has a short grey line under the number saying what it measures and when to use it: MAE "Average miss in dollars. Best for the
  everyday error.", RMSE "Like MAE, but big misses count extra. Best for spotting bad days.", error % "Miss as a share of the price. Best for comparing price levels.", R² "Does it explain the daily moves? 0 = no better
  than average." (the live-record and volatility cards got the same treatment first and lost it again on 14 Nov, see "Text trimmed"). The texts live in `ui/cards.py` (`NOTES`, `metric_card()`, `card_row()`); a card is a bordered container holding an `st.metric`
  and a caption, and the row uses `vertical_alignment="distribute"` so every card has the same height even when one note wraps to an extra line. The cards stay in one row down to about 140 px each (a 759 px window
  with the sidebar collapsed) and the notes simply wrap; tooltips still carry the longer definitions.
- **Chart titles (owner request, same day).** The title of "Price history and forecast" was a small grey caption and several other chart titles were bold body text, so they were easy to miss. Every chart title is now
  a section-size heading (h3, 17.5 px, weight 600) through one helper, `theme.chart_title(text, help=...)`: the header chart, the accuracy scorecard, the new live-record chart ("Live forecasts vs. the actual close", it had no title),
  "Actual daily moves vs. the model's limit", "Long-run volatility", "Average miss (MAE) by model" and "Range reliability by method" (was "... by way of drawing the range"). The panel titles inside the figures ("Actual vs. forecast price",
  "Actual − Forecast ($)") are bold 13 px dark grey instead of 12 px light grey (`theme.panel_title`). The theme's heading sizes were not touched because the sidebar title is an h3 and the sidebar is tuned to fit one screen.
  `:large[...]` is not a Streamlit markdown directive (it renders as literal text); a blue accent bar and a chart icon before the title were compared in a scratch app and dropped to keep one heading style on the page.
- **Text trimmed (owner, 14 Nov: "the text still looks cluttered, remove what is not relevant").** Rule applied: no line may repeat what a title, a tab name, a chip or a tooltip already says. Removed: the "Research project" badge
  (the disclaimer stays); the card caption "GOLD CLOSING PRICE FORECAST (XAU/USD)" (the status badge "Live · updated ..." briefly moved into its place in the forecast card and is back beside the title since the briefing was removed);
  "Read it as about today's close, within the range" (the reliability tab says it; the caption keeps only "Research only, not trading advice."); the one-line captions under the first two tab names; the notes on the live-record
  and volatility cards (their chips and tooltips remain; the four accuracy cards kept a note until the owner removed those too, see "Cards stripped"); the "Daily changes" statistics line (moved into its tooltip); the *About* intro paragraph, which repeated the header
  briefing and the sidebar; and filler in the reliability tab (the verdict lost a clause, the R² row shows the rank correlation instead of an explanation). The header briefing was cut to shorter lines first (later removed, see the Header briefing bullet), the live-chart legend
  reads "Forecast + 90% range / Actual (inside range) / Actual (outside range) / Next forecast", and the "How to read this" popover no longer mentions the old fifth card. Text counted by script on the rendered page (charts, tables and
  collapsed expanders excluded, icon names included): *Forecast vs. actual* 310 to 250 words, *Actual market data* 364 to 287, *How reliable?* 370 to 333, *About* 846 to 790 (about 450 of those sit in collapsed expanders). No chart or
  section was removed.
- **Cards stripped (owner, 14 Nov: selected the four notes and the four "simple guess" chips and said "remove all of it").** The four accuracy cards now show only the label (with its tooltip) and the number. The score of the simple guess
  "tomorrow = today" moved into each card's tooltip (MAE $46.63, RMSE $68.94, error % 1.14%, R² −0.001), and the title's tooltip said how the model compares (all of these simple-guess sentences were cut again on 17 Nov, see "Hero chart, tooltips and disclaimer"). The same chip was removed from the live-record cards
  ("simple guess: $49.57" and the "80%" under the range reliability; both are in the tooltips now). The volatility cards keep their unit chips ("per day", "over a year"). `ui/cards.py` lost its `NOTES` and the `note` argument.
- **Pill removed (owner, 14 Nov: selected it and said "remove this").** The coloured verdict pill "Not clearly better than a simple guess" beside "Accuracy · 449 backtest days" is gone. What it said now sits in the tooltip of
  the title (the simple guess misses by $46.63 a day, this model by $46.83, +$0.20, p = 0.38, a difference counts as clear only below 0.05) and in the card tooltips. The plain-language verdict ("Not better than a simple guess. Read the
  forecast as about today's close. The range, however, is trustworthy.") is now only on *How reliable?*, so the first screen no longer states that the model does not beat the baseline; the owner chose that. The popover's pill bullet went too.
- **Panels, then a navigation bar (owner, 14 Nov).** First the owner selected the four tab labels and asked for windows that hide the content until clicked, all content kept: the tabs became four stacked `st.expander`s. Then the owner
  selected the four panel headers and asked for them in one row, so they are now a navigation bar: one full-width `st.segmented_control` (icons and bold labels, same names) placed under the accuracy cards. Nothing is selected when the
  page loads, so no section is shown; clicking a button shows that section below the bar, clicking another switches, clicking the active one again hides it (one section at a time, which is the difference from the expanders). Only the selected
  section's code runs, so reruns are lighter than with tabs or expanders. The sections' content is unchanged and the expanders inside them (volatility models, side check, all statistics, the numbers, method, glossary) work as before. The
  control is a plain script rerun, not a fragment; the section stays selected in `st.session_state["section"]`. Checked on Streamlit 1.56 and 1.66.1 by clicking every button (waiting for the rerun to finish): the four buttons share one row,
  each section's charts fill the full width with no text overlap and no exception, the second click hides the content; `AppTest` runs the page once per section (the section is primed through `session_state`), online and offline.
- **Hero chart, tooltips and disclaimer (owner, 17 Nov).** (1) The header chart is now titled "Actual vs Forecast" (it was "Price history and forecast"; the owner first asked for "Actual vs Predict" and then for "... vs Forecast") and opens on **1M** instead
  of 3M. (2) The label "Live forecasts start" and then the dotted vertical divider itself were removed; the chart's tooltip says where the real forecasts begin ("Forecasts before 29 Oct are replayed on past data; from then on they are real
  forecasts, saved before each day", the date is read from the live log). (3) The simple-guess sentences ("A simple guess (repeating the last close) misses by $46.63" and the like) were cut from **every** header tooltip: the four accuracy cards,
  the Accuracy title and the popover, and from the live-record MAE tooltip; the R² tooltip was rewritten as a test-score picture ("Like a test score for catching the daily ups and downs. 1 means every move was predicted; 0 means none was caught, no better
  than assuming a typical day." The owner then cropped out the rest of the first draft ("below 0 means worse than that. Measured only on days the model had never seen."). The header therefore no longer says anything about the simple guess; the comparison is only in *How reliable?* (evidence row,
  verdict, the model chart) and in the *About* glossary; the owner chose that. (4) The line under the accuracy cards now reads "Made only for a Computer Engineering senior project. This is not trading advice." and the *About* footer says
  the same (before: "Research only, not trading advice.").
- **Popover and divider moved (owner, 17 Nov, with sketches).** The "How to read this" popover now sits at the right end of the "Accuracy · 449 backtest days" title row (before: under the cards, opposite the disclaimer, which now stands alone). In the
  forecast card the line between the price row and the chart was moved up: the row and the line live in a nested `st.container(gap=None)`, so the space above the line is just the line's own margin and it sits as far from the row as from
  the chart title below it (before the line was about 40% further from the row than from the title). The Browser pane's interaction tools were unavailable that day, so the spacing was reasoned from Streamlit's margins and checked only by
  `AppTest` (no exception, the popover and the title share one flex container on 1.56 and 1.66.1), not measured on screen; if the owner still finds it too loose, the next option is a bordered chart container instead of a divider (the line's own margin cannot be changed without custom CSS).
- **Evidence rows explain themselves (owner, 17 Nov, one row at a time).** On *How reliable?* the one line beside each number in "The evidence" is now what the metric is, not a fact about it: MAE "On a typical day the forecast was about this many dollars away
  from the real close." (it was "A simple guess, repeating the last close, misses by $46.63"), R² "A test score for catching the daily ups and downs: 1 means every move was predicted, 0 means none was." (it was the rank correlation line) and
  range reliability "How often the real close stayed inside the range. The aim is 9 days out of 10." (it was "The range held on 348 of 389 test days"). The facts that left the visible line moved into each row's tooltip: the simple guess's miss
  ($46.63 a day, 1.14% of the price) with the p-value, the rank correlation (+0.006, p = 0.90) and 348 of 389 test days. The chips ("+$0.20 vs. baseline", "No predictive power", "goal 90%") are unchanged.
- **"(this model)" stands out; LSTM first (owner, 17 Nov).** In "Average miss (MAE) by model" the LSTM bar is now the first bar (then the simple guess, Ridge and gradient boosting) and the words "(this model)" under it are bold and in the model's blue;
  the same emphasis is on the first bar of "Range reliability by method" ("LSTM (this model)") and on the LSTM label of the side-check chart. The owner then asked for the name and the number to stand out too, so on the model's bar the name ("LSTM") and the value ($46.83, 89.5%) are bold blue at 14 px
  and "(this model)" is bold blue at the normal size. One helper does it, `theme.highlight(text, size=None)` (Plotly label markup: bold `<span>` in the model
  blue), so any new chart label for this model should use it; hover texts keep plain words (`tab_reliability._plain`). The Browser pane's page tools were unavailable that day, so the markup was checked on the figure objects, not on screen:
  if a Plotly build ever showed the tags as text, drop the `<span>` and keep `<b>` (the owner's screenshot later showed the 14 px bold blue rendering correctly: $46.83 in bold blue, larger than the other values).
  The owner then asked for "LSTM" instead of "Recent errors" on the first bar of the range chart. "Recent errors" had named the *method*, not the model: the range is the forecast +/- 1.645 x an exponentially weighted average (lambda 0.94, recent days count more)
  of the model's own past errors, while the bar "All past errors" takes the 5th and 95th percentile of every past error with all days weighted equally (85.9% coverage, it adapts slowly). The chart tooltip now says "this model's range (sized from its recent errors)";
  the last bar, first labelled "Recent errors + VIX", was then renamed "LSTM + VIX" (owner, 17 Nov): it is the same range with the VIX added to the width model, not the VIX fed into the LSTM, and the tooltip says so.
- **Tabs.** *Forecast vs. actual* (the model's forecasts next to what happened), *Actual market data* (real prices and what can be measured from them), *How reliable?*, *About the project* (removed on 17 Nov, see the end of this phase). (They first started with a one-line caption each; removed on 14 Nov, the tab names and
  section titles say the same.) Inside *Forecast vs. actual* the **backtest comes first** with a grey badge "Simulated on past data"; the **live record** follows with a green badge "Real forecasts", so the
  reader can tell the replayed past from forecasts that were saved before the day happened.
- **Actual vs. forecast in the header chart.** The orange actual line now has the blue forecast made the day before (dashed, one dot per day on short windows), the 90% range of each forecast as a light band,
  the new forecast with its range bar, and a short panel of bars "Actual − Forecast ($)" (red: the price ended above the forecast, blue: below). Forecasts are placed on the day they are *for*
  (the old backtest chart put them on the day they were made, one trading day early); the backtest chart uses the same alignment now. A dotted vertical line marked where real forecasts start (28 Oct; line and label removed on 17 Nov);
  before it the forecasts come from the backtest. The band for backtest days is forecast x e^(+-1.645 sigma) with the EWMA sigma stored in `volatility_forecast.csv`; for live days it is the saved range.
  New helper `data.forecast_history()` builds the table (backtest rows plus live-log rows, live wins on the same day) and reproduces the backtest MAE of $46.83.
- **Live record chart (new).** Under the three live cards: each real forecast as a diamond with its 90% range bar, the actual close beside it (a circle inside the range, a cross outside) and the next forecast as an
  open diamond. With the table below it, the live record is now visual as well as numeric.
- **Division of content.** The header and the first three sections hold only results and charts; *About the project* holds everything that explains what was built (intro, the five pipeline steps, "What we tried",
  the method, the glossary, links). Nothing was deleted: the "What we tried" chart and every other chart are where they were. Superseded on 17 Nov: *About the project* was removed at the owner's request and its content moved to the README (see below).
- **Live pipeline check.** With the schedule moved to 03:00 UTC (10:00 Thailand, 4 Nov) the unattended runs on 5, 6 and 7 Nov each added the previous day's bar and a forecast (the 8 and 9 Nov runs found nothing new: a weekend and a Monday morning), so the live log has one row per trading day since 4 Nov: 6 scored forecasts
  (4 inside the 90% range, 2 outside; MAE \$56.32 vs. \$57.40 for "tomorrow = today") and the forecast for Monday 9 Nov waiting. Six days say nothing about accuracy yet.
- **Verification.** No exceptions and no text overlap in any chart on any tab (script over every `.main-svg` text box) on Streamlit 1.56 and 1.66.1 at 1512 x 857, and the first revision of the header, scorecard and tabs also
  at 1366 x 650, 1280 and 1024 px. The sidebar still fits 857 px without scrolling. After the second revision (forecast day top-right, note cards; the later moves of the day above the price and the removal of the briefing were checked on 1.56 and 1.66.1 at 1512 x 857 with no exception, no chart text overlap and the sidebar fitting) the page was re-checked on 1.56 at 1512 x 857 and 759 px (sidebar collapsed) and on 1.66.1 at
  1268 px (sidebar open): no exception, no overlap, no horizontal overflow; the four scorecard cards are 161 px wide at 1268 px and 142 px at 759 px, equal in height, with the notes wrapping to three or four lines. The browser pane
  kept changing width while testing, so 1024 px was not re-measured for the second revision. An offline run (S3 unreachable) and an online run of the whole page under `AppTest` raise nothing.
- **The verdict box, the side check, the statistics table and four tooltips leave *How reliable?* (owner, 17 Nov).** The owner asked to remove the heading "Can this forecast be relied on?" with the verdict box under it, the two expanders at the bottom of *How reliable?* ("Side check: does it get up or down right?" and "All the statistics, model by model")
  and the four `?` tooltips of the evidence block (the heading and the MAE, R² and range-reliability rows), and to keep their content in the README. The section now starts with the evidence block and ends with the range chart; the one-line explanation beside each evidence number stays.
  The code that only served the removed parts is gone too (`_verdict`, `_direction`, `_details`, `MODEL_LABELS`, `logic.direction_verdict`, `logic.pct_miss`, `logic.PHRASE`). The content is recorded in "Content removed from the dashboard" below.
  The table's CSV download (`daily_walk_forward_metrics.csv`) went with it; the backtest section keeps its own predictions CSV. Direction accuracy stays a labelled secondary check, now in the README only.
- **The *About the project* section leaves the page (owner, 17 Nov).** The owner asked to remove the whole section (the five pipeline steps, the "What we tried" chart with its numbers, "The method in plain words", the glossary, the source-code and README buttons and the footer line)
  and to keep the information in the README. The navigation bar now has three sections. `tab_about.py` and the loaders only it used (`data.horizons`, `data.feature_rows`, `data.FEATURE_SETS`, `data.REPO_URL`) are deleted.
  The content is recorded in "Content removed from the dashboard" below, rebuilt from the result files (the table holds the same data as the "What we tried" chart). The page no longer links to the repository or to this README, because the two buttons lived only in that section;
  the "Made only for a Computer Engineering senior project, not trading advice" line stays under the accuracy cards.
- **A short project description is back under the title (owner, 17 Nov).** The explanation of the project that used to sit under "Gold Price Forecast" (three lines on 9 Nov, two lines until 14 Nov, then removed) returns, first shortened to one line of 16 words ("An LSTM neural network forecasts the next trading day's gold price (XAU/USD) with a 90% range."), then, at the owner's request ("a bit longer"), to two lines of 30 words. A longer version that explained the techniques (five lines with the 20-day window, the 365-day training, the walk-forward test and the EWMA range method, read from the result files) was rejected by the owner as too long;
  the concise wording came back and the disclaimer was appended as a third line: 1. "An LSTM neural network forecasts the next trading day's gold price (XAU/USD) with a 90% range." 2. "Live forecasts are saved before the day happens, then scored against the real close." 3. "Made only for a Computer Engineering senior project. This is not trading advice."
  It names the model (the glossary left with the About section), says *live* because the replayed forecasts were not saved beforehand, and does not mention the simple guess (owner, 14 and 17 Nov). The technical details stay in the README. The three-line original read: "An LSTM neural network forecasts the next trading day's gold price (XAU/USD)
  and a 90% range around it. Every forecast is saved before the day happens, then scored against the real close. It is judged against a simple guess, tomorrow = today: the model has to beat that to be useful."
- **The `?` of the header chart title leaves (owner, 17 Nov).** "Actual vs Forecast" has no tooltip any more; what it said is kept in "Content removed from the dashboard" below. The *Forecast vs. actual* section still tells replayed from real forecasts with its two badges
  ("Simulated on past data", "Real forecasts").
- **Backtest filter moved to the right (owner, 17 Nov).** The window filter (1M / 3M / 6M / All) of *Backtest · 449 test days* moved from its own row under the title to the right end of the title row, where the header chart's filter already sits;
  the title and the grey "Simulated on past data" badge stay together on the left. It also saves one row of height. The price-history and indicator filters in *Actual market data* still sit on their own rows.
- **The disclaimer joins the project description (owner, 17 Nov).** "Made only for a Computer Engineering senior project. This is not trading advice." is now the last line of the description under the title (three short lines in all) instead of a caption under the accuracy cards,
  which now carry no caption. The description is a constant text, so the disclaimer shows even when the result files are missing.
- **Dashboard closed (owner, 17 Nov).** The owner declared the dashboard finished for now and asked for the README to be completed. The final structure is in §10 ("Current dashboard"); the rules that keep it that way are in `AGENTS.md`.
  What remains lies outside the page: the weekly retrain, the live-log hardening and the paper (see Status and next steps and §15).

#### Content removed from the dashboard (November 17, 2026)

The owner asked for these parts to leave the page and to keep their content here. Everything is rebuilt from `daily/results/walk_forward_results.json` and `regression_metrics.json` (449 test days, 2025-01-09 to 2026-10-23);
the code is in git history (last commit that still had it: `618b274`).

**The heading "Can this forecast be relied on?" and the verdict box under it** (top of *How reliable?*). The box had three wordings, picked by the Diebold-Mariano test of the MAE against the simple guess (p below 0.05 and a clearly lower or higher MAE; otherwise "same"):

- better: "**A measurable edge.** The average miss (MAE) is clearly lower than a simple guess (repeating the last close)."
- worse: "**Worse than a simple guess.** The average miss (MAE) is clearly higher than just repeating today's price."
- same (the one shown for the 449-day window, p = 0.38): "**Not better than a simple guess.** Read the forecast as *about today's close*. The range, however, is trustworthy."

**Side check: "does it get up or down right?"** (an expander at the bottom of *How reliable?*; direction is a classification metric, so it was only a labelled side check).

Right on **51.9%** of days; always saying “up” scores **55.2%**. The model cannot be told apart from that (p = 0.15).

Tooltip: Calling up or down is a yes/no question, not how a price forecast is judged, so this is only a side check. The model says “up” on about nine days in ten, so its score mostly follows how often gold rose. 95% interval: 47.3% to 56.5%; the bars showed the same intervals.

Bars it showed (direction accuracy with its 95% interval; the dashed line was the always-up score):

| Model | Direction right | 95% interval | vs. always up (p) |
|---|---|---|---|
| Always guess up | 55.2% | 50.6% to 59.8% | 1.000 |
| Follow last move | 45.2% | 40.7% to 49.8% | 0.000 |
| Ridge (linear) | 49.0% | 44.4% to 53.6% | 0.009 |
| Gradient boosting (trees) | 52.6% | 47.9% to 57.1% | 0.255 |
| LSTM (this model) | 51.9% | 47.3% to 56.5% | 0.155 |

**All the statistics, model by model** (the second expander).

| Model | MAE (\$) | Error % (MAPE) | RMSE (\$) | vs. baseline (p) | R² out-of-sample | Rank correlation | Direction right (%) | vs. always up (p) | AUC |
|---|---|---|---|---|---|---|---|---|---|
| Recent average return | \$46.36 | 1.136% | \$69.07 | 0.941 | +0.000 | -0.027 | 55.2% | 1.000 | - |
| Ridge | \$47.87 | 1.171% | \$70.35 | 0.023 | -0.043 | -0.103 | 49.0% | 0.009 | - |
| Gradient boosting | \$50.68 | 1.237% | \$75.05 | 0.000 | -0.183 | -0.034 | 50.3% | 0.041 | - |
| Gradient boosting (up/down) | - | - | - | - | - | - | 52.6% | 0.255 | 0.498 |
| LSTM (this model) | \$46.83 | 1.147% | \$69.45 | 0.385 | -0.011 | +0.006 | 51.9% | 0.155 | 0.463 |
| Follow last move | - | - | - | - | - | - | 45.2% | 0.000 | - |

Caption: p below 0.05 = a clear difference. 449 test days, 2025-01-09 to 2026-10-23; the baseline (a simple guess: repeat the last close) has MAE \$46.63. The expander also offered the table as `daily_walk_forward_metrics.csv`; the same numbers are in `daily/results/walk_forward_results.json` and `regression_metrics.json`.

**Tooltips of the evidence block** (the heading and the three rows; the one-line explanation beside each number stayed).

- *The evidence* (heading): Based on 449 test days (2025-01-09 to 2026-10-23); small differences are within noise.
- *MAE · average miss*: A simple guess (repeating the last close) misses by \$46.63 a day, 1.14% of the price. The model cannot be told apart from it (p = 0.38; below 0.05 counts as a clear difference).
- *R² · predicts moves?*: Out-of-sample R²: 0 means no better than a simple average, below 0 is worse. Rank correlation with the actual moves: +0.006 (p = 0.90).
- *Range reliability*: The real close stayed inside the range on 348 of 389 test days. Ideal: 90%. Its width, \$223, is the same as a no-change range (\$221): the model does not narrow the uncertainty.

**Tooltip of the header chart "Actual vs Forecast"** (the `?` next to its title; the date was read from the live log).

- Orange line: the real closing price each day. Blue dashed line: the forecast made the day before, with its range shaded light blue. Blue diamond: the new forecast and its range. Bars at the bottom: how far off each forecast was (red: the price ended higher than forecast, blue: lower). Forecasts before 29 Oct are replayed on past data; from then on they are real forecasts, saved before each day.

**The *About the project* section** (the fourth section of the navigation bar; it only explained what was built and tried). Its parts in the order shown, with every number rebuilt from the same result files:

**How the forecast is made.** A fully automatic pipeline on AWS (Lambda and S3) runs every day, without anyone pressing a button.

1. **Collect**: the latest gold price from Yahoo Finance, daily at 10:00 (Thailand time)
2. **Check**: automatic data-quality checks
3. **Forecast**: an LSTM reads the last 20 days (retrained weekly)
4. **Add the range**: a 90% range from the model's recent errors
5. **Publish and score**: saved to Amazon S3, scored when the real close arrives

**What we tried.** Every idea was judged the same way: the MAE (average miss) against a simple guess (repeating the last close), on the same test days. The same holds for the range: five ways of drawing it, a GARCH model, the VIX and a network trained on the range itself all failed to beat a simple weighted average of recent errors.

| Group | Experiment | MAE | Baseline MAE (simple guess) | Difference | p-value | Verdict | Test days |
|---|---|---|---|---|---|---|---|
| Main model | LSTM (this model) | \$46.83 | \$46.63 | +0.4% | 0.38 | No clear difference | 449 |
| Other models | Ridge (linear model) | \$47.87 | \$46.63 | +2.7% | 0.02 | Worse | 449 |
| Other models | Gradient boosting (trees) | \$50.68 | \$46.63 | +8.7% | 0.00 | Worse | 449 |
| Extra inputs | LSTM + macro and cross-asset prices | \$46.80 | \$46.63 | +0.4% | 0.46 | No clear difference | 449 |
| Extra inputs | LSTM + NFP release days | \$46.74 | \$46.63 | +0.2% | 0.59 | No clear difference | 449 |
| Extra inputs | LSTM + FOMC and CPI release days | \$46.38 | \$46.63 | -0.5% | 0.97 | No clear difference | 449 |
| Extra inputs | LSTM + CFTC positioning | \$46.62 | \$46.63 | -0.0% | 0.80 | No clear difference | 449 |
| Longer horizon | LSTM, price 5 trading days ahead | \$109.50 | \$109.19 | +0.3% | 0.43 | No clear difference | 444 |
| Longer horizon | LSTM, price 20 trading days ahead | \$216.95 | \$219.40 | -1.1% | 0.59 | No clear difference | 429 |

Chart: one bar per experiment showing the error against the baseline (left = better). Grey: no real difference · Red: clearly worse. Nothing came out clearly better.

Longer-horizon p-values are adjusted for overlapping forecasts.

**The method in plain words.**

- **Why a change, not a price?** The first version predicted the price and broke when gold reached levels it had never seen (error about \$640 against about \$51 for a simple guess that repeats the last close). Now the model predicts the day's percentage change and converts it: forecast = today's close × e^(predicted change).
- **Why is every test on unseen days?** The model is retrained every 7 trading days on the latest 365 days and judged only on the days that follow. Inputs use past data only; automated tests check it.
- **How is the 90% range computed?** Forecast ± 1.645 × the typical recent miss (a weighted average of past errors, decay 0.94). It uses only errors known before each forecast, so its 89.5% range reliability is honest.
- **Why isn't volume an input?** 3% of days show zero or missing volume in Yahoo Finance's data, so every input comes from the closing price alone.
- **Why is up/down only a side check?** A price forecast is judged by how far off it is, not by a yes/no call. The model says “up” on 88% of test days, so its score mostly follows how often gold rose.

**Glossary.**

- **LSTM.** A neural network built for sequences. Here it reads the last 20 days and predicts the next day's price change.
- **MAE (average miss per day).** Mean absolute error: the average gap between the forecast and the actual close, in dollars. Lower is better.
- **RMSE.** Root mean squared error: like MAE, but big misses count extra, so it shows how bad the worst days were. Lower is better.
- **MAPE (error %).** Mean absolute percentage error: the average miss as a share of the price.
- **Simple guess (baseline).** Assuming the price stays the same as the last close; also called the naive or no-change forecast. A price forecast is only useful if it clearly beats this.
- **p-value.** How likely a difference this large would be to appear by luck alone. Below 0.05 counts as a clear difference; above it, the two are treated as equal.
- **R² (out-of-sample).** The share of the daily price move the forecast explains. 0 means no better than a simple average; below 0 means worse.
- **Rank correlation (IC).** Whether days with a higher forecast really had higher actual moves. 0 means no relationship, 1 a perfect match.
- **Range reliability (90% range).** The range is where the close should land 9 days out of 10; reliability is how often it really did in testing. 90% is ideal.
- **Walk-forward test.** Retrain on the past, test on the days that follow, then move forward and repeat, so the model never sees the days it is judged on.
- **EWMA.** Exponentially weighted moving average: recent values count more than old ones. Used to size the range.
- **VIX.** The “fear gauge” of the US stock market. Tested as an input; it did not help.

**Footer.** Two buttons, "Source code" (https://github.com/aphiwat-data/lstm_dashboard) and "Full write-up (README)" (https://github.com/aphiwat-data/lstm_dashboard/blob/main/README.md), and the line: "Made only for a Computer Engineering senior project, not trading advice · Prices: Yahoo Finance (GC=F)".

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
13. Check how a layout primitive behaves at several widths before building on it: Streamlit's stretch children squeeze rather than wrap unless their content has a minimum size.
    Generate prose from the data, so wording cannot drift from the numbers.
14. Never delete what the owner did not ask to delete. A mass removal of charts had to be restored from git; since then every removal needs an explicit request, and the removed content moves to the README.
15. A check on a second runtime must prove which runtime ran. A scratch virtual environment silently fell back to Streamlit 1.56, so several "1.66.1" runs were not what they claimed; print the version in the check itself.
16. Say what is live and what is replayed. "Every forecast is saved before the day" is true only of the live forecasts; the replayed (backtest) forecasts were not saved beforehand.
17. Give every manual step of an unattended system an owner and a reminder: the weekly retrain was documented but never scheduled, and the serving model is weeks old.
18. An audit trail that can be overwritten is not an audit trail: the live log needs versioning or an external timestamp before it is cited as evidence.

### Where things live
| Path | What |
|---|---|
| `notebooks/`, `automation/` | Original pipeline (notebooks 01-06, the price-level model, superseded) and the live daily pipeline: `lambda_function.py` (deployed 2026-10-29), `deploy.py`, `retrain.py`, the IAM policies and the tests |
| `daily/` | Corrected daily dataset builder, causality tests, walk-forward evaluation, regression-first metrics (`regression_metrics.py`), multi-day walk-forward (`walk_forward_horizon.py`), the interval experiments (`interval_experiment.py`, `lstm_quantile.py`, `interval_vix.py`), the volatility comparison (`volatility_experiment.py`) and the serving core of the live pipeline (`serving.py`, `export_serving_model.py`, `serving_artifacts/`). `results/` (the main price-only run) feeds the dashboard; `results/with_macro/`, `with_macro_calendar/`, `with_macro_calendar_full/` and `with_cot/` are the Phase 9 feature-search runs on the same rows; `results/snapshot_2026-09-18/` keeps the earlier 424-day outputs of all five |
| `hourly/` | Hourly experiments: downloader, builder, tests, baselines, models, bid-vs-mid check and the cross-asset test (`results/with_aux_mid/`); `results/summary_hourly_findings.png` is the one-page summary |
| `dashboard/app.py`, `dashboard/ui/` | Streamlit app (deployed on Streamlit Community Cloud from `main`): `app.py` is the entry point, `ui/` the header, sidebar, metric cards, one module per section (three sections), cached data loaders and the chart style; `dashboard/assets/` holds the MFU and ADT logos and `.streamlit/config.toml` the theme |
| `docs/TROUBLESHOOTING.md` | Chronological error log for the original pipeline |
| `AGENTS.md` | Rules for any coding agent: public-repo privacy, metric policy, one results window, the owner's dashboard rules, research rules for the paper |
| S3 bucket `gold-lstm-forecast` (`ap-southeast-2`), not in git | `bronze/`, `silver/`, `gold/` (data, features, the live pipeline's predictions) and `model/serving/serving_model.npz` (the live model) |

### Status and next steps
*As of 2026-11-17.* See the **Handoff** block at the top and §12 for the tracked open items. In short:
- **Modelling: closed as a negative result.** Every lever tried (models, five feature sets, horizons, daily and hourly data, interval methods) is level with the simple guess on price error; the honest write-up is a well-instrumented negative result with a reliable 90% range.
- **Dashboard: finished (owner, 17 Nov).** The final structure is in §10 and the rules that keep it that way are in `AGENTS.md`.
- **Live pipeline: running unattended since 2026-10-30** (11 forecasts scored by 2026-11-17). To do: run `python3 automation/retrain.py` (the model was last written on 2026-10-29), make the live log tamper-proof if it will be cited (S3 versioning with Object Lock, or a daily git commit of the log)
  and correct the *Live record* tooltip, which still says nothing can be edited afterwards.
- **Paper: an IEEE-format paper is the next step (owner, 2026-11-17).** The advisor wants a publication; the owner will work with this README and Claude as research helpers. §15 is the research dossier: the claim the evidence supports, every number with its source file, the method in reproducible detail,
  threats to validity, extra experiments to strengthen the paper, a checked bibliography, an IEEE venue shortlist, a section-by-section writing plan and the rules for working with Claude.
- **Risk track (E6, done 2026-11-24):** volatility is predictable (positive R²) and HAR-RV on hourly realised variance beats the shipped EWMA over 2010-2026; the price R² stays negative. Results in §15.5, protocol in §15.8; follow-ups: intervals built on the HAR-RV volatility, E3, E7-E8.
- **Housekeeping:** detach `AmazonEC2ContainerRegistryFullAccess` from `Auto`, stop the SageMaker notebook instance when idle, and optionally refresh the 449-day window with new data (§12 item 12).

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
├── README.md                    ← this file: handoff, project journey, reference sections, research dossier for the paper (§15)
├── AGENTS.md                    ← rules for any coding agent: privacy, metric policy, one results window, the owner's dashboard rules, research rules
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
│   ├── app.py                   ← Streamlit dashboard entry point (deployed on Streamlit Community Cloud; reads S3 + daily/results) — see §10
│   ├── ui/                      ← the page: data.py, logic.py, theme.py, cards.py, header.py, sidebar.py, tab_track.py, tab_market.py, tab_reliability.py (Phase 12-13; tab_about.py removed on 17 Nov)
│   ├── assets/                  ← mfu_logo.png, adt_logo.png (sidebar header)
│   ├── requirements.txt         ← streamlit>=1.56 (Cloud installs the newest release), boto3, pandas, numpy, plotly, pyarrow; no TensorFlow
│   └── .python-version          ← 3.11
├── .streamlit/config.toml       ← theme colour and minimal toolbar
├── .claude/launch.json          ← preview-server config for Claude Code sessions (optional)
├── .devcontainer/               ← dev container definition
├── .python-version              ← 3.11
├── automation/                  ← daily inference-refresh Lambda — see §11 and automation/README.md
│   ├── README.md                ← full design rationale + deploy/teardown steps
│   ├── lambda_function.py       ← the daily handler v2 (ingest, data quality, numpy LSTM forecast, live log); test_lambda_function.py tests it against an in-memory S3
│   ├── deploy.py, retrain.py    ← idempotent boto3 deploy (plan/apply/teardown) and the weekly retrain; v1 files (convert_and_export.py, Dockerfile, deploy.sh, *.py.old) are kept for the record
│   └── iam_*.json, scheduler_*.json  ← least-privilege policies for the Lambda's own role
├── daily/                       ← corrected daily pipeline (Sep 2026) — see Journey and §8.0
│   ├── build_daily_dataset.py   ← scale-free features + log-return targets from the Silver table; --macro / --calendar / --cot add the Phase 9 feature sets
│   ├── test_no_lookahead_daily.py
│   ├── walk_forward.py          ← walk-forward LSTM (retrain every 7 days, 365-day window) + baselines + tomorrow's forecast
│   ├── horizon_always_up.py     ← always-up accuracy by horizon (Phase 8 table)
│   ├── serving.py, export_serving_model.py, test_serving.py, serving_artifacts/   ← numpy serving core, model export, seed model (Phase 11)
│   ├── regression_metrics.py, test_regression_metrics.py   ← price error, out-of-sample R², causal interval coverage (Phase 10)
│   ├── walk_forward_horizon.py  ← price 5 / 20 trading days ahead vs. no-change (results/horizon_h5, horizon_h20)
│   ├── interval_experiment.py, lstm_quantile.py, interval_vix.py   ← 90% interval method comparison and the VIX test
│   ├── volatility_experiment.py ← volatility forecast comparison (EWMA, GARCH, VIX; QLIKE) → volatility_forecast.csv, volatility_comparison.json
│   ├── requirements.txt
│   └── results/                 ← main run (449-day window): walk_forward_results.json, wf_predictions.csv, regression_metrics.json, latest_forecast.json, interval_comparison.json, interval_vix.json, volatility_forecast.csv, volatility_comparison.json (read by the dashboard)
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
| 6 | §3.4 Feature Engineering / §3.5 dataset sizes: Silver = 5,491 rows (Jun 2004–Apr 2026), Gold = 5,431 rows after dropping 61 rows, train = 4,887 rows, test = 544 rows (the proposal's figures, from a collection through Apr 2026) | **Re-verified 2026-10-29: Verified from S3 on 2026-10-29: Silver = 5,606 rows (2004-06-01 to 2026-09-21), Gold = 5,546 rows (2004-08-25 to 2026-09-18); split at `2024-12-31` by date gives train = 5,115 and test = 431 rows. The proposal's 4,887/544 split cannot be reproduced from the date rule, so use the verified numbers.** The *code* that produces these (60-row drop for MA60 + 1-row drop for undefined target = 61; split at `2024-12-31`) matches the proposal's description exactly | No known deviation — listed here for completeness since every other numeric claim in the proposal was checked. If the dataset has been re-collected since (new trading days appended), these exact row counts will drift slightly and should be re-printed from a fresh run before finalizing the report. |

---

## 4. AWS Account State (actual, as of now)

- **Account**: owned by a friend of the student. Account ID intentionally
  omitted from this document — this README is public on GitHub, and the
  account belongs to someone outside this project.
- **Login identity used day-to-day**: IAM user `Auto`
  (ARN `arn:aws:iam::<ACCOUNT_ID>:user/Auto`) — **not** the root user.
- **IAM policies currently attached to user `Auto`** (checked 2026-11-17):
  - `AmazonS3FullAccess`
  - `AmazonSageMakerFullAccess`
  - `IAMFullAccess` (added so `Auto` can manage its own access keys —
    had to be attached by someone with existing IAM permissions, since
    `Auto` could not grant this to itself)
  - `AWSLambda_FullAccess` and `AmazonEventBridgeSchedulerFullAccess` (for the daily pipeline, Phase 11)
  - `AmazonEC2ContainerRegistryFullAccess` (left over from the abandoned container deployment, Phase 4; nothing uses it, so it can be detached)
  - `IAMUserChangePassword`
- **Region**: `ap-southeast-2` (Asia Pacific — Sydney). All S3 and
  SageMaker resources live here.
- **S3 bucket**: `gold-lstm-forecast`
- **Daily pipeline (since 2026-10-29, §11)**: Lambda `gold-lstm-forecast-daily-refresh` (python3.12, 1024 MB, zip package plus the AWS-managed "AWS SDK for pandas" layer), an EventBridge Scheduler schedule of the same name
  (`cron(0 3 * * ? *)` UTC = 10:00 Thailand, ENABLED), two roles (`gold-lstm-forecast-daily-refresh-role` for the function and `gold-lstm-forecast-daily-refresh-scheduler-role` for the schedule; the least-privilege policies are in `automation/`)
  and CloudWatch metrics in the namespace `GoldLSTM`. S3 prefixes it uses: `bronze/streaming/` (append-only bar log), `gold/xauusd_daily/predictions/` (`latest_forecast.json`, `forecast_log.csv`, `dq_report.json`) and
  `model/serving/serving_model.npz` (last written 2026-10-29). No CloudWatch alarms or SNS topic exist (`Auto` lacks the permissions). Cost: far inside the free tier.
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
"next day" to define `target`). Gold dataset: 5,431 rows when first built (through Apr 2026); 5,546 rows through 2026-09-18 when re-checked on S3 on 2026-10-29.

### 04_Preprocessing.ipynb — the leakage-critical stage

Config: `SEQ_LEN = 60`, `FEATURES = ['close','return','ma7','ma14','ma30','ma60','volatility_7','momentum_7']`, `TARGET = 'target'`.

1. **Chronological split** at `split_date = '2024-12-31'` — originally reported as train = 4,887
   rows, test = 544 rows (2025–Apr 2026); on the S3 data of 2026-10-29 the same rule gives train = 5,115 rows, test = 431 rows (2025-01-02 to 2026-09-18). Never random/shuffled
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

> **Update (17 Nov 2026): final form.** The owner closed the dashboard work on 17 Nov 2026. The "Current dashboard" subsection below describes exactly what is deployed; the notes in this block quote are the history that led to it,
> and what was removed on the way is kept in "Content removed from the dashboard" (Phase 13).
>
> **Update (30 Oct 2026): rebuilt for readability (Journey, Phase 12).** The page is now `dashboard/app.py` plus the `dashboard/ui/` package, uses only the result files, the live
> pipeline's S3 output and the price history, and no longer loads TensorFlow or the original-pipeline views mentioned in the updates below; the "Current dashboard" subsection
> describes what is deployed today. The older notes are kept for the record.
>
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

**STATUS: live; closed by the owner on 17 Nov 2026.** The structure in the next subsection is what a reader sees today; change it only on the owner's request (the rules are in `AGENTS.md`).

### Current dashboard (17 November 2026, final)
- **Hosting.** Streamlit Community Cloud, app path `dashboard/app.py`, Python 3.11, redeployed automatically from `main`. AWS credentials live in the app's Secrets (never in git).
  Local run: `python3 -m streamlit run dashboard/app.py` from the repo root (AWS credentials are needed for the live forecast and the price history; without them the page falls back to
  the committed result files). Requirements: streamlit, boto3, pandas, numpy, plotly, pyarrow.
- **Code layout.** `app.py` (page config, loaders, header, navigation bar and the selected section), `ui/data.py` (every cached loader; each degrades to None when a source is missing), `ui/header.py`, `ui/sidebar.py`, `ui/cards.py` (`metric_card`, `card_row`),
  `ui/tab_track.py`, `ui/tab_market.py`, `ui/tab_reliability.py`, `ui/theme.py` (colours, `usd()`, `chart_title`, `panel_title`, `highlight`, the shared chart style) and `ui/logic.py` (the price verdict and the chip colour).
  The theme (one light theme, the advisor's preference) and the font are in `.streamlit/config.toml`.
- **Sidebar.** Credits only: MFU and ADT logos, project title, degree and school, authors with student IDs, advisor and committee. The page has no link to the repository or to this README.
- **Header.** The title with a freshness badge ("Live · updated N h ago"; its tooltip gives the 10:00 refresh and the data and model dates); three lines under it (what the model forecasts, that live forecasts are saved before the day and scored against the real close,
  the senior-project disclaimer); the forecast card (the forecast day, the forecast price, its change against the last close, and the 90% range with the *range reliability* badge: 89.5% in testing, the live count in its tooltip);
  the chart "Actual vs Forecast" (window 1M by default, 3M, 6M, 1Y: the actual line, the forecast made the day before with its range band, the new forecast with its range bar, the daily miss as bars);
  the scorecard "Accuracy · 449 backtest days" (MAE · average miss, RMSE, Error % (MAPE), R² (out-of-sample); label and number only, definitions in tooltips) with a "How to read this" popover.
- **Navigation bar** (one `st.segmented_control`, nothing selected at load, click the active one to hide it):
  - *Forecast vs. actual*: **Backtest · 449 test days** (badge "Simulated on past data"; window filter 1M / 3M / 6M / All at the right end of the title row, default 3M; actual vs. forecast price and the daily miss; CSV download of the predictions), then **Live record**
    (badge "Real forecasts"; cards: forecasts scored, MAE, range reliability; the chart of each real forecast with its range and the actual close; the table of the last 15 scored forecasts).
  - *Actual market data*: gold price history (1Y / 5Y / 10Y / All with volume and the test-period marker), volatility (typical daily move, yearly pace, VIX; actual moves against the model's limit; an expander comparing volatility models; long-run volatility),
    daily changes (histogram) and what the model looks at (six inputs, one picker).
  - *How reliable?*: **The evidence** (three rows: MAE · average miss with its chip against the baseline, R² · predicts moves?, range reliability with the goal 90%), the chart *Average miss (MAE) by model* (LSTM first, simple guess, Ridge, gradient boosting)
    and the chart *Range reliability by method* (LSTM, all past errors, GARCH, LSTM learns the range, LSTM + VIX).
- **Data it reads.** From the repository (`daily/results/`): `walk_forward_results.json`, `wf_predictions.csv`, `regression_metrics.json`, `volatility_forecast.csv`, `volatility_comparison.json`, `interval_comparison.json`, `interval_vix.json` (cache keyed on file mtime, so a refresh shows up
  without a restart); from S3 (`gold-lstm-forecast`): `gold/xauusd_daily/predictions/latest_forecast.json` and `forecast_log.csv`, `bronze/streaming/xauusd_daily_incremental.csv` and `silver/xauusd_daily_clean.parquet` (the price history).
  The `with_*` and `horizon_h*` result folders are no longer read by the page (they fed the removed "What we tried" chart; their numbers are in "Content removed from the dashboard" and §15).
- **Rules that came from real bugs:** native Streamlit elements only; `width="stretch"` instead of `use_container_width`; escape dollar signs in markdown and `st.metric` as `\$`; wrapping containers instead of `st.columns` where the open sidebar would squeeze content, and stretch only for `st.metric` cards
  (text-heavy cards and charts are squeezed instead of wrapping); charts use the system font; keep visible text short (explanations go in `help=` tooltips, popovers or expanders, and `$` in a tooltip must be escaped like anywhere else); check every layout change at several widths on
  Streamlit 1.56 and 1.66 and on the live URL (see Phase 12). The full list, with the owner's wording rules, is in `AGENTS.md`.
- **What was removed, and where it went.** The header briefing, the metric notes and chips, the verdict pill and the verdict box, the side-check and all-statistics expanders, the evidence and chart-title tooltips, the *About the project* section (pipeline steps, "What we tried", method, glossary, links) and the old
  Method/Data-explorer tabs: all documented in Phase 13 ("Content removed from the dashboard"), with the numbers rebuilt from the result files.

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
- Sets the page up (`st.set_page_config`: wide layout, sidebar 320 px), then loads the evaluation (`data.results()`, `data.regression()`), the current forecast (`data.current_forecast`: the live pipeline's `latest_forecast.json` when it is newer than the one stored with the last walk-forward run, otherwise the stored one),
  the live log and the volatility frame, and builds the forecast history (`data.forecast_history`: backtest and live forecasts aligned to the day they were for).
- Renders the sidebar and the header, then the navigation bar and only the selected section (`key="section"`; in `AppTest` prime `st.session_state["section"]` to test a section).
- Reads AWS credentials from the environment (locally) or Streamlit Secrets (Cloud); no credential is in the repo. If S3 cannot be read, the loaders return None and the page falls back to the committed result files; if there is no forecast at all it shows an error and stops.
- All UI text is in English and written for readers who are not specialists; numbers in prose are read from the result files.
- The original `app.py` (Gold parquet, scalers, `lstm_model.keras`, live baselines, 60-day window, `awswrangler`) was replaced in Phase 12; its description is in the git history and in "Layout of the original design" above.

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
pip install -r dashboard/requirements.txt
```
`dashboard/requirements.txt` (since Phase 12; the original design also needed awswrangler, tensorflow and scikit-learn for the views that were removed):
```
streamlit>=1.56
boto3>=1.34
pandas>=2.0
numpy>=1.24
plotly>=5.18
pyarrow>=14
```

**4. Run**:
```bash
python3 -m streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`. Close the process when done — nothing on
AWS keeps running as a result (S3 is storage-only and billed negligibly at
this data volume; no compute cost from running the dashboard).

---

## 11. Daily Automation Pipeline (Lambda + EventBridge Scheduler)

**Current status (2026-11-17): running.** Deployed on 2026-10-29 and unattended since 2026-10-30; the EventBridge schedule is `cron(0 3 * * ? *)` UTC = 10:00 Thailand since 2026-11-04. The latest run (2026-11-17, 03:00 UTC) produced the forecast made from the 2026-11-16 close; the live log holds 12 forecasts, 11 of them scored.
Open: the weekly retrain (`automation/retrain.py`) has not been run since 2026-10-29, the log is not tamper-proof and there are no alarms (Handoff; §12 items 18-22). Where the original v1 text below disagrees (08:00 Thailand, a TFLite container, `convert_and_export.py` plus `deploy.sh`),
this paragraph, Phase 11 and `automation/README.md` win: deploy with `python3 automation/deploy.py apply`.

**Status (2026-10-29): deployed and scheduled (see Journey, Phase 11).** The design below is the original v1 (price-level model, TFLite, container image),
kept for the record; the running v2 serves the return-target model with numpy and is described in Phase 11 and in `automation/README.md`.
**Original status text:** designed and code-complete in `automation/`, not yet deployed or
run against the live AWS account — it was built in an environment with no
network path to AWS, so "ready to deploy" was the accurate claim, not
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
silently patched; see `automation/README.md` §5 for the full explanation. (Superseded in Phase 12: the dashboard no longer reads the Gold table; its forecast comes from the pipeline's `latest_forecast.json`.)

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

1. ~~**Thesis write-up.**~~ **Superseded (2026-10-29): the paper will be rewritten from scratch, so this checklist is kept only for the record and the Word files are no longer maintained; results are documented in this README.** *Earlier status (Oct 2026): the advisor allowed rewriting, so the old proposal is superseded and a new report is
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
3. ~~Re-verify the exact row counts~~ **Done (2026-10-29), read-only from S3:** Verified from S3 on 2026-10-29: Silver = 5,606 rows (2004-06-01 to 2026-09-21), Gold = 5,546 rows (2004-08-25 to 2026-09-18); split at `2024-12-31` by date gives train = 5,115 and test = 431 rows. The older counts (5,491 / 5,431 / 4,887 / 544) are kept in §3 and §6 as the proposal's figures.
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
8. ~~Actually deploy and verify the daily automation~~ **Done (2026-10-29, Phase 11):** redesigned (numpy serving, zip + layer), deployed, invoked, schedule ENABLED.
   Unattended runs verified (from 2026-10-30; the schedule moved to 03:00 UTC on 2026-11-04). Remaining: run `retrain.py` weekly (not done since 2026-10-29), optionally add alarms. *Original item:* deploy and verify the daily automation (§11) against the
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

11. **Clean up the old AWS leftovers** (Journey, phase 4): the Lambda is now finished (item 8) and the unused, empty ECR repository was deleted (2026-10-29);
    only the broad policies below (`AmazonEC2ContainerRegistryFullAccess` is no longer needed and was still attached on 2026-11-17) can still be detached. Earlier text: the ECR
    repository `gold-lstm-forecast-daily-refresh` and the policies
    `AWSLambda_FullAccess`, `AmazonEC2ContainerRegistryFullAccess`,
    `AmazonEventBridgeSchedulerFullAccess` attached to `Auto` are standing changes
    on the friend's account; detach/delete them if the automation is dropped.
12. **Refresh the corrected daily results** after new data arrives:
    `python daily/build_daily_dataset.py` then `python daily/walk_forward.py ...`
    (commands in the script docstrings), commit `daily/results/`. **Last done 2026-10-27** (data through 2026-10-26,
    see Journey, Phase 7 refresh note). The S3 silver table is still at 2026-09-21: re-run notebooks 01-02 to bring it forward (the Lambda's newer bars sit in `bronze/streaming/` and nothing absorbs them into Silver, item 10), or pass a locally extended silver parquet to `--silver`. When refreshing, re-run the main walk-forward
    **and all four feature-search variants on the same rows**, and update the README (the dashboard reads the result files by itself; the Word files are no longer maintained); delete or
    repoint `daily/data/macro_cache.parquet` and `cot_cache.parquet` first, because a stale cache is reused as-is (see Handoff).
13. **Daily-pipeline cross-asset + calendar + COT features: done (Phase 9, all addenda), null** —
    DXY, silver, S&P 500, VIX, 10Y yield, TIP, NFP, FOMC (204 dates), CPI (271 dates) and CFTC Commitment of
    Traders positioning all added to the daily model; best single result is LSTM AUC 0.520 (macro+NFP+FOMC+CPI,
    without COT), p vs. always-up = 0.367 — closer than every other variant but never significant; adding COT
    on top made it worse (AUC 0.498). This particular feature search found nothing; other ideas remain open. The hourly version of
    cross-asset features (silver, EURUSD, S&P 500, DXY as hourly bars) was run too and is also null (hourly
    addendum in the Journey). Of the four missing hourly ask months, 2014-04, 2015-07 and 2016-08 were downloaded on 2026-10-29 (`hourly/download_dukascopy.py xauusd --start 2014-04 --price ask --raw-dir data/raw_ask`);
    2026-09 and 2026-10 failed after retries and are still missing (retry later). The hourly datasets and results were **not** rebuilt with the new months.
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
18. **Weekly retrain of the serving model.** `python3 automation/retrain.py` (needs TensorFlow and AWS credentials, so it runs on a machine of the owner's, not in Lambda). The model object was last written on 2026-10-29 and nothing schedules the retrain, so every live forecast since comes from that model.
19. **Make the live record tamper-proof if it will be cited.** `forecast_log.csv` is an ordinary S3 object and the Lambda replaces the row of the same as-of date on a re-run, so edits are possible and invisible. Options: S3 bucket versioning with Object Lock, or a daily commit of the log to git.
    Until then do not call the live forecasts pre-registered. The *Live record* tooltip on the page ("nothing can be edited afterwards") overstates this and should be reworded.
20. **Back up `forecast_log.csv`** (one copy outside the bucket) before anything is changed.
21. **Alarms.** No CloudWatch alarms or SNS topic; the IAM user `Auto` lacks the permissions and the owner chose not to widen it. A failed run is visible only as a gap in the live log and in the Lambda's CloudWatch logs.
22. **Absorb the streaming log into Silver** (item 10) and refresh the 449-day window once Silver is current; the live record already runs past the window's end (2026-10-23).
23. **Paper (IEEE format; owner, 2026-11-17).** The research dossier is §15; the extra experiments that would strengthen the paper are listed there (§15.8).

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

---

## 15. Research dossier for the IEEE paper

*Added 2026-11-17. The owner's decision: the next step is an IEEE-format paper, written with this README and Claude as research helpers. Every external fact below is dated; anything marked "check" must be re-checked before it goes into a submission.*

### 15.0 How to use this section
- It turns the README into a working knowledge base. Each statement is one of three kinds: **generated** from the committed result files (the table says which), **read from the code** (a path is given), or a **plan or judgement** (labelled as such). If a sentence in a draft cannot be traced to one of these, it does not go into the paper.
- Read §15.1 (what may be claimed) and §15.13 (the rules for working with Claude) before drafting anything. The Handoff block above gives the current state; Phases 5-13 give the history behind each number.
- Where to find things: data and splits §15.3, method §15.4, results §15.5, pitfalls §15.6, threats §15.7, extra experiments §15.8, bibliography §15.9, figures §15.10, writing plan §15.11, reproducibility §15.12, decisions needed §15.15, checklist §15.16.

### 15.1 The paper at a glance
**Working titles (pick one with the advisor).**
1. *Is the LSTM Better Than Repeating Yesterday? A Causal Walk-Forward Audit of Next-Day Gold Price Forecasting*
2. *Calibrated but Not Skilled: Honest 90% Intervals for an LSTM Gold Price Forecast*
3. *When Deep Learning Meets the Random Walk: Evaluation Pitfalls in Gold Price Forecasting and a Reproducible Causal Protocol*

**Research questions.** RQ1: does a walk-forward LSTM forecast the next-day closing price of gold more accurately than the no-change forecast ("the simple guess")? RQ2: do extra inputs (macro and cross-asset prices, release calendars, trader positioning), other horizons (5 and 20 days) or hourly data change the answer? RQ3: can a 90% prediction interval be calibrated honestly (using only information known before each forecast), and which interval method works best? RQ4: which evaluation mistakes inflate apparent skill in this setting, and by how much?

**The evidence supports (may be written, with the numbers and sources given):**
1. On 449 walk-forward test days (2025-01-09 to 2026-10-23) the LSTM's MAE is \$46.83 against \$46.63 for no-change (Diebold-Mariano p = 0.385 with 24 lags, 0.400 with none): no evidence of skill (Table III).
2. On returns there is no skill either: out-of-sample R² -0.011 against the rolling mean, Spearman IC +0.006 (p = 0.90) (Table III).
3. Direction accuracy is 51.9% (Wilson 95% CI 47.3-56.5%), below the always-up rate of 55.2% (p = 0.155), AUC 0.463; the model says "up" on 88% of days (Table IV).
4. The null holds for five input sets (Table V), for 5- and 20-day horizons (Table VI) and for hourly data (Phases 5-6).
5. The shipped band (forecast ± 1.645 × an exponentially weighted average of past errors, decay 0.94, fixed in advance) covers 89.5% of the 389 scored days (Kupiec p = 0.725); empirical quantiles of all past errors, GARCH(1,1) and filtered historical simulation under-cover (85.3-85.9%, Kupiec p ≤ 0.010) and an LSTM trained on the quantiles is far too narrow (63.5%) (Table VII). Centring the same band on the no-change forecast instead of the LSTM gives the same coverage and an interval score that is no worse (314 against 317, p = 0.32): the interval's quality comes from the volatility model, not from the LSTM.
6. For one-day variance forecasts the EWMA with decay 0.94 has the lowest QLIKE of seven candidates (Table VIII).
7. The evaluation pitfalls found on the way are quantified (§15.6).
8. A serverless pipeline has produced and scored live forecasts unattended since 2026-10-30 (11 scored on 2026-11-17; descriptive only).
9. Risk, unlike price, is predictable (E6, Tables E6-1 to E6-6): the R² of every volatility forecast against the realised variance is positive (the shipped EWMA 0.19, HAR-RV with release-day dummies 0.29 on 2010-2026), and HAR-RV built on hourly realised variance beats the shipped EWMA on the long window (QLIKE -0.054, DM p < 0.001, lower in 15 of 17 years for HAR-RV logs); on the 428-day primary window the same models are better but not significantly (p = 0.10), and an LSTM does not help.

**The evidence does not support (do not write):**
- That the LSTM forecasts gold prices accurately, beats its baselines or is useful for trading. On this evidence the opposite holds, and no strategy, transaction cost or economic value was tested.
- The old "~70% accuracy" target: it is superseded, and always-up alone scores 55.2% (75.6% at 60 days, Table X).
- That macro, calendar or positioning inputs cannot help: with 449 days the test has limited power, so the correct wording is "no evidence of an effect", not "no effect".
- That one input set is best: FOMC+CPI shows 57.5% accuracy and AUC 0.520, but p against always-up is 0.367, and with five variants and many metrics some small p-values appear by chance.
- Generality: one asset (COMEX gold futures as a proxy for XAU/USD), one 449-day window, mostly a rising market (55.2% up days).
- That HAR-RV beats the shipped EWMA with statistical significance on the 2025-2026 window (p = 0.10 for HAR-RV-X; the significant result is the long window), that the LSTM helps volatility forecasts (it does not), or that the volatility result changes the return R² (it does not).
- That the live forecasts are "pre-registered" or immutable: the log can be overwritten (§12 item 19).
- That the 89.5% coverage is a clean out-of-sample figure: the decay 0.94 was fixed in advance, but the alternatives were compared on the same window, so it is slightly optimistic.

**Contributions (draft).** C1: a causal, leakage-tested walk-forward protocol with regression-first metrics and Diebold-Mariano tests against no-change, released as code. C2: a documented null across models, five input sets, three horizons and two data frequencies. C3: an interval comparison showing that an EWMA of past errors beats GARCH, filtered historical simulation, empirical quantiles, an LSTM quantile model and VIX-scaled variants on honest coverage. C4: a quantified catalogue of evaluation pitfalls from the project's own mistakes. C5: an unattended AWS pipeline with a live scoring log (descriptive).

**Abstract draft (about 180 words; the JCSSE limit is about 150, so trim).** Deep sequence models are often reported to forecast gold prices well, but such reports rarely test against the strongest benchmark: repeating the last close. We build a causal walk-forward evaluation (retraining every 7 trading days on 365 days; 449 out-of-sample days from 2025 to 2026) of an LSTM with a 20-day window that predicts the next-day log-return of COMEX gold futures from 16 scale-free price features. The LSTM's mean absolute error (USD 46.83) is statistically indistinguishable from the no-change forecast (USD 46.63; Diebold-Mariano p = 0.38), its out-of-sample R² is -0.011 and its rank correlation with realised returns is +0.006 (p = 0.90). Adding macro, calendar and positioning data, forecasting 5 or 20 days ahead, and using hourly data do not change this. A 90% prediction interval built from an exponentially weighted average of past errors covers 89.5% of 389 scored days and outperforms GARCH, empirical-quantile and LSTM-quantile alternatives. We quantify four evaluation pitfalls (price-level scaling, majority-class accuracy, test-window calibration and bid-ask artefacts) and describe an unattended live forecasting pipeline.

**Keywords / IEEE Index Terms (candidates; check against the IEEE Thesaurus).** Forecasting, Long short-term memory, Predictive models, Time series analysis, Neural networks, Reproducibility of results, Uncertainty; free keywords: gold price, walk-forward validation, prediction intervals, Diebold-Mariano test, negative results.

### 15.2 Venues and IEEE format
*Checked on 2026-11-17 from the venues' own pages and search results; deadlines move, so check again.*

| Venue | Why it fits | Facts | Deadlines |
|---|---|---|---|
| **JCSSE 2027**, the 24th International Joint Conference on Computer Science and Software Engineering, 16-18 June 2027, Chiang Rai | Co-organised by the School of Applied Digital Technology at Mae Fah Luang University (the owner's school); track "Machine Learning and Intelligent Systems" | 4-6 A4 pages in PDF, 10-pt Times New Roman, abstract about 150 words, strictly the IEEE conference template (Word and LaTeX); double-blind review (at least three reviewers); similarity score at most 25% for IEEE Xplore; any use of generative AI must be fully declared at submission; submissions through EDAS. Accepted papers are "submitted for possible inclusion into IEEE Xplore", not guaranteed. At least one author must register and present. Fees in THB, early / late: IEEE member or student 12,600 / 14,000, non-member 14,000 / 15,500, participant 7,000. There is no finance track | 1st round **27 Nov 2026** (notification 15 Jan 2027, too soon for a full paper); **2nd round 22 Jan 2027** (notification 5 Mar 2027); final manuscript 7 May 2027. Source: jcsse2027.mfu.ac.th. Check: the CFP PDF was not read, the registration page gives two early-bird dates (5 Feb and 23 Apr), and an aggregator listing and a search snippet gave other dates |
| **IEEE Access** (journal) | Rolling submission; has "Applied Research" and "Negative Result" article types; no page limit | APC USD 2,160 plus local taxes (5% off for IEEE members, 20% off for IEEE and society members, none for students; discounts do not stack); single-anonymised review with at least two reviews and a binary accept/reject decision; acceptance about 20%; 4-6 weeks from submission to publication; 20 pages or fewer recommended; double-column IEEE Access template (Word and LaTeX source plus PDF both required); ORCID for the submitting author and 3-10 keywords; submissions must "present a clear advance over the state-of-the-art" or are desk-rejected, and a negative result is acceptable only if the question is meaningful and the study rigorous; AI-generated text must be disclosed in the acknowledgements; a conference paper may be extended if similarity is below 35%. Source: ieeeaccess.ieee.org/authors | Rolling |
| Not available now | IEEE CIFEr / SSCI 2027 (Gold Coast, 14-17 Feb 2027; deadlines 10 Aug and 10 Sep 2026 passed); ICSEC 2026 (Kasetsart, 18-21 Nov 2026); ICMLA 2026 (5-7 Oct 2026; the 2027 call is not out); IEEE CIFEr 2026 (Tokyo, Sep 2026); IEEE CAI 2027 (Singapore, 30 Nov-2 Dec 2027; paper deadline "coming soon") | | Watch the CAI and ICMLA 2027 calls |

**Plan (judgement, not fact).** (1) Aim at the JCSSE 2027 second round (22 Jan 2027) with a 4-6 page paper built on RQ1-RQ3 and the pitfalls; the first round on 27 Nov is not realistic. (2) December 2026: freeze the protocol, run the cheap experiments E1, E8, E9 and E10 (§15.8), draft the Introduction, Method and Results. (3) Early January: figures, full draft, advisor review. (4) By 22 Jan: anonymise (§15.15) and submit. (5) After the conference or in parallel: an extended IEEE Access article (Applied Research or Negative Result) adding E2-E7; keep the overlap with the conference paper below 35% similarity. A top-tier venue is not a realistic target for a single-asset null result unless E3-E6 produce a stronger story (angle A with B, §15.8).

**IEEE format, general conventions (the venue's template and instructions override these).** LaTeX class `IEEEtran`: `\documentclass[conference]{IEEEtran}` (add `a4paper` for JCSSE) and the `IEEEtran.bst` bibliography style, numbered references in order of first citation. Abstract in one paragraph without citations, formulas or abbreviations that are not defined; then Index Terms. Table captions above the table (TABLE I, roman numerals, small capitals); figure captions below ("Fig. 1."); figures at 300 dpi or vector, legible when printed in grey, 3.5 in wide for one column and 7.2 in for two; equations numbered and referred to as (1); define every abbreviation at first use; no footnotes in the abstract. For a double-blind review remove author names, affiliations, acknowledgements and any URL that identifies the authors (the public GitHub repository and the dashboard do), and cite one's own work in the third person.

### 15.3 Data
**Table I. Data and splits** (`daily/results/build_report.json`).

| Item | Value |
|---|---|
| Raw daily bars | 5,631 (2004-06-01 to 2026-10-26), no duplicate dates, no gap longer than 5 days |
| Modelling table | 5,531 rows (2004-08-26 to 2026-10-23), 16 features |
| Train / validation / test rows | 4,835 / 247 / 449 (embargo 5 bars; up-day share 52.6% / 57.9% / 55.2%) |
| Invalid OHLC rows | 347 (6.2%; by year 2004 33%, 2005 26%, 2006 34%, 2007 28%, 2008 17%, 2009 4%, 2010 2%, 2011 8%); features use the close only |
| Zero-volume share | 2.8% (volume is not used) |
| Extreme one-day moves (|log return| above 7%) | 2006-06-13 (-7.6%); 2008-09-17 (+8.6%); 2009-03-19 (+7.5%); 2013-04-15 (-9.8%); 2026-01-30 (-11.4%) |

**Sources.**
- **Daily gold**: COMEX gold futures, continuous contract `GC=F`, from Yahoo Finance (`yfinance`), 2004-06-01 onwards; the live pipeline appends settled bars from the Yahoo chart endpoint each morning. It is a futures series (roll effects, settlement at the COMEX close), a proxy for spot XAU/USD: the paper must say so. The cleaned history is in S3 (`silver/xauusd_daily_clean.parquet`, last row 2026-09-21) and newer bars are in `bronze/streaming/xauusd_daily_incremental.csv`. Every feature is computed from the close alone because open, high and low are unreliable before 2012.
- **Hourly gold**: Dukascopy spot XAU/USD bid and ask, 2004-2026, hourly (used only in the side experiments, Phases 5-6); mid = (bid + ask) / 2. Auxiliary hourly series: silver, EUR/USD, S&P 500 index (from 2011-09) and a dollar-index proxy (from 2017-12). Ask data are missing for 2026-09 and 2026-10 (rate limiting); the raw data must not be redistributed.
- **Macro and cross-asset (daily, Yahoo)**: DXY (`DX-Y.NYB`), silver (`SI=F`), S&P 500 (`^GSPC`), VIX (`^VIX`), 10-year yield (`^TNX`) and TIP. **Calendars**: NFP (first Friday of the month), FOMC dates (204) and CPI release dates (271), lists inside `daily/build_daily_dataset.py`. **Positioning**: CFTC Legacy Commitments of Traders (public Socrata API), made available at report date + 3 days to avoid look-ahead.
- **Data-quality findings** (reported in `daily/results/build_report.json` and Phases 5-7): 6.2% of daily rows have inconsistent OHLC (mostly 2004-2011); the Silver `_1w` and `_1m` columns hold the period's final values and are therefore never used; the hourly downloader saved partial files when the feed answered HTTP 429, which row-count checks caught; about a third of hourly rows were market-closed bars with zero volume.
- **Licences (check).** The vendors' terms decide whether data may be redistributed; the repository commits no raw vendor data (`.gitignore` excludes the parquet and array files). Draft data-availability statement: *Daily prices were obtained from Yahoo Finance and hourly spot prices from Dukascopy; their terms do not allow redistribution of the raw series, so only code and derived result files are released at <anonymised link>.*

### 15.4 Method (reproducible detail)
**Notation.** Day t, close c_t, log-return r_t = ln(c_t / c_{t-1}). A forecast made at the close of day t targets y_{t+1} = r_{t+1}; the forecast price is ĉ_{t+1} = c_t · exp(ŷ_{t+1}); the no-change ("simple guess") forecast is ĉ_{t+1} = c_t, i.e. a return forecast of 0.

**Features (16, all computed from the close, causal and scale-free; `make_features` in `daily/build_daily_dataset.py`).** f_ret_n = ln(c_t / c_{t-n}) for n = 1, 3, 5, 10, 20, 60; f_vol_n = standard deviation of the daily log-returns over the last n days for n = 5, 20, 60; f_ma_gap_n = c_t / mean(c over the last n days) - 1 for n = 7, 14, 30, 60; f_rsi_14 = 100 - 100 / (1 + mean gain / mean loss) over 14 days (simple means of the price differences); f_dow_sin and f_dow_cos = sin and cos of 2π · weekday / 5. The extra input sets add 18 (macro and cross-asset returns, VIX change and z-score, gold/silver ratio), 2 (NFP day, days since NFP), 4 (FOMC and CPI day and days since) and 4 (COT positioning) features (Table V).

**Splits.** Train to 2023-12-31, validation to 2024-12-31, test from 2025; an embargo of 5 bars (at least the longest horizon) separates the splits. The evaluation itself is walk-forward, so the split only fixes where the test window starts (2025-01-09).

**Walk-forward protocol (`daily/walk_forward.py`).** For each block of 7 consecutive test days, fit every model on the 365 labelled rows that precede the block (a label is known only after its day), predict the block, then move forward. Standardisation (mean and standard deviation of the training window, missing values set to 0, clipped to ±10) is re-fitted inside every window; the return target is divided by the training window's standard deviation. The first 20-day input window reaches back 19 days before the first training row. Result: 449 one-step forecasts (2025-01-09 to 2026-10-23).

**LSTM (`build_rnn` in `hourly/train_models.py`).** Input: 20 days × 16 features. LSTM(32) → Dropout(0.2) → Dense(32, ReLU) → two heads: a linear head for the standardised return (Huber loss, δ = 1) and a sigmoid head for the direction (binary cross-entropy), loss weights 1:1. Adam, learning rate 1e-3, batch size 64, at most 30 epochs, early stopping on the validation loss (the last 15% of the training window, taken chronologically) with patience 3 and the best weights restored. Three seeds (0-2) are trained per block and their predictions averaged. The forecast price uses the return head; the direction head is used only for the secondary direction metrics.

**Baselines on identical rows.** No-change; rolling drift (the training window's mean return, a constant forecast, which is "always up"); Ridge (median imputation, standardisation, α = 100) and gradient-boosted trees (`HistGradientBoostingRegressor`: depth 3, learning rate 0.05, 200 iterations, L2 1.0) on the 16 features of day t; a gradient-boosting classifier for direction; the momentum rule (sign of the recent return).

**Metrics (`daily/regression_metrics.py`).** MAE and RMSE in dollars of the price error; MAPE; skill 1 - MAE / MAE_no-change; Theil's U = RMSE / RMSE_no-change; R² on returns (sklearn definition: the total sum of squares is centred on the mean return, so a constant zero forecast scores slightly below 0); out-of-sample R² against zero and against the rolling mean, 1 - SSE / SSE_benchmark; information coefficient IC = Spearman rank correlation between forecast and realised return, with a p-value from the t distribution using n_eff = n / h.

**Tests.** *Diebold-Mariano* on the squared **return** errors against the no-change forecast (whose return error is y_t): loss differential d_t = e_t² - y_t², Bartlett-kernel long-run variance with L lags, two-sided normal p-value; L = 24 for the headline p (carried over from the hourly pipeline), L = 0 also reported; no Harvey-Leybourne-Newbold small-sample correction. This is not the loss of the dollar-MAE headline: a DM test on absolute price errors would match MAE better (experiment E1). *Direction*: accuracy of the sign, Wilson 95% interval, exact binomial test against the always-up rate, AUC from the direction head. For h > 1 labels overlap, so n_eff = n / h and the DM variance uses h-dependent lags (max(24, 2h)).

**Interval methods (`daily/regression_metrics.py`, `daily/interval_experiment.py`, `daily/lstm_quantile.py`, `daily/interval_vix.py`).** Shipped band: σ_t² = Σ_k λ^k e²_{t-1-k} / Σ_k λ^k with λ = 0.94 over all past return errors e (known before day t), price interval c_t · exp(ŷ_{t+1} ± 1.645 σ_t); first scored forecast after a 60-day warm-up (449 - 60 = 389 scored days). Alternatives: the 5% and 95% empirical quantiles of all past errors; GARCH(1,1) with Gaussian errors and with filtered historical simulation, refitted every 7 days; an LSTM trained with the pinball loss to output the quantiles directly (2 seeds); σ = exp(a + b ln EWMA + c ln VIX) fitted by Gaussian quasi-maximum-likelihood on the last 1,000 days (with same-day VIX, previous-day VIX or no VIX). Scores: coverage, mean width, interval score (Gneiting-Raftery, α = 0.10: width + (2/α) × distance outside), Kupiec proportion-of-failures test, coverage in each half.

**Volatility (`daily/volatility_experiment.py`).** One-day variance forecasts of r_{t+1}² scored with QLIKE = ln σ² + r² / σ² (Patton) and a DM test with HAC lag 5, against EWMA 0.94.

**Multi-day horizons (`daily/walk_forward_horizon.py`).** The same protocol forecasting the price h = 5 and 20 trading days ahead against "price in h days = price today"; training rows stop h days before each block (a label is only known h days later); n_eff = n / h with a HAC DM variance.

**Causality and correctness tests (all plain Python scripts, all pass on 2026-11-17).** `daily/test_no_lookahead_daily.py` (edits future prices and asserts past features do not change; multiplying all prices by a constant changes no feature; covers the price, macro, calendar and COT features), `daily/test_regression_metrics.py` (known values by hand, perfect and no-change forecasts, a band that covers about 90% on i.i.d. errors, bands use only past errors), `daily/test_serving.py` (the numpy forward pass equals Keras to about 1e-8; features from a short tail equal those from the full history), `automation/test_lambda_function.py` (the handler against an in-memory S3: idempotent replays, scoring of the previous forecast, a critical data-quality failure publishes nothing), `daily/test_volatility_har.py` (E6: realised-variance aggregation and the settlement boundary, HAR inputs never use the future, event dummies describe the target day, the EWMA and GARCH recursions, QLIKE and the DM test behave as they should). The hourly pipeline adds a mutation check (the tests fail when a look-ahead feature is injected) and a label-shuffle control (Phase 5).

**Live system (descriptive; §11, Phase 11).** EventBridge Scheduler (03:00 UTC) → Lambda (python3.12) that appends the newest settled bar, runs data-quality checks, computes the 20-day feature window, runs the LSTM forward pass in numpy from the weights in `model/serving/serving_model.npz`, adds the EWMA band and writes `latest_forecast.json`; every earlier forecast is scored once its next close is known (`forecast_log.csv`). The model is retrained by hand (`automation/retrain.py`) on the latest 365 labelled days.

### 15.5 Results for the paper
*Tables I and III-VIII are generated from the committed result files on 2026-11-17 (449 test days unless stated); Tables IX and X are copied from the Phase 5-8 notes; regenerate after any refresh. "OS R²" is out-of-sample R² against the rolling mean; p-values below 0.001 are shown as <0.001.*

**Table II. Model hyperparameters.** Section 15.4 lists them; the protocol block in `walk_forward_results.json` records: retrain every 7 days, 365 training days, window 20, 32 LSTM units, 3 seeds, target "log-return of next-day close".

**Table III. Price accuracy** (`walk_forward_results.json`, `regression_metrics.json`).

| Model | MAE (\$) | RMSE (\$) | MAPE (%) | MAE vs no-change | DM p (lag 0) | DM p (lag 24) | R² (returns) | OS R² vs rolling mean | IC (Spearman) | IC p |
|---|---|---|---|---|---|---|---|---|---|---|
| No-change (the simple guess) | 46.63 | 68.94 | 1.143 | – | – | – | -0.003 | -0.0006 | – | – |
| Rolling drift (always up) | 46.36 | 69.07 | 1.136 | -0.6% | 0.951 | 0.941 | -0.003 | +0.0000 | -0.027 | 0.561 |
| Ridge | 47.87 | 70.35 | 1.171 | +2.7% | 0.028 | 0.023 | -0.045 | -0.0427 | -0.103 | 0.029 |
| Gradient boosting (regressor) | 50.68 | 75.05 | 1.237 | +8.7% | <0.001 | <0.001 | -0.186 | -0.1833 | -0.034 | 0.473 |
| **LSTM (walk-forward)** | 46.83 | 69.45 | 1.147 | +0.4% | 0.400 | 0.385 | -0.014 | -0.0113 | +0.006 | 0.896 |

Reading: the LSTM's MAE is 0.4% above no-change (p = 0.385); Ridge and gradient boosting are significantly worse; every learned model has a negative out-of-sample R²; the rolling drift is level with no-change. All honest R² definitions are negative for the LSTM: -0.014 on returns, -0.011 out of sample against a zero forecast and -0.011 against the rolling mean (the no-change forecast scores -0.003 and -0.001). The R² of the *price level* is 0.989 for the LSTM and 0.989 for no-change (0.9892 against 0.9894): it looks excellent and says nothing, because prices are highly autocorrelated (pitfall 13).

**Table IV. Direction (secondary check).**

| Model | Direction accuracy | Wilson 95% CI | vs always-up (pt; always-up = 55.2%) | p vs always-up | AUC |
|---|---|---|---|---|---|
| Rolling drift (always up) | 55.2% | 50.6% to 59.8% | +0.0 | 1.000 | – |
| Ridge (sign of forecast) | 49.0% | 44.4% to 53.6% | -6.2 | 0.009 | – |
| Gradient boosting regressor (sign) | 50.3% | 45.7% to 54.9% | -4.9 | 0.041 | – |
| Gradient boosting classifier | 52.6% | 47.9% to 57.1% | -2.7 | 0.255 | 0.498 |
| **LSTM (direction head)** | 51.9% | 47.3% to 56.5% | -3.3 | 0.155 | 0.463 |
| Momentum rule | 45.2% | 40.7% to 49.8% | -10.0 | <0.001 | – |

Reading: always-up scores 55.2% because 55.2% of the test days were up; the LSTM says "up" on 88% of days, so its accuracy mostly follows how often gold rose.

**Table V. Input sets** (`daily/results/with_*/`; same 449 rows; LSTM only).

| LSTM input set | Features | Test days | LSTM MAE (\$) | No-change MAE (\$) | DM p | Direction acc. | AUC | p vs always-up | IC | OS R² |
|---|---|---|---|---|---|---|---|---|---|---|
| Price history only (main run) | 16 | 449 | 46.83 | 46.63 | 0.385 | 51.9% | 0.463 | 0.155 | +0.006 | -0.011 |
| + macro and cross-asset prices | 34 | 449 | 46.80 | 46.63 | 0.461 | 55.2% | 0.510 | 1.000 | +0.043 | -0.014 |
| + NFP release days | 36 | 449 | 46.74 | 46.63 | 0.595 | 55.0% | 0.514 | 0.924 | +0.055 | -0.010 |
| + FOMC and CPI release days | 40 | 449 | 46.38 | 46.63 | 0.974 | 57.5% | 0.520 | 0.367 | +0.070 | -0.001 |
| + CFTC positioning | 44 | 449 | 46.62 | 46.63 | 0.797 | 54.6% | 0.498 | 0.776 | +0.054 | -0.005 |

Reading: no input set beats the main run's MAE significantly; the best direction result (FOMC and CPI days) is not significant (p against always-up 0.367).

**Table VI. Horizons** (`daily/results/horizon_h5/`, `horizon_h20/`).

| Horizon (trading days) | n / n_eff | No-change MAE (\$) | LSTM MAE (\$) | LSTM vs no-change | DM p | OS R² | IC (p) |
|---|---|---|---|---|---|---|---|
| 1 | 449 / 449 | 46.63 | 46.83 | +0.4% | 0.385 | -0.011 | +0.006 (0.90) |
| 5 | 444 / 88 | 109.19 | 109.50 | +0.3% | 0.430 | -0.040 | -0.018 (0.87) |
| 20 | 429 / 21 | 219.40 | 216.95 | -1.1% | 0.586 | +0.034 | +0.085 (0.72) |

Reading: at 20 days the LSTM is 1.1% better in MAE with a positive OS R², but n_eff = 21 and p = 0.59.

**Table VII. 90% interval methods** (`interval_comparison.json`; 389 scored days).

| Method | Coverage (goal 90%) | Mean width \$ (% of price) | Interval score (\$) | Kupiec p | Coverage 1st / 2nd half | Score minus EWMA (p) |
|---|---|---|---|---|---|---|
| EWMA Gaussian, lambda 0.94 (current) | 89.5% | 223 (5.28%) | 317 | 0.725 | 88% / 91% | +0.0 (1.00) |
| Empirical quantiles, all past errors | 85.9% | 194 (4.68%) | 334 | 0.010 | 86% / 86% | +16.6 (0.16) |
| GARCH(1,1), Gaussian | 85.6% | 203 (4.80%) | 322 | 0.006 | 83% / 88% | +5.3 (0.24) |
| GARCH(1,1), filtered historical sim. | 85.3% | 200 (4.74%) | 322 | 0.004 | 83% / 88% | +4.9 (0.31) |
| No-change centre, EWMA (reference) | 88.7% | 221 (5.24%) | 314 | 0.398 | 87% / 91% | -3.0 (0.32) |
| LSTM quantile (pinball loss) | 63.5% | 129 (3.04%) | 457 | <0.001 | 63% / 64% | +140.2 (<0.01) |

Reading: the shipped EWMA band is the only method whose coverage a Kupiec test at 5% does not reject; centring it on the no-change forecast instead of the LSTM gives the same coverage and a similar score. VIX variants (`interval_vix.json`):

| Method | Coverage (goal 90%) | Mean width \$ (% of price) | Interval score (\$) | Kupiec p | Coverage 1st / 2nd half | Score minus EWMA (p) |
|---|---|---|---|---|---|---|
| EWMA lambda 0.94 (current) | 89.5% | 223 (5.28%) | 317 | 0.725 | 88% / 91% | +0.0 (1.00) |
| EWMA, intercept+slope refit (no VIX) | 87.1% | 208 (4.91%) | 322 | 0.071 | 84% / 90% | +4.5 (0.31) |
| EWMA + VIX (same-day close) | 86.1% | 203 (4.78%) | 322 | 0.015 | 83% / 89% | +4.8 (0.29) |
| EWMA + VIX (previous day) | 86.4% | 203 (4.79%) | 322 | 0.023 | 83% / 90% | +4.4 (0.32) |
| VIX only (previous day) | 77.1% | 148 (3.56%) | 359 | <0.001 | 79% / 75% | +41.7 (0.01) |

**Table VIII. One-day variance forecasts** (`volatility_comparison.json`; DM with HAC lag 5).

| Variance forecast | QLIKE (lower is better) | QLIKE minus EWMA 0.94 | DM p | Mean sigma (% per day) |
|---|---|---|---|---|
| EWMA 0.94 (current) | -7.355 | +0.000 | – | 1.51 |
| EWMA 0.97 | -7.325 | +0.031 | 0.273 | 1.53 |
| GARCH(1,1) | -7.288 | +0.067 | 0.069 | 1.39 |
| VIX scaled (ratio refit) | -6.686 | +0.669 | 0.065 | 1.02 |
| EWMA 0.94 + VIX | -7.292 | +0.063 | 0.175 | 1.39 |
| EWMA 0.94 + EWMA 0.985 | -7.278 | +0.077 | 0.050 | 1.41 |
| EWMA 0.94 + EWMA 0.985 + VIX | -7.267 | +0.088 | 0.073 | 1.38 |

Reading: EWMA 0.94 has the lowest QLIKE; the differences from GARCH(1,1) (p = 0.069) and from EWMA 0.94 + EWMA 0.985 (p = 0.050) are borderline and untested for multiplicity. Risk is weakly predictable even though price is not: a Mincer-Zarnowitz regression of the realised squared return on the EWMA variance forecast gives R² = 0.020 (slope 0.55), and the Spearman correlation between the realised |return| and the forecast sigma is +0.203 over the 449 days (positive but small, as expected when one squared return is a noisy proxy for a day's variance). E6, which uses realised variance from the hourly data instead, finds a Mincer-Zarnowitz R² of 0.17 to 0.30 (Tables E6-1 to E6-6).

**Table IX. Hourly side experiments** (Dukascopy XAU/USD, test 2025+, `hourly/results/`; window 48 bars, 21 features, two-head GRU; always-up is the baseline). Bid prices (Phase 5), then mid prices (Phase 6):

| Horizon | Always up | Best model accuracy | AUC range | n_eff |
|---|---|---|---|---|
| 1 bar | 51.5% | 51.4% (GRU ensemble) | 0.517-0.518 | 10,197 |
| 4 bars | 52.9% | 51.5% (HistGB regressor) | 0.495-0.500 | 2,548 |
| 24 bars | 55.1% | 53.7% (HistGB regressor) | 0.497-0.520 | 424 |

| Horizon (mid prices) | Always up | Best model accuracy | AUC range | n_eff |
|---|---|---|---|---|
| 1 bar | 51.7% | 51.6% (HistGB regressor) | 0.514-0.516 | 9,780 |
| 4 bars | 53.2% | 51.1% (HistGB regressor) | 0.495-0.499 | 2,444 |
| 24 bars | 55.8% | 54.2% (HistGB classifier) | 0.510-0.520 | 406 |

Adding silver, EUR/USD, S&P 500 and a dollar-index proxy changed nothing (IC with and without the extras: HistGB 0.0246 to 0.0244, GRU 0.0279 to 0.0275; Phase 6 addendum, `hourly/results/with_aux_mid/`).

**Table X. Always-up by horizon** (`daily/horizon_always_up.py`; share of up days).

| Horizon (trading days) | Test 2025-01-09 to 2026-10-23 | All years 2004-2026 | Test n_eff |
|---|---|---|---|
| 1 | 55.1% | 52.7% | 448 |
| 5 | 58.8% | 55.9% | 88 |
| 20 | 62.0% | 56.3% | 21 |
| 60 | 75.6% | 62.9% | 6 |
| 120 | 71.1% | 68.9% | 2 |

**Live record (descriptive, 2026-11-17).** 12 forecasts logged since 2026-10-28, 11 scored: MAE \$48.49 against \$48.21 for the simple guess; 9 of 11 inside the 90% range. Far too few days to conclude anything.

### E6 results (Tables E6-1 to E6-6) (2026-11-24; protocol in §15.8; code `daily/volatility_har_experiment.py`; files `daily/results/paper_e6/e6_summary.json` and `e6_primary_forecasts.csv`)
*Primary window: 428 origins (2025-01-09 to 2026-09-24; the hourly data end on 2026-09-25). Long window: 4,118 origins (2010-01-04 to 2026-09-24; no LSTM). Settlement boundary chosen on the development window: 13:30 ET. "QLIKE minus EWMA" is negative when the model is better; the DM test uses a Bartlett HAC variance with 5 lags and the Bonferroni level for the six comparisons is 0.0083. The multiplier "matched" means the EWMA and GARCH forecasts were rescaled to the RV level with a constant fitted on the training window.*

**Table E6-1. Primary window, target = realised variance of the next day.**

| Variance forecast | QLIKE | QLIKE minus EWMA 0.94 (matched) | DM p | Mincer-Zarnowitz R² | MZ slope | Mean forecast / mean target |
|---|---|---|---|---|---|---|
| EWMA 0.94 (matched) | -7.564 | +0.000 | – | 0.172 | 0.96 | 1.00 |
| GARCH(1,1) (matched) | -7.545 | +0.020 | 0.356 | 0.189 | 1.11 | 0.82 |
| EWMA 0.94 of RV | -7.552 | +0.012 | 0.381 | 0.136 | 0.86 | 0.97 |
| HAR-RV levels | -7.590 | -0.025 | 0.343 | 0.209 | 0.53 | 0.92 |
| HAR-RV logs | -7.594 | -0.030 | 0.147 | 0.239 | 1.27 | 0.80 |
| HAR-RV-X | -7.625 | -0.060 | 0.103 | 0.302 | 1.33 | 0.81 |
| LSTM-RV | -7.493 | +0.071 | 0.171 | 0.130 | 1.60 | 0.67 |
| EWMA 0.94 (shipped, unmatched) | -7.566 | -0.001 | 0.882 | 0.185 | 0.97 | 1.08 |

**Table E6-2. Long window, target = realised variance of the next day.**

| Variance forecast | QLIKE | QLIKE minus EWMA 0.94 (matched) | DM p | Mincer-Zarnowitz R² | MZ slope | Mean forecast / mean target |
|---|---|---|---|---|---|---|
| EWMA 0.94 (matched) | -8.310 | +0.000 | – | 0.187 | 0.81 | 1.05 |
| GARCH(1,1) (matched) | -8.295 | +0.015 | 0.178 | 0.189 | 1.08 | 1.02 |
| EWMA 0.94 of RV | -8.321 | -0.011 | 0.133 | 0.175 | 0.89 | 1.00 |
| HAR-RV levels | -8.335 | -0.025 | 0.001 | 0.275 | 0.96 | 1.00 |
| HAR-RV logs | -8.339 | -0.029 | <0.001 | 0.246 | 1.25 | 0.94 |
| HAR-RV-X | -8.364 | -0.054 | <0.001 | 0.289 | 1.28 | 0.94 |
| EWMA 0.94 (shipped, unmatched) | -8.315 | -0.005 | 0.052 | 0.195 | 0.81 | 1.10 |

**Table E6-3. The same forecasts scored against the squared futures return of the next day** (the product's own, very noisy, target; primary window, then long window).

| Variance forecast | QLIKE | QLIKE minus EWMA 0.94 (matched) | DM p | Mincer-Zarnowitz R² | MZ slope | Mean forecast / mean target |
|---|---|---|---|---|---|---|
| EWMA 0.94 (matched) | -7.311 | +0.000 | – | 0.019 | 0.56 | 0.92 |
| GARCH(1,1) (matched) | -7.222 | +0.089 | 0.042 | 0.019 | 0.60 | 0.75 |
| EWMA 0.94 of RV | -7.350 | -0.039 | 0.462 | 0.034 | 0.74 | 0.89 |
| HAR-RV levels | -7.362 | -0.051 | 0.584 | 0.086 | 0.59 | 0.84 |
| HAR-RV logs | -7.367 | -0.056 | 0.435 | 0.074 | 1.21 | 0.73 |
| HAR-RV-X | -7.379 | -0.068 | 0.312 | 0.081 | 1.19 | 0.74 |
| LSTM-RV | -7.229 | +0.082 | 0.150 | 0.030 | 1.32 | 0.61 |
| EWMA 0.94 (shipped, unmatched) | -7.337 | -0.026 | 0.115 | 0.020 | 0.55 | 0.98 |

| Variance forecast | QLIKE | QLIKE minus EWMA 0.94 (matched) | DM p | Mincer-Zarnowitz R² | MZ slope | Mean forecast / mean target |
|---|---|---|---|---|---|---|
| EWMA 0.94 (matched) | -8.154 | +0.000 | – | 0.042 | 0.70 | 0.95 |
| GARCH(1,1) (matched) | -8.153 | +0.001 | 0.955 | 0.035 | 0.84 | 0.93 |
| EWMA 0.94 of RV | -8.190 | -0.037 | 0.034 | 0.054 | 0.90 | 0.91 |
| HAR-RV levels | -8.199 | -0.045 | 0.007 | 0.093 | 1.01 | 0.91 |
| HAR-RV logs | -8.191 | -0.038 | 0.011 | 0.072 | 1.22 | 0.85 |
| HAR-RV-X | -8.237 | -0.083 | <0.001 | 0.085 | 1.26 | 0.86 |
| EWMA 0.94 (shipped, unmatched) | -8.169 | -0.015 | <0.001 | 0.042 | 0.68 | 1.00 |

**Table E6-4. QLIKE by year, long window** (target = realised variance).

| Year | Days | EWMA 0.94 (matched) | GARCH(1,1) (matched) | HAR-RV logs | HAR-RV-X |
|---|---|---|---|---|---|
| 2010 | 252 | -8.140 | -8.138 | -8.165 | -8.166 |
| 2011 | 252 | -7.872 | -7.874 | -7.915 | -7.924 |
| 2012 | 250 | -8.345 | -8.314 | -8.369 | -8.372 |
| 2013 | 252 | -7.843 | -7.842 | -7.820 | -7.849 |
| 2014 | 252 | -8.522 | -8.534 | -8.545 | -8.556 |
| 2015 | 252 | -8.360 | -8.394 | -8.380 | -8.424 |
| 2016 | 250 | -8.137 | -8.182 | -8.221 | -8.243 |
| 2017 | 251 | -9.070 | -9.017 | -9.071 | -9.101 |
| 2018 | 250 | -9.158 | -9.169 | -9.190 | -9.193 |
| 2019 | 252 | -8.919 | -8.845 | -8.934 | -8.982 |
| 2020 | 230 | -7.821 | -7.632 | -7.925 | -7.918 |
| 2021 | 231 | -8.374 | -8.390 | -8.379 | -8.415 |
| 2022 | 251 | -8.282 | -8.304 | -8.335 | -8.369 |
| 2023 | 251 | -8.655 | -8.611 | -8.630 | -8.670 |
| 2024 | 207 | -8.405 | -8.449 | -8.470 | -8.542 |
| 2025 | 252 | -7.995 | -7.970 | -8.027 | -8.012 |
| 2026 | 183 | -7.000 | -6.974 | -7.026 | -7.117 |

HAR-RV logs has the lower QLIKE than EWMA 0.94 in 15 of 17 years.

**Table E6-5. Pre-specified pairs** (first model minus second).

| Window | Pair (first minus second, QLIKE on RV) | QLIKE difference | DM p |
|---|---|---|---|
| primary | HAR-RV-X vs HAR-RV logs | -0.0303 | 0.238 |
| primary | LSTM-RV vs HAR-RV-X | +0.1317 | 0.072 |
| primary | HAR-RV logs vs GARCH(1,1) (matched) | -0.0497 | 0.089 |
| long | HAR-RV-X vs HAR-RV logs | -0.0251 | <0.001 |
| long | HAR-RV logs vs GARCH(1,1) (matched) | -0.0440 | <0.001 |

**Table E6-6. Realised variance on scheduled-release days, 2004-2026** (descriptive; the day-of flags use the futures-day label of the target day).

| Day type | Event days | Mean RV on event days / other days | Mean log difference | Welch t |
|---|---|---|---|---|
| NFP day | 249 | 1.21 | +0.313 | +5.4 |
| CPI day | 262 | 1.07 | +0.120 | +2.0 |
| FOMC decision day | 193 | 0.74 | -0.279 | -3.8 |
| day after FOMC | 194 | 2.13 | +0.738 | +10.5 |

**Reading (written after the results; nothing in the protocol was changed).**
1. *Risk is predictable, and the R² is positive for every model.* Against the realised variance the Mincer-Zarnowitz R² of the level-matched shipped EWMA is 0.17 (primary) and 0.19 (long); HAR-RV-X reaches 0.30 and 0.29. This is a risk result: the R² of the *return* forecast (Table III, -0.011) is unchanged.
2. *HAR-RV beats the shipped EWMA, but the evidence depends on the window.* On the long window all three HAR variants have a lower QLIKE (HAR-RV-X -0.054, logs -0.029, levels -0.025; DM p <0.001, <0.001, 0.001, all below the Bonferroni level) and HAR-RV logs wins in 15 of 17 years. On the 428-day primary window the same models are also better (-0.060, -0.030, -0.025) but not significantly (p = 0.103, 0.147, 0.343).
3. *Release days matter.* HAR-RV-X beats HAR-RV logs on the long window (-0.025, p <0.001); on the primary window the sign is the same and the difference is not significant (p = 0.238). Realised variance is 1.21 times higher on NFP days (Welch t = 5.4), 1.07 times on CPI days (t = 2.0), 0.74 times on FOMC decision days (t = -3.8) and 2.13 times on the day after (t = 10.5): the 14:00 ET announcement falls after the 13:30 ET settlement, so its reaction lands in the next futures day (the model includes both dummies).
4. *GARCH(1,1) and the EWMA of realised variance are no better than the shipped EWMA* (primary p = 0.356 and 0.381; long p = 0.178 and 0.133); HAR-RV logs beats GARCH on the long window (-0.044, p <0.001).
5. *The LSTM does not help.* LSTM-RV has QLIKE +0.071 against the shipped EWMA (p = 0.171), +0.132 against HAR-RV-X with the same inputs (p = 0.072) and a Mincer-Zarnowitz R² of 0.13.
6. *Against the squared return the ranking is the same on the long window* (HAR-RV-X -0.083, p <0.001; R² 0.042 to 0.085) and not significant on the primary window (p = 0.312); the squared return is a noisy proxy, so low R² there is expected.
7. *Caveats.* The proxy is hourly bid-based realised variance with a settlement boundary chosen on 2006-2007; the DM test has no small-sample correction; the primary window lies inside the long window; the HAR logs and HAR-RV-X forecasts are low on the primary window (mean forecast 0.81 of the mean realised variance) because 2025-2026 variance was higher than in the training windows; HAR-RV needs hourly data, which the daily pipeline does not collect, so using it in the shipped band would be a new data feed.
8. *Deviations from the protocol.* None in the models or the scoring. The experiment script stopped three times at its consistency check with the shipped EWMA before any model had run (the shipped run lacks the embargo days at the split boundaries and the trading day 2026-04-06, so the two EWMA paths differ by up to 10% in the first rows and up to 3% later); the check was loosened in code each time and no model result was seen before the final run. No exploratory variant was run after the results.

### 15.6 Evaluation pitfalls found in this project (the audit angle)
**Table XI. Evaluation pitfalls.**

| # | Pitfall | What happened | Size of the effect | Where |
|---|---|---|---|---|
| 1 | Price-level target with train-only scalers | When gold rose above \$4,000 in 2025-26, inputs and targets left the range the scalers and the model had seen | MAE about \$639 against about \$51 for trivial baselines; fixed with a log-return target and scale-free features | Phase 7 |
| 2 | Majority-class accuracy | Gold trends up, so "always up" scores 55.2% (daily test) and 51.5 / 52.9 / 55.1% (hourly, 1 / 4 / 24 bars); the LSTM says "up" on 88% of days | The LSTM is 3.3 points below always-up; "76% accuracy at 60 days" needs no model (always-up 75.6%) | Phases 7-8, Tables IV and X |
| 3 | Interval calibrated on the test window | "Covers 90% of test days" was true by construction | The honest empirical-quantile band covers 85.9%; the EWMA of past errors 89.5% | Phase 10, Table VII |
| 4 | Bid-only prices | A gradient-boosting "56-60% on confident predictions" came from the hours around the daily break, where the spread widens 2-3 times | Top-5% edge over always-up +6.7 points (bid) against +0.2 (mid) | Phase 6 |
| 5 | Silent partial downloads | The hourly downloader saved partial files on HTTP 429 | Caught by row-count and coverage checks | Phase 5 |
| 6 | Future information in aggregate columns | The Silver `_1w` and `_1m` columns hold the period's final values | Not used; look-ahead tests added | Phase 7 |
| 7 | Wrong direction definition | The old code compared day-to-day changes of predictions with those of prices instead of the signs of returns | Definition fixed | Phase 7 |
| 8 | A "fix" verified on the latest weeks only | The MAE fix held for three weeks of the test period | Re-checked over the whole period | Phase 7 |
| 9 | Picking the best of many variants | FOMC+CPI shows 57.5% direction accuracy | p = 0.367; a handful of p-values between 0.006 and 0.04 are what multiple testing alone produces | Phase 9, Table V |
| 10 | Test choices that could change the verdict | DM lag 0 against 24 | p = 0.40 against 0.385, conclusion unchanged | Phase 10, Table III |
| 11 | Comparing alternatives on the window used to report the winner | The EWMA decay was fixed in advance, the alternatives were compared on the same window | The 89.5% is slightly optimistic | Phase 10 |
| 12 | Claiming more than was live | "Every forecast is saved before the day" is true of live forecasts only; the replayed ones were not | Wording corrected | Phase 13 |
| 13 | R² on price levels | The R² of the forecast price against the actual price is 0.989 for the LSTM and 0.989 for the no-change forecast (0.9892 against 0.9894), because prices are highly autocorrelated; on returns the LSTM's R² is -0.014 | A positive-looking R² that no benchmark would fail to reach; the early "R² about 0.87" was this trap (§12 item 2) | Phase 7, Table III |

### 15.7 Threats to validity and limitations (write these into the paper)
- **Construct.** Gold futures (`GC=F`, a continuous contract) stand in for spot XAU/USD. The headline MAE is in dollars (it depends on the price level), the models predict log-returns, and the DM test uses squared return errors: say which loss each statement refers to (E1 adds an absolute-error DM and bootstrap intervals).
- **Internal.** Look-ahead tests cannot prove the absence of leakage. Hyperparameters (window 20, 32 units, dropout 0.2, learning rate, patience) are not tuned inside the walk-forward loop; the repository holds six static-training variants (`daily/results/model_results_lstm_w*.json`: window 20 or 60, horizons 1 and 5, training from 2012) and no other tuning record, so the paper must state which settings were fixed beforehand (question for the owner, §15.15). Seed variance is not reported (E2).
- **Statistical.** n = 449 (n_eff 88 and 21 at 5 and 20 days); about twenty tests are reported without a multiplicity correction (E3); the DM test assumes forecasts could be compared as such although the models are re-estimated every 7 days (Giacomini-White is the matching framework); the test window is one rising market (55.2% up days); the interval's 89.5% is slightly optimistic (pitfall 11).
- **External.** One asset, one period, one frequency (plus a side experiment on hourly data); a null here says nothing about other assets, regimes or information sets.
- **Reproducibility.** TensorFlow training is not bit-reproducible across machines; vendor data can be revised (a continuous futures series can be rewritten); raw data cannot be redistributed; the live log can be overwritten (§12 item 19); the serving model has not been retrained since 2026-10-29.
- **Ethics.** The work is a student project and not trading advice; no personal data is involved; AI assistance must be disclosed (§15.12).

### 15.8 Extra experiments that would strengthen the paper (plan)
Each runs in its own results folder (for example `daily/results/paper_*/`) so that the dashboard's 449-day window and the numbers above stay consistent; add the code under `daily/` with tests, and record outcomes here.

| ID | Experiment | Reviewer question it answers | Design | Effort | Expectation |
|---|---|---|---|---|---|
| E1 | Bootstrap intervals and better DM tests | Is MAE equal to no-change, and is the test appropriate? | Moving-block bootstrap CI of the MAE and RMSE differences; DM on absolute price errors; Harvey-Leybourne-Newbold correction; Giacomini-White | hours | Null stays; CI width shows the power |
| E2 | Seed variance and ablations | Is the null robust to hyperparameters? | 10 seeds; GRU, MLP and a linear model on the same features; window 5 / 10 / 20 / 60; units 16 / 32 / 64 | a day of compute | Null stays |
| E3 | Multiplicity correction | Could a "best" variant be luck, and is anything better than no-change after correction? | Hansen SPA and the model confidence set over every model and input set tried (about 30 candidates); the `arch` package has both | 1-2 days | Nothing survives |
| E4 | Long window and power analysis | Does the null hold across regimes? | Walk-forward over about 2010-2026 (retrain every 20 days to cut compute); minimum detectable MAE gain at n = 449 | several days | Null stays; shows low power in 2025-26 |
| E5 | Multi-asset replication | Is it gold-specific? | Same pipeline on silver, EUR/USD, S&P 500, 10-year yield | several days | Null stays; supports a general audit |
| E6 | Volatility track (the positive-result candidate) | Is risk predictable when price is not? | Realised volatility from the hourly data; HAR-RV with NFP, FOMC and CPI days against EWMA, GARCH and an LSTM; QLIKE and DM | done 2026-11-24 (452 s of compute) | **Done, protocol frozen first:** positive R² for risk; HAR-RV beats the shipped EWMA on 2010-2026 (p < 0.001) and, not significantly, on the 428-day window; the LSTM does not help (Tables E6-1 to E6-6) |
| E7 | Conformal intervals | Can distribution-free intervals match EWMA? | Split conformal and adaptive conformal inference around the LSTM and no-change centres | 2-3 days | Comparable coverage |
| E8 | Conditional coverage | Are violations independent? | Christoffersen independence and conditional-coverage tests on the current bands | hours | Mild clustering likely |
| E9 | Forecast encompassing | Does the LSTM add anything to no-change? | Mincer-Zarnowitz regression of realised on forecast returns; encompassing test | hours | Slope near 0 |
| E10 | Live-log hardening | Can the live record be trusted? | S3 versioning with Object Lock or a daily commit of the log; back it up (§12 items 19-20) | hours | Needed before citing the live record |
| E11 | Frozen protocol | Were choices made before seeing results? | Commit and tag a protocol file before any long-window or multi-asset run | an hour | Credibility |
| E12 | Search log | How many variants were tried? | Count every model, input set, horizon and interval method tried (needed for E3 and for deflated claims) | an hour | About 30 |

**E6 protocol (frozen 2026-11-24 and committed before the first run on any reported window; code `daily/volatility_har_experiment.py`, tests `daily/test_volatility_har.py`, results in `daily/results/paper_e6/`).**
- *Question:* is next-day gold risk better forecast by realised volatility from hourly data (HAR-RV, with scheduled-release days) than by the shipped EWMA of daily squared returns, and does an LSTM add anything?
- *Target:* RV(d), the sum of squared hourly log-returns of the Dukascopy XAU/USD **bid** series over the window that ends at the futures settlement of day d (bars with fewer than 12 per day are dropped). The window boundary is whichever of 13:30 ET and 17:00 ET makes the summed hourly returns correlate best with the daily GC=F return on the development window 2006-2007 (a measurement check, not a forecasting result; 13:30 ET won, correlation 0.951 against 0.915).
- *Models* (variance of day t+1 forecast at the close of day t): EWMA 0.94 on daily returns (the shipped one) and GARCH(1,1), both level-matched to RV with a multiplier fitted on the training window; EWMA 0.94 on RV; HAR-RV in levels (OLS, floored at 10% of the training mean) and in logs (bias-corrected); HAR-RV-X (logs plus dummies for the target day: NFP, CPI, FOMC, and FOMC on day t because the 14:00 ET announcement can fall either side of the settlement); LSTM-RV (LSTM(32), 20-day window of the HAR-RV-X inputs, 3 seeds, the same recipe as the return model).
- *Fitting:* the last 1,000 trading days known at the block's first origin; refit every 7 origins on the primary window and every 21 on the long window; the LSTM runs on the primary window only.
- *Windows:* PRIMARY = the origins of the main walk-forward run (2025-01-09 onwards) that have RV for the next day (the hourly data end 2026-09-25); LONG = 2010-01-01 onwards. One common sample per window.
- *Scoring:* QLIKE (ln f + RV / f; Patton 2011) against RV(t+1) as the primary target and against the squared futures return as the secondary target (the product's own target, as in Table VIII); Diebold-Mariano test against EWMA 0.94 with a Bartlett HAC variance (5 lags); also MSE, Mincer-Zarnowitz R² and slope, and bias. The family of comparisons against EWMA has six models, so the Bonferroni level is 0.05 / 6 = 0.0083. Pre-specified pairs: HAR-RV-X against HAR-RV logs (do event days help?) and LSTM-RV against HAR-RV-X (does the LSTM help?).
- *Disclosure:* a debugging run on the development window 2006-2007 was made before this freeze; nothing was chosen from it except the code fixes and the boundary check above. Results will be reported whether or not they favour the HAR models.

**Angles (from the 2026-11-09 discussion).** (A) Audit paper: this project's pitfalls plus E3-E5; (B) "predictable risk, unpredictable price": E6 with E7-E8 as the positive-result section; (C) a multi-asset benchmark that includes time-series foundation models. Suggested: A as the backbone, B as the positive-result section.

### 15.9 Related work map and bibliography
*Every DOI below was looked up in Crossref and every arXiv identifier in the arXiv API on 2026-11-17 (title, authors, journal, volume and pages matched). Items without a DOI are marked. Only these references may be cited; add a new one only after checking it the same way (§15.13). The Welch-Goyal paper is cited as Welch and Goyal, the order in the journal.*

**What to claim from the literature.** (1) Simple benchmarks are hard to beat out of sample [1], [3], [38]-[39]: the no-change forecast is the right yardstick and a null is the expected result under market efficiency [36]-[37]. (2) Evaluation of forecasts needs care: accuracy measures [5], pitfalls [4], validation schemes [6]-[7], overlapping labels and backtest overfitting [8]-[9], multiple testing [10]-[13]. (3) Comparing forecasts: Diebold-Mariano [14] with its small-sample correction [15] and caveats [16], conditional predictive ability for rolling estimation [17]. (4) Intervals and volatility: coverage tests [18]-[19], interval score [20], EWMA [21], GARCH [22], filtered historical simulation [23], QLIKE [24], HAR-RV [25], conformal intervals [26]-[27], Wilson interval [28]. (5) Deep learning in finance and for gold [29]-[35]: report that headline gains rarely survive a no-change benchmark, and cite [32] (LSTM for gold) as the closest prior work.

| # (★ = core for the short paper) | Reference (IEEE style) | Use in the paper |
|---|---|---|
| | **Forecast evaluation, benchmarks and pitfalls** | |
| 1 ★ | I. Welch and A. Goyal, "A comprehensive look at the empirical performance of equity premium prediction," *Review of Financial Studies*, vol. 21, no. 4, pp. 1455–1508, 2008. doi: `10.1093/rfs/hhm014` | Simple benchmarks (the historical mean) are hard to beat out of sample: motivates the no-change benchmark |
| 2 | S. Gu, B. Kelly, and D. Xiu, "Empirical asset pricing via machine learning," *Review of Financial Studies*, vol. 33, no. 5, pp. 2223–2273, 2020. doi: `10.1093/rfs/hhaa009` | Out-of-sample R-squared convention; machine learning can help in the cross-section (contrast with our single-asset null) |
| 3 ★ | S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "Statistical and machine learning forecasting methods: Concerns and ways forward," *PLOS ONE*, vol. 13, no. 3, Art. no. e0194889, 2018. doi: `10.1371/journal.pone.0194889` | Machine-learning forecasts often fail to beat simple statistical benchmarks |
| 4 ★ | H. Hewamalage, K. Ackermann, and C. Bergmeir, "Forecast evaluation for data scientists: common pitfalls and best practices," *Data Mining and Knowledge Discovery*, vol. 37, no. 2, pp. 788–832, 2023. doi: `10.1007/s10618-022-00894-5` | Catalogue of evaluation pitfalls; frames our audit |
| 5 ★ | R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *International Journal of Forecasting*, vol. 22, no. 4, pp. 679–688, 2006. doi: `10.1016/j.ijforecast.2006.03.001` | Choice of accuracy measures (MAE, RMSE, MAPE, scaled errors) |
| 6 | C. Bergmeir and J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Information Sciences*, vol. 191, pp. 192–213, 2012. doi: `10.1016/j.ins.2011.12.028` | Validation schemes for time series (why we use walk-forward) |
| 7 | C. Bergmeir, R. J. Hyndman, and B. Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction," *Computational Statistics & Data Analysis*, vol. 120, pp. 70–83, 2018. doi: `10.1016/j.csda.2017.11.003` | When ordinary cross-validation is valid for autoregressive models |
| 8 ★ | M. López de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ, USA: Wiley, 2018. ISBN 978-1-119-48208-6 | Embargo and purging of overlapping labels; backtest overfitting |
| 9 | D. H. Bailey and M. López de Prado, "The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality," *Journal of Portfolio Management*, vol. 40, no. 5, pp. 94–107, 2014. doi: `10.3905/jpm.2014.40.5.094` | Selection bias when many variants are tried |
| 10 | C. R. Harvey, Y. Liu, and H. Zhu, "… and the cross-section of expected returns," *Review of Financial Studies*, vol. 29, no. 1, pp. 5–68, 2016. doi: `10.1093/rfs/hhv059` | Multiple testing: nominal p-values overstate evidence when many factors are tried |
| 11 | H. White, "A reality check for data snooping," *Econometrica*, vol. 68, no. 5, pp. 1097–1126, 2000. doi: `10.1111/1468-0262.00152` | Data-snooping correction (bootstrap reality check) |
| 12 | P. R. Hansen, "A test for superior predictive ability," *Journal of Business & Economic Statistics*, vol. 23, no. 4, pp. 365–380, 2005. doi: `10.1198/073500105000000063` | SPA test: planned correction over every model tried (experiment E3) |
| 13 | P. R. Hansen, A. Lunde, and J. M. Nason, "The model confidence set," *Econometrica*, vol. 79, no. 2, pp. 453–497, 2011. doi: `10.3982/ECTA5771` | Model confidence set (experiment E3) |
| | **Comparing forecasts** | |
| 14 ★ | F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995. doi: `10.1080/07350015.1995.10524599` | The Diebold-Mariano test used against the no-change forecast |
| 15 ★ | D. Harvey, S. Leybourne, and P. Newbold, "Testing the equality of prediction mean squared errors," *International Journal of Forecasting*, vol. 13, no. 2, pp. 281–291, 1997. doi: `10.1016/S0169-2070(96)00719-4` | Small-sample correction of the DM test (not yet applied; experiment E1) |
| 16 | F. X. Diebold, "Comparing predictive accuracy, twenty years later: A personal perspective on the use and abuse of Diebold-Mariano tests," *Journal of Business & Economic Statistics*, vol. 33, no. 1, p. 1, 2015. doi: `10.1080/07350015.2014.983236` | Correct and incorrect uses of DM tests (they compare forecasts, not models) |
| 17 ★ | R. Giacomini and H. White, "Tests of conditional predictive ability," *Econometrica*, vol. 74, no. 6, pp. 1545–1578, 2006. doi: `10.1111/j.1468-0262.2006.00718.x` | Predictive-ability tests that remain valid with rolling-window estimation, as in walk-forward retraining |
| | **Intervals, scoring rules and volatility** | |
| 18 ★ | P. H. Kupiec, "Techniques for verifying the accuracy of risk measurement models," *Journal of Derivatives*, vol. 3, no. 2, pp. 73–84, 1995. doi: `10.3905/jod.1995.407942` | Proportion-of-failures coverage test (Kupiec p) |
| 19 | P. F. Christoffersen, "Evaluating interval forecasts," *International Economic Review*, vol. 39, no. 4, pp. 841–862, 1998. doi: `10.2307/2527341` | Independence and conditional-coverage tests (experiment E8) |
| 20 ★ | T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359–378, 2007. doi: `10.1198/016214506000001437` | The interval score used to compare intervals |
| 21 ★ | J.P. Morgan/Reuters, *RiskMetrics - Technical Document, 4th ed.*. J.P. Morgan/Reuters, New York, 1996. (UNVERIFIED metadata: no DOI; check the page number of the lambda = 0.94 recommendation before citing it) | Origin of the EWMA decay 0.94 used for the shipped band |
| 22 | T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307–327, 1986. doi: `10.1016/0304-4076(86)90063-1` | GARCH(1,1) alternative |
| 23 | G. Barone-Adesi, K. Giannopoulos, and L. Vosper, "VaR without correlations for portfolios of derivative securities," *Journal of Futures Markets*, vol. 19, no. 5, pp. 583–602, 1999. doi: `10.1002/(SICI)1096-9934(199908)19:5<583::AID-FUT5>3.0.CO;2-S` | Filtered historical simulation alternative |
| 24 ★ | A. J. Patton, "Volatility forecast comparison using imperfect volatility proxies," *Journal of Econometrics*, vol. 160, no. 1, pp. 246–256, 2011. doi: `10.1016/j.jeconom.2010.03.034` | QLIKE is a robust loss for volatility forecasts |
| 25 | F. Corsi, "A simple approximate long-memory model of realized volatility," *Journal of Financial Econometrics*, vol. 7, no. 2, pp. 174–196, 2009. doi: `10.1093/jjfinec/nbp001` | HAR-RV model for the volatility track (experiment E6) |
| 26 | I. Gibbs and E. Candès, "Adaptive conformal inference under distribution shift," in *Advances in Neural Information Processing Systems 34 (NeurIPS)*, 2021. arXiv: `2106.00170` | Adaptive conformal intervals for non-stationary series (experiment E7) |
| 27 | C. Xu and Y. Xie, "Conformal prediction interval for dynamic time-series," in *Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139*, 2021. arXiv: `2010.09107` | Conformal intervals for time series (experiment E7) |
| 28 ★ | E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, 1927. doi: `10.1080/01621459.1927.10502953` | Wilson confidence interval for direction accuracy |
| | **Deep learning for finance and gold** | |
| 29 ★ | S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. doi: `10.1162/neco.1997.9.8.1735` | The LSTM |
| 30 ★ | T. Fischer and C. Krauss, "Deep learning with long short-term memory networks for financial market predictions," *European Journal of Operational Research*, vol. 270, no. 2, pp. 654–669, 2018. doi: `10.1016/j.ejor.2017.11.054` | LSTM for financial prediction; reported gains that later weakened |
| 31 ★ | O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005-2019," *Applied Soft Computing*, vol. 90, Art. no. 106181, 2020. doi: `10.1016/j.asoc.2020.106181` | Survey of deep learning for financial time series |
| 32 ★ | I. E. Livieris, E. Pintelas, and P. Pintelas, "A CNN-LSTM model for gold price time-series forecasting," *Neural Computing and Applications*, vol. 32, no. 23, pp. 17351–17360, 2020. doi: `10.1007/s00521-020-04867-x` | Closest prior work: deep learning for gold price forecasting |
| 33 | A. Vidal and W. Kristjanpoller, "Gold volatility prediction using a CNN-LSTM approach," *Expert Systems with Applications*, vol. 157, Art. no. 113481, 2020. doi: `10.1016/j.eswa.2020.113481` | Deep learning for gold volatility (relevant to the volatility track) |
| 34 | D. G. Baur and B. M. Lucey, "Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold," *The Financial Review*, vol. 45, no. 2, pp. 217–229, 2010. doi: `10.1111/j.1540-6288.2010.00244.x` | Why gold is studied (hedge and safe haven) |
| 35 | D. G. Baur and T. K. McDermott, "Is gold a safe haven? International evidence," *Journal of Banking & Finance*, vol. 34, no. 8, pp. 1886–1898, 2010. doi: `10.1016/j.jbankfin.2009.12.008` | Safe-haven evidence across countries |
| | **Market efficiency** | |
| 36 ★ | E. F. Fama, "Efficient capital markets: A review of theory and empirical work," *Journal of Finance*, vol. 25, no. 2, pp. 383–417, 1970. doi: `10.2307/2325486` | Efficient-market hypothesis: why a no-change forecast is hard to beat |
| 37 ★ | A. Timmermann and C. W. J. Granger, "Efficient market hypothesis and forecasting," *International Journal of Forecasting*, vol. 20, no. 1, pp. 15–27, 2004. doi: `10.1016/S0169-2070(03)00012-8` | Forecasting under market efficiency; data-mining risk |
| 38 | R. A. Meese and K. Rogoff, "Empirical exchange rate models of the seventies: Do they fit out of sample?," *Journal of International Economics*, vol. 14, no. 1-2, pp. 3–24, 1983. doi: `10.1016/0022-1996(83)90017-X` | Classic: structural models do not beat a random walk out of sample |
| 39 | A. W. Lo, "The adaptive markets hypothesis: Market efficiency from an evolutionary perspective," *Journal of Portfolio Management*, vol. 30, no. 5, pp. 15–29, 2004. doi: `10.3905/jpm.2004.442611` | Predictability may vary over time (limits of a null result) |

<details><summary>BibTeX for all 39 references (verified 2026-11-17)</summary>

```bibtex
@article{welch2008,
  author = {Welch, Ivo and Goyal, Amit},
  title = {A comprehensive look at the empirical performance of equity premium prediction},
  journal = {Review of Financial Studies},
  year = {2008},
  volume = {21},
  number = {4},
  pages = {1455--1508},
  doi = {10.1093/rfs/hhm014}
}

@article{gu2020,
  author = {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  title = {Empirical asset pricing via machine learning},
  journal = {Review of Financial Studies},
  year = {2020},
  volume = {33},
  number = {5},
  pages = {2223--2273},
  doi = {10.1093/rfs/hhaa009}
}

@article{makridakis2018,
  author = {Makridakis, Spyros and Spiliotis, Evangelos and Assimakopoulos, Vassilios},
  title = {Statistical and machine learning forecasting methods: Concerns and ways forward},
  journal = {PLOS ONE},
  year = {2018},
  volume = {13},
  number = {3},
  pages = {e0194889},
  doi = {10.1371/journal.pone.0194889}
}

@article{hewamalage2023,
  author = {Hewamalage, Hansika and Ackermann, Klaus and Bergmeir, Christoph},
  title = {Forecast evaluation for data scientists: common pitfalls and best practices},
  journal = {Data Mining and Knowledge Discovery},
  year = {2023},
  volume = {37},
  number = {2},
  pages = {788--832},
  doi = {10.1007/s10618-022-00894-5}
}

@article{hyndman2006,
  author = {Hyndman, Rob J. and Koehler, Anne B.},
  title = {Another look at measures of forecast accuracy},
  journal = {International Journal of Forecasting},
  year = {2006},
  volume = {22},
  number = {4},
  pages = {679--688},
  doi = {10.1016/j.ijforecast.2006.03.001}
}

@article{bergmeir2012,
  author = {Bergmeir, Christoph and Benítez, José M.},
  title = {On the use of cross-validation for time series predictor evaluation},
  journal = {Information Sciences},
  year = {2012},
  volume = {191},
  pages = {192--213},
  doi = {10.1016/j.ins.2011.12.028}
}

@article{bergmeir2018,
  author = {Bergmeir, Christoph and Hyndman, Rob J. and Koo, Bonsoo},
  title = {A note on the validity of cross-validation for evaluating autoregressive time series prediction},
  journal = {Computational Statistics \& Data Analysis},
  year = {2018},
  volume = {120},
  pages = {70--83},
  doi = {10.1016/j.csda.2017.11.003}
}

@book{lopezdeprado2018,
  author = {López de Prado, Marcos},
  title = {Advances in Financial Machine Learning},
  publisher = {Wiley},
  address = {Hoboken, NJ, USA},
  year = {2018},
  isbn = {978-1-119-48208-6}
}

@article{bailey2014,
  author = {Bailey, David H. and López de Prado, Marcos},
  title = {The deflated {S}harpe ratio: Correcting for selection bias, backtest overfitting, and non-normality},
  journal = {Journal of Portfolio Management},
  year = {2014},
  volume = {40},
  number = {5},
  pages = {94--107},
  doi = {10.3905/jpm.2014.40.5.094}
}

@article{harvey2016,
  author = {Harvey, Campbell R. and Liu, Yan and Zhu, Heqing},
  title = {\ldots and the cross-section of expected returns},
  journal = {Review of Financial Studies},
  year = {2016},
  volume = {29},
  number = {1},
  pages = {5--68},
  doi = {10.1093/rfs/hhv059}
}

@article{white2000,
  author = {White, Halbert},
  title = {A reality check for data snooping},
  journal = {Econometrica},
  year = {2000},
  volume = {68},
  number = {5},
  pages = {1097--1126},
  doi = {10.1111/1468-0262.00152}
}

@article{hansen2005,
  author = {Hansen, Peter R.},
  title = {A test for superior predictive ability},
  journal = {Journal of Business \& Economic Statistics},
  year = {2005},
  volume = {23},
  number = {4},
  pages = {365--380},
  doi = {10.1198/073500105000000063}
}

@article{hansen2011,
  author = {Hansen, Peter R. and Lunde, Asger and Nason, James M.},
  title = {The model confidence set},
  journal = {Econometrica},
  year = {2011},
  volume = {79},
  number = {2},
  pages = {453--497},
  doi = {10.3982/ECTA5771}
}

@article{diebold1995,
  author = {Diebold, Francis X. and Mariano, Roberto S.},
  title = {Comparing predictive accuracy},
  journal = {Journal of Business \& Economic Statistics},
  year = {1995},
  volume = {13},
  number = {3},
  pages = {253--263},
  doi = {10.1080/07350015.1995.10524599}
}

@article{harvey1997,
  author = {Harvey, David and Leybourne, Stephen and Newbold, Paul},
  title = {Testing the equality of prediction mean squared errors},
  journal = {International Journal of Forecasting},
  year = {1997},
  volume = {13},
  number = {2},
  pages = {281--291},
  doi = {10.1016/S0169-2070(96)00719-4}
}

@article{diebold2015,
  author = {Diebold, Francis X.},
  title = {Comparing predictive accuracy, twenty years later: A personal perspective on the use and abuse of {Diebold-Mariano} tests},
  journal = {Journal of Business \& Economic Statistics},
  year = {2015},
  volume = {33},
  number = {1},
  pages = {1},
  doi = {10.1080/07350015.2014.983236}
}

@article{giacomini2006,
  author = {Giacomini, Raffaella and White, Halbert},
  title = {Tests of conditional predictive ability},
  journal = {Econometrica},
  year = {2006},
  volume = {74},
  number = {6},
  pages = {1545--1578},
  doi = {10.1111/j.1468-0262.2006.00718.x}
}

@article{kupiec1995,
  author = {Kupiec, Paul H.},
  title = {Techniques for verifying the accuracy of risk measurement models},
  journal = {Journal of Derivatives},
  year = {1995},
  volume = {3},
  number = {2},
  pages = {73--84},
  doi = {10.3905/jod.1995.407942}
}

@article{christoffersen1998,
  author = {Christoffersen, Peter F.},
  title = {Evaluating interval forecasts},
  journal = {International Economic Review},
  year = {1998},
  volume = {39},
  number = {4},
  pages = {841--862},
  doi = {10.2307/2527341}
}

@article{gneiting2007,
  author = {Gneiting, Tilmann and Raftery, Adrian E.},
  title = {Strictly proper scoring rules, prediction, and estimation},
  journal = {Journal of the American Statistical Association},
  year = {2007},
  volume = {102},
  number = {477},
  pages = {359--378},
  doi = {10.1198/016214506000001437}
}

@techreport{riskmetrics1996,
  author = {{J.P. Morgan/Reuters}},
  title = {RiskMetrics - Technical Document, 4th ed.},
  institution = {J.P. Morgan/Reuters, New York},
  year = {1996},
  note = {4th ed.; no DOI}
}

@article{bollerslev1986,
  author = {Bollerslev, Tim},
  title = {Generalized autoregressive conditional heteroskedasticity},
  journal = {Journal of Econometrics},
  year = {1986},
  volume = {31},
  number = {3},
  pages = {307--327},
  doi = {10.1016/0304-4076(86)90063-1}
}

@article{barone1999,
  author = {Barone-Adesi, Giovanni and Giannopoulos, Kostas and Vosper, Les},
  title = {{VaR} without correlations for portfolios of derivative securities},
  journal = {Journal of Futures Markets},
  year = {1999},
  volume = {19},
  number = {5},
  pages = {583--602},
  doi = {10.1002/(SICI)1096-9934(199908)19:5<583::AID-FUT5>3.0.CO;2-S}
}

@article{patton2011,
  author = {Patton, Andrew J.},
  title = {Volatility forecast comparison using imperfect volatility proxies},
  journal = {Journal of Econometrics},
  year = {2011},
  volume = {160},
  number = {1},
  pages = {246--256},
  doi = {10.1016/j.jeconom.2010.03.034}
}

@article{corsi2009,
  author = {Corsi, Fulvio},
  title = {A simple approximate long-memory model of realized volatility},
  journal = {Journal of Financial Econometrics},
  year = {2009},
  volume = {7},
  number = {2},
  pages = {174--196},
  doi = {10.1093/jjfinec/nbp001}
}

@misc{gibbs2021,
  author = {Gibbs, Isaac and Candès, Emmanuel},
  title = {Adaptive conformal inference under distribution shift},
  year = {2021},
  eprint = {2106.00170},
  archivePrefix = {arXiv},
  note = {Advances in Neural Information Processing Systems 34 (NeurIPS)}
}

@misc{xu2021,
  author = {Xu, Chen and Xie, Yao},
  title = {Conformal prediction interval for dynamic time-series},
  year = {2021},
  eprint = {2010.09107},
  archivePrefix = {arXiv},
  note = {Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139}
}

@article{wilson1927,
  author = {Wilson, Edwin B.},
  title = {Probable inference, the law of succession, and statistical inference},
  journal = {Journal of the American Statistical Association},
  year = {1927},
  volume = {22},
  number = {158},
  pages = {209--212},
  doi = {10.1080/01621459.1927.10502953}
}

@article{hochreiter1997,
  author = {Hochreiter, Sepp and Schmidhuber, Jürgen},
  title = {Long short-term memory},
  journal = {Neural Computation},
  year = {1997},
  volume = {9},
  number = {8},
  pages = {1735--1780},
  doi = {10.1162/neco.1997.9.8.1735}
}

@article{fischer2018,
  author = {Fischer, Thomas and Krauss, Christopher},
  title = {Deep learning with long short-term memory networks for financial market predictions},
  journal = {European Journal of Operational Research},
  year = {2018},
  volume = {270},
  number = {2},
  pages = {654--669},
  doi = {10.1016/j.ejor.2017.11.054}
}

@article{sezer2020,
  author = {Sezer, Omer Berat and Gudelek, Mehmet Ugur and Ozbayoglu, Ahmet Murat},
  title = {Financial time series forecasting with deep learning: A systematic literature review: 2005-2019},
  journal = {Applied Soft Computing},
  year = {2020},
  volume = {90},
  pages = {106181},
  doi = {10.1016/j.asoc.2020.106181}
}

@article{livieris2020,
  author = {Livieris, Ioannis E. and Pintelas, Emmanuel and Pintelas, Panagiotis},
  title = {A {CNN}-{LSTM} model for gold price time-series forecasting},
  journal = {Neural Computing and Applications},
  year = {2020},
  volume = {32},
  number = {23},
  pages = {17351--17360},
  doi = {10.1007/s00521-020-04867-x}
}

@article{vidal2020,
  author = {Vidal, Andrés and Kristjanpoller, Werner},
  title = {Gold volatility prediction using a {CNN}-{LSTM} approach},
  journal = {Expert Systems with Applications},
  year = {2020},
  volume = {157},
  pages = {113481},
  doi = {10.1016/j.eswa.2020.113481}
}

@article{baur2010,
  author = {Baur, Dirk G. and Lucey, Brian M.},
  title = {Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold},
  journal = {The Financial Review},
  year = {2010},
  volume = {45},
  number = {2},
  pages = {217--229},
  doi = {10.1111/j.1540-6288.2010.00244.x}
}

@article{baur2010b,
  author = {Baur, Dirk G. and McDermott, Thomas K.},
  title = {Is gold a safe haven? International evidence},
  journal = {Journal of Banking \& Finance},
  year = {2010},
  volume = {34},
  number = {8},
  pages = {1886--1898},
  doi = {10.1016/j.jbankfin.2009.12.008}
}

@article{fama1970,
  author = {Fama, Eugene F.},
  title = {Efficient capital markets: A review of theory and empirical work},
  journal = {Journal of Finance},
  year = {1970},
  volume = {25},
  number = {2},
  pages = {383--417},
  doi = {10.2307/2325486}
}

@article{timmermann2004,
  author = {Timmermann, Allan and Granger, Clive W. J.},
  title = {Efficient market hypothesis and forecasting},
  journal = {International Journal of Forecasting},
  year = {2004},
  volume = {20},
  number = {1},
  pages = {15--27},
  doi = {10.1016/S0169-2070(03)00012-8}
}

@article{meese1983,
  author = {Meese, Richard A. and Rogoff, Kenneth},
  title = {Empirical exchange rate models of the seventies: Do they fit out of sample?},
  journal = {Journal of International Economics},
  year = {1983},
  volume = {14},
  number = {1-2},
  pages = {3--24},
  doi = {10.1016/0022-1996(83)90017-X}
}

@article{lo2004,
  author = {Lo, Andrew W.},
  title = {The adaptive markets hypothesis: Market efficiency from an evolutionary perspective},
  journal = {Journal of Portfolio Management},
  year = {2004},
  volume = {30},
  number = {5},
  pages = {15--29},
  doi = {10.3905/jpm.2004.442611}
}
```

</details>

### 15.10 Figures and tables plan
| Item | Content | Data source | Status |
|---|---|---|---|
| Fig. 1 | Protocol and system diagram: data → features → walk-forward retraining → forecast + EWMA band → live scoring | §15.4 | to draw (TikZ or draw.io) |
| Fig. 2 | Actual close, LSTM forecast and 90% band (zoom on the last 120 test days) with the daily miss as bars | `wf_predictions.csv`, band from `volatility_forecast.csv` | to create (dashboard has an equivalent chart) |
| Fig. 3 | MAE by model with the DM p-values | Table III | to create |
| Fig. 4 | Coverage by interval method against the 90% goal, with the Kupiec 5% acceptance region | Table VII | to create |
| Fig. 5 | Input sets: AUC (with CI) against always-up | Table V | optional, for the extended version |
| Fig. 6 | Live record: scored forecasts with their ranges | `forecast_log.csv` (S3) | when the record is longer |
| Tables | I data and splits, II hyperparameters, III price accuracy, IV direction, V input sets, VI horizons, VII intervals, VIII volatility, IX hourly, X always-up by horizon, XI pitfalls (§15.6) | §15.3-15.6 | generate with a script from the result files, never type numbers |

For the 4-6 page conference paper use Tables I (compact), III, VII and XI (pitfalls, short form), Fig. 1 and either Fig. 2 or Fig. 4; move the rest to the extended version. Style: serif font (Times), 3.5 in or 7.2 in wide, vector PDF, readable in grey; keep the colour identity of the dashboard (the model in blue, `COLORS["model"]` in `dashboard/ui/theme.py`; actual in orange). Put the figure and table scripts in `paper/` when writing starts.

### 15.11 Writing plan
**Conference paper (JCSSE, 4-6 A4 pages, about 150-word abstract).** I Introduction (0.6 page: claims of deep learning for gold, the missing benchmark, RQ1-RQ3, contributions); II Related work (0.5: groups 1, 2, 5 of §15.9); III Data (0.4: Table I); IV Method (1.2: features, LSTM, walk-forward, baselines, metrics and tests, intervals); V Results (1.5: Tables III, V-compact, VII, Fig. 2 or 4); VI Discussion and limitations (0.5: why a null is expected, the pitfalls of §15.6, the threats of §15.7); VII Conclusion (0.2); Acknowledgements (AI-use statement, §15.12); References (about 20 of the 39).

**Journal article (IEEE Access, about 10-14 pages).** Add the extended related work, the ablations (E2), SPA and MCS (E3), the long window and power analysis (E4), the volatility track (E6), conformal intervals (E7) and the live record; keep the similarity to the conference paper below 35%.

**LaTeX skeleton.**
```latex
\documentclass[conference,a4paper]{IEEEtran}
\usepackage{cite,amsmath,graphicx,booktabs}
\begin{document}
\title{Is the LSTM Better Than Repeating Yesterday? A Causal Walk-Forward Audit of Next-Day Gold Price Forecasting}
\author{\IEEEauthorblockN{Anonymous for review}}   % JCSSE is double-blind: no names, no affiliations
\maketitle
\begin{abstract} ... \end{abstract}
\begin{IEEEkeywords} gold price, long short-term memory, walk-forward validation, prediction intervals, negative results \end{IEEEkeywords}
\section{Introduction} ...
\section*{Acknowledgment}  % disclose AI assistance here (and at submission, as JCSSE asks)
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
```

### 15.12 Reproducibility, availability and ethics
- **Environment (this machine, 2026-11-17).** Python 3.12.5 locally (3.11 on Streamlit Cloud, `.python-version`); TensorFlow 2.21.0, Keras 3.15.1, NumPy 1.26.4, pandas 3.0.2, scikit-learn 1.9.1, SciPy 1.17.1, yfinance 1.7.0. The pins for the dashboard are in `dashboard/requirements.txt`. Hardware and run time of one full walk-forward are not recorded: measure and add them (needed for the paper). Seeds 0-2 per block (`keras.utils.set_random_seed`); no GPU.
- **Reproduce each table.** Table III-IV: `daily/walk_forward.py` then `daily/regression_metrics.py`; Table V: `daily/build_daily_dataset.py` with the flags listed in the Handoff block, then `walk_forward.py` per folder; Table VI: `daily/walk_forward_horizon.py`; Table VII: `daily/interval_experiment.py`, `daily/lstm_quantile.py`, `daily/interval_vix.py`; Table VIII: `daily/volatility_experiment.py`; Table IX: `hourly/run_full_pipeline.sh`, `hourly/train_models.py`; Table X: `daily/horizon_always_up.py`. The silver data must first be extended to the test window's end (Handoff, "Refreshing the results").
- **Code availability.** Public repository https://github.com/aphiwat-data/lstm_dashboard; cite a tagged commit (create the tag `paper-v1` at submission). **Check: the repository has no LICENSE file**, so by default all rights are reserved; choose one (for example MIT or Apache-2.0) before the paper points to it. For a double-blind review use an anonymised mirror and say so.
- **Data availability.** §15.3. **Live log**: not public; the dashboard shows the last 15 scored forecasts.
- **AI-assistance disclosure.** IEEE requires AI-generated content in an article (text, figures, images, code) to be disclosed in the acknowledgements, naming the system and the sections concerned; JCSSE asks for full declaration at submission; IEEE Access rejects articles with fabricated content. Draft statement: *The authors used Claude (Anthropic) as a research assistant for code review, drafting and editing of text and checking of references. All numbers were produced by the authors' code and verified against the result files, all references were checked against Crossref, and the authors take full responsibility for the content.* Do not list an AI system as an author.
- **Ethics.** No human subjects or personal data; the work is not financial advice (keep the disclaimer); the student names and IDs on the dashboard are public by the owner's choice.

### 15.13 Working with Claude (rules)
1. **Read first.** The Handoff block, §15.1, this list and `AGENTS.md` before any drafting.
2. **Numbers come only from files.** Every number in a draft is quoted from a result file or from a table in §15.5, with the file and key; if it is not there, compute it with a script (not by hand) or say it is missing. Never round up, never mix windows.
3. **References only from §15.9.** No citation, DOI, page number or quotation from memory. A new reference is checked against Crossref (or the publisher's page) and added to §15.9 and the BibTeX first; unverified ones are marked UNVERIFIED and kept out of the text.
4. **Metric policy.** Lead with price error against the simple guess; direction accuracy is a labelled secondary check; no trading advice and no claim of profit.
5. **One results window.** Headline numbers are on the 449-day window; any other window (long window, other assets) lives in its own results folder and is labelled in text and tables.
6. **Claims discipline.** Use the supported and unsupported lists of §15.1. Say "no evidence of skill", not "no skill"; say "not significantly different", not "equal"; report effect sizes with intervals (E1) as soon as they exist.
7. **Provenance of external facts.** Venue rules, deadlines and fees come from §15.2 with their date; re-check before use.
8. **Privacy.** No AWS identifiers or credentials; author names only where the venue allows (double-blind review needs anonymisation, §15.15).
9. **Disclose AI assistance** as §15.12 says, at the venue's required place.
10. **End every session** by updating §15.15 (decisions and open questions) and §12; paper drafts, scripts and figures live in `paper/` (create it when writing starts); results and decisions are still recorded in this README only.

**Prompt templates.** (a) "Generate Table III in IEEE LaTeX (`booktabs`) from `daily/results/walk_forward_results.json` and `regression_metrics.json`; do not type numbers." (b) "Write the Related Work paragraph on forecast evaluation using only [1], [3]-[7] from §15.9; list each claim and its reference." (c) "Audit this paragraph: list every factual claim, the file or table that supports it, and flag the unsupported ones." (d) "Implement experiment E1 in `daily/` with tests, run it into `daily/results/paper_e1/` and add the result table to §15.5." (e) "Trim the abstract to 150 words without losing a number." (f) "Anonymise the manuscript for double-blind review and list what you removed."

**Definition of done for a section.** Every number traced; every reference in §15.9; limits and threats stated; no sentence that claims more than §15.1 allows; compiles with `IEEEtran`; within the page budget.

### 15.14 Notation and glossary
| Symbol or term | Meaning |
|---|---|
| c_t, r_t | close and log-return of day t |
| y_{t+1}, ŷ_{t+1} | next-day log-return and its forecast |
| simple guess, no-change | the forecast that tomorrow's close equals today's (return forecast 0) |
| e_t | return error y_t - ŷ_t |
| σ_t, λ | EWMA volatility of past errors, decay 0.94 |
| MAE, RMSE, MAPE | mean absolute error, root mean squared error, mean absolute percentage error |
| OS R² | out-of-sample R² against a benchmark (zero or the rolling mean) |
| IC | Spearman rank correlation of forecast and realised return |
| DM | Diebold-Mariano test of equal predictive accuracy |
| AUC | area under the ROC curve of the direction head |
| Kupiec p | p-value of the proportion-of-failures test for interval coverage |
| interval score | width + (2/α) × distance outside the interval (α = 0.10) |
| QLIKE | ln σ² + r² / σ², a robust loss for variance forecasts |
| walk-forward | repeated retraining on the past and testing on the days that follow |
| n_eff | effective sample size n / h when labels overlap over h days |

### 15.15 Decisions and open questions (update every session)
1. **Venue and timeline.** JCSSE 2027 second round (22 Jan 2027) then IEEE Access? Who registers and presents (at least one author must; budget THB 12,600-14,000 for an IEEE member or student)? *Open.*
2. **Angle.** A with B (§15.8)? Which experiments are in scope before 22 Jan (suggested: E1, E8, E9, E10, E11, E12 only)? *Open.*
3. **Authors and affiliations.** Order, the advisor's role, the committee's role, affiliations, ORCID iDs (required by IEEE Access). *Open.*
4. **Anonymisation.** JCSSE is double-blind; the public repository and the dashboard identify the authors. Use an anonymised mirror for review and the public link after acceptance? *Open.*
5. **Licence.** Which licence for the code (§15.12)? *Open.*
6. **Hyperparameter history.** Were window 20, 32 units, dropout 0.2 and patience 3 chosen before looking at the test window, or tuned on it? The paper must say. *Open: owner to confirm.*
7. **Live record.** Harden the log (E10) before citing it, or describe it only as a demonstration? *Open.*
8. **Compute budget** for E2, E4-E6 (CPU only so far). *Open.*
9. **Advisor's expectation** of the venue tier and the deadline. *Open.*

### 15.16 Pre-submission checklist
- [ ] Every number traced to a file (rule 2); tables generated by script; figures regenerated from the result files.
- [ ] Every reference in §15.9 with its DOI; BibTeX compiles; no unverified entry in the text.
- [ ] IEEE template for the venue; page limit and abstract length met; Index Terms; units and abbreviations defined.
- [ ] Claims within §15.1; limitations and threats stated (§15.7); no trading advice.
- [ ] Double-blind: names, affiliations, acknowledgements and identifying links removed; similarity check below the venue's limit (25% for IEEE Xplore at JCSSE).
- [ ] AI-use disclosure in the acknowledgements and at submission; data and code availability statements; licence chosen.
- [ ] A clean clone reproduces the tables (commands of §15.12); environment and run time recorded; commit tagged.
- [ ] Registration, copyright form and presentation arranged (one author must present).
