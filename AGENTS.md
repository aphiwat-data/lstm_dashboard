# AGENTS.md

Project: next-day XAU/USD (gold) forecasting with an LSTM, plus a Streamlit dashboard (Mae Fah Luang University, Computer
Engineering senior project). **Read the "Handoff: where things stand" block at the top of `README.md` first**: it has the current
results, run commands, the refresh procedure and the open items. The rest of the README is the detailed record.

The owner writes in Thai. Reply in Thai; keep code, README text and commit messages in English.

## Rules that matter

- **The repo is public.** Never commit AWS keys, the AWS account ID or any credentials. The deployed app gets AWS access from
  Streamlit Cloud Secrets. The student names and IDs in the dashboard sidebar are intentional.
- **Metric policy (advisor feedback).** Judge the price forecast by price error (MAE, RMSE, % error, R², IC) against the
  "tomorrow = today" baseline. Directional accuracy is a classification metric: keep it only as a labelled secondary check.
- **One results window.** Every daily number (README, dashboard, thesis tables) is on the same walk-forward test window. A refresh
  re-runs the main run and all four feature-search variants on the same rows, then updates README, dashboard and thesis tables in
  the same change. Keep the previous outputs in a dated `daily/results/snapshot_*` folder, and delete stale
  `daily/data/*_cache.parquet` files first (they are reused without checking their end date).
- **The project is open to new approaches.** The old proposal and its targets (including the ~70% direction accuracy) are superseded; the advisor
  allows trying anything. The earlier feature searches (macro, NFP, FOMC+CPI, COT; daily and hourly) were null: treat that as evidence, not a
  prohibition. Judge every new idea with the regression-first metrics and the causal, same-window protocol, and report nulls honestly.
- **Dashboard (`dashboard/app.py`).** Native Streamlit elements only (no `unsafe_allow_html`), `width="stretch"`, escape dollar signs
  in markdown as `\$`, and use `st.container(horizontal=True)` instead of `st.columns` where the open sidebar would squeeze content.
  Streamlit Cloud installs a newer Streamlit than local development used, so check layout changes at 1280, 1024, 900, 760 and 390 px
  and on the live URL after pushing.
- **`main` is the deployed branch.** Every push redeploys the public app within about a minute. Make small commits and run
  `python3 -m py_compile dashboard/app.py` before pushing.
- **Document results in `README.md` only.** The owner will rewrite the paper from scratch, so do not edit Chapter 3 or other Word files; keep the
  README (Journey / Phase notes, scoreboard, Handoff) current after every piece of work.

## Commands

See "How to run things" in the README Handoff block (dashboard, causality tests, daily build, walk-forward, horizon table, and the
flags behind each feature-search folder).
