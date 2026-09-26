"""
Next-Day Gold Price Forecasting with LSTM — Streamlit Dashboard
Reads pipeline outputs from S3 (Gold layer, trained model, scalers, walk-forward
results) and renders the Model Evaluation + Next-Day Forecast view described in
Section 3.8 of the project proposal.

Run locally:
    streamlit run app.py
"""
import tempfile
import pickle
import json
from contextlib import nullcontext
from pathlib import Path

import boto3
import streamlit as st
import pandas as pd
import numpy as np
import awswrangler as wr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Config — must match the S3_BUCKET used in notebooks 01-06
# ---------------------------------------------------------------------------
S3_BUCKET  = "gold-lstm-forecast"
DATA_PATH  = f"s3://{S3_BUCKET}/gold/xauusd_daily/features"
GOLD_PATH  = f"{DATA_PATH}/xauusd_features.parquet"

FEATURES   = ["close", "return", "ma7", "ma14", "ma30", "ma60", "volatility_7", "momentum_7"]
TARGET     = "target"
SEQ_LEN    = 60
SPLIT_DATE = "2024-12-31"
RETRAIN_EVERY_DAYS = 7  # matches notebook 05's walk-forward RETRAIN_EVERY (simulated trading days)
FORECAST_KEY = "gold/xauusd_daily/predictions/latest_forecast.json"  # written by automation/lambda_function.py

FEATURE_LABELS = {
    "return": "Daily Return",
    "ma7": "7-Day Moving Average",
    "ma14": "14-Day Moving Average",
    "ma30": "30-Day Moving Average",
    "ma60": "60-Day Moving Average",
    "volatility_7": "7-Day Rolling Volatility",
    "momentum_7": "7-Day Momentum",
}

# Categorical palette — validated with the project's palette validator
# (CVD-safe adjacent pairs, normal-vision floor, contrast; see
# docs/TROUBLESHOOTING.md). One fixed hue per entity, never cycled.
COLORS = {
    "actual": "#EB6834",              # orange — the real price series
    "LSTM Walk-Forward": "#2A78D6",   # blue   — the proposed model (kept consistent across charts)
    "Naive Persistence": "#1BAF7A",   # aqua
    "AR(5) Baseline": "#EDA100",      # yellow
    "Linear Regression": "#E87BA4",   # magenta
    "return_dist": "#008300",         # green  — reused from the Feature Explorer's spare palette slots,
    "volume": "#4A3AA7",              # purple — now given a fixed identity since each gets its own chart
}
# Diverging pair for the prediction-error panel (blue <-> red, neutral gray
# midpoint) — warm red draws the eye to under-prediction, which is this
# project's key documented limitation during the 2025-2026 price surge.
ERROR_OVER  = "#2A78D6"   # model predicted too HIGH (blue, cool)
ERROR_UNDER = "#E34948"   # model predicted too LOW  (red, warm — the surge story)

st.set_page_config(
    page_title="Gold Price LSTM Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading — cached so S3/model load only happens once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and model from S3...")
def load_all():
    df = wr.s3.read_parquet(path=GOLD_PATH)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    tmpdir = tempfile.mkdtemp()
    for fname in [
        "feature_scaler.pkl", "target_scaler.pkl",
        "wf_pred.npy", "wf_actual.npy", "test_dates.csv",
        "lstm_model.keras",
    ]:
        wr.s3.download(path=f"{DATA_PATH}/{fname}", local_file=f"{tmpdir}/{fname}")

    with open(f"{tmpdir}/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(f"{tmpdir}/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    wf_pred    = np.load(f"{tmpdir}/wf_pred.npy")
    wf_actual  = np.load(f"{tmpdir}/wf_actual.npy")
    test_dates = pd.read_csv(f"{tmpdir}/test_dates.csv")["date"].values
    model      = load_model(f"{tmpdir}/lstm_model.keras")

    return df, feature_scaler, target_scaler, wf_pred, wf_actual, test_dates, model


@st.cache_data(ttl=3600, show_spinner=False)
def get_model_last_retrained():
    """S3 LastModified of the trained model object — a proxy for when notebook 05
    last actually retrained (vs. just data being refreshed by 01-04)."""
    meta = wr.s3.describe_objects(path=f"{DATA_PATH}/lstm_model.keras")
    last_modified = list(meta.values())[0]["LastModified"]
    return pd.Timestamp(last_modified).tz_localize(None)


@st.cache_data(ttl=300, show_spinner=False)
def get_lambda_forecast():
    """Reads the daily inference-only forecast written by the Lambda automation
    (automation/lambda_function.py), if it has ever run. Returns None rather
    than raising when the object doesn't exist yet — this feature is additive
    and the dashboard's own on-the-fly forecast above still works without it."""
    try:
        obj = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=FORECAST_KEY)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


V2_DIR = Path(__file__).resolve().parents[1] / "daily" / "results"


@st.cache_data(show_spinner=False)
def load_v2():
    """Corrected daily evaluation (daily/walk_forward.py): walk-forward LSTM on log-returns, judged against the
    always-up rate. Returns (results, predictions) or (None, None) when the files are not in the repository."""
    try:
        res = json.loads((V2_DIR / "walk_forward_results.json").read_text())
        preds = pd.read_csv(V2_DIR / "wf_predictions.csv", parse_dates=["date"])
        return res, preds
    except Exception:
        return None, None


def compute_baselines(df, feature_scaler, target_scaler):
    """Recomputes Naive / AR(5) / Linear Regression exactly as notebook 06 does,
    so the dashboard never drifts from the report's own evaluation logic."""
    train_df = df[df["date"] <= SPLIT_DATE].copy()
    test_df  = df[df["date"] >  SPLIT_DATE].copy()

    naive_pred   = test_df["close"].values
    naive_actual = test_df["target"].values

    X_tr_lr = feature_scaler.transform(train_df[FEATURES])
    X_te_lr = feature_scaler.transform(test_df[FEATURES])
    y_tr_lr = target_scaler.transform(train_df[[TARGET]]).ravel()
    y_te_lr = target_scaler.transform(test_df[[TARGET]]).ravel()
    lr = LinearRegression().fit(X_tr_lr, y_tr_lr)
    lr_pred   = target_scaler.inverse_transform(lr.predict(X_te_lr).reshape(-1, 1)).ravel()
    lr_actual = target_scaler.inverse_transform(y_te_lr.reshape(-1, 1)).ravel()

    close_all  = df["close"].values
    target_all = df[TARGET].values
    N = 5
    Xar = np.column_stack([close_all[i:len(close_all) - N + i + 1] for i in range(N)])
    yar = target_all[N - 1:]
    sp  = len(train_df) - N
    ar  = LinearRegression().fit(Xar[:sp], yar[:sp])
    ar_pred   = ar.predict(Xar[sp + 1:])
    ar_actual = yar[sp + 1:]

    return {
        "Naive Persistence": (naive_actual, naive_pred),
        "AR(5) Baseline":    (ar_actual, ar_pred),
        "Linear Regression": (lr_actual, lr_pred),
    }


def get_metrics(actual, pred):
    mask = ~np.isnan(pred)
    a, p = np.array(actual)[mask], np.array(pred)[mask]
    return {"MAE": mean_absolute_error(a, p), "RMSE": float(np.sqrt(mean_squared_error(a, p)))}


def directional_accuracy(actual, pred):
    a, p = np.array(actual), np.array(pred)
    return float(np.mean((np.diff(a) > 0) == (np.diff(p) > 0)) * 100)


def forecast_next_day(df, feature_scaler, target_scaler, model):
    latest = df.sort_values("date").tail(SEQ_LEN)
    X = feature_scaler.transform(latest[FEATURES])
    X = X.reshape(1, SEQ_LEN, len(FEATURES))
    pred_scaled = model.predict(X, verbose=0)[0][0]
    return float(target_scaler.inverse_transform([[pred_scaled]])[0][0])


def forecast_interval(point_forecast, wf_actual, wf_pred, lower_pct=5, upper_pct=95):
    """Empirical interval around a point forecast, built from the walk-forward
    error distribution (actual - predicted) rather than assuming errors are
    normally distributed — appropriate given the documented under-prediction
    skew during the 2025-2026 surge (see Model Evaluation / Methodology)."""
    residuals = np.array(wf_actual) - np.array(wf_pred)
    lower = point_forecast + np.percentile(residuals, lower_pct)
    upper = point_forecast + np.percentile(residuals, upper_pct)
    return lower, upper


# ---------------------------------------------------------------------------
# Load + compute
# ---------------------------------------------------------------------------
df, feature_scaler, target_scaler, wf_pred, wf_actual, test_dates, model = load_all()
baselines = compute_baselines(df, feature_scaler, target_scaler)

min_len = min(len(wf_actual), *[len(a) for a, _ in baselines.values()])
results, dir_acc = {}, {}
for name, (a, p) in baselines.items():
    a_trim, p_trim = a[-min_len:], p[-min_len:]
    results[name]  = get_metrics(a_trim, p_trim)
    dir_acc[name]  = directional_accuracy(a_trim, p_trim)
results["LSTM Walk-Forward"] = get_metrics(wf_actual, wf_pred)
dir_acc["LSTM Walk-Forward"] = directional_accuracy(wf_actual, wf_pred)
model_names = ["Naive Persistence", "AR(5) Baseline", "Linear Regression", "LSTM Walk-Forward"]

latest_row    = df.sort_values("date").iloc[-1]
next_day_pred = forecast_next_day(df, feature_scaler, target_scaler, model)
delta         = next_day_pred - latest_row["close"]
forecast_lo, forecast_hi = forecast_interval(next_day_pred, wf_actual, wf_pred)

dates_wf     = pd.to_datetime(test_dates[-len(wf_pred):])
pred_error   = wf_actual - wf_pred  # positive = model under-predicted, negative = over-predicted

model_last_retrained = get_model_last_retrained()
days_since_retrain    = (pd.Timestamp.now(tz="UTC").tz_localize(None) - model_last_retrained).days
lambda_forecast       = get_lambda_forecast()
v2, v2_preds          = load_v2()

# 30-day rolling volatility, computed on the fly for the dashboard view (the
# model itself trains on the 7-day volatility_7 feature — see Feature Explorer)
df["volatility_30"] = df["return"].rolling(30).std()

# ---------------------------------------------------------------------------
# Sidebar — project facts, always visible, doesn't require a tab click
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About This Project")
    st.caption(
        "Next-Day Gold Price Forecasting with LSTM — undergraduate Computer "
        "Engineering senior project, Mae Fah Luang University."
    )
    st.markdown("**Dataset**")
    st.text(f"Ticker: GC=F (COMEX Gold Futures)\nRange: {df['date'].min().date()} – {df['date'].max().date()}\nRows: {len(df):,}")
    st.markdown("**Train / Test Split**")
    if v2 is not None:
        st.text(f"Chronological split: {SPLIT_DATE}\nScalers re-fit inside each\ntraining window (no look-ahead)")
    else:
        st.text(f"Chronological split: {SPLIT_DATE}\nScalers fit on train only\n(leakage prevention)")
    st.markdown("**Model**")
    if v2 is not None:
        pr = v2["protocol"]
        st.text(
            f"LSTM({pr['lstm_units']}), 2 heads: return + direction\nInput window: {pr['window']} days\n"
            f"Walk-forward retrain: every {pr['retrain_every_days']} days\n{pr['train_len_days']}-day rolling window\n"
            f"Target: next-day log-return\nScalers re-fit in every window"
        )
        st.caption(f"Original pipeline (price-level target): 2-layer LSTM (64, 32), window {SEQ_LEN}.")
    else:
        st.text(f"2-layer LSTM (64, 32 units)\nInput window: {SEQ_LEN} days\nWalk-forward retrain: every {RETRAIN_EVERY_DAYS} days\n365-day rolling training window")

    st.markdown("**Model Freshness**")
    if v2 is not None:
        st.text(f"Data through: {v2['latest_forecast']['as_of_date']}\nForecast generated: {v2['latest_forecast']['generated_at'][:10]}")
        st.caption("Regenerate with daily/walk_forward.py (see daily/results).")
    else:
        st.text(
            f"Last retrained: {model_last_retrained.date()}\n"
            f"{days_since_retrain} day(s) since last retrain"
        )
        if days_since_retrain > RETRAIN_EVERY_DAYS:
            st.warning(
                f"Overdue for retrain by {days_since_retrain - RETRAIN_EVERY_DAYS} day(s) "
                f"(cadence: every {RETRAIN_EVERY_DAYS} days). This is about the model "
                f"itself going stale — separate from whether the underlying price data "
                f"(above) is up to date.",
                icon="⚠️",
            )
        else:
            st.caption(f"Within the {RETRAIN_EVERY_DAYS}-day retrain cadence.")
    st.divider()
    if v2 is not None:
        st.caption(
            "Corrected evaluation comes from daily/walk_forward.py (results committed in daily/results). "
            "The original pipeline's charts (recomputed baselines + S3 walk-forward arrays) remain under "
            "Model Evaluation > Original pipeline."
        )
    else:
        st.caption(
            "Baseline models (Naive, AR(5), Linear Regression) are recomputed "
            "live in this app from the Gold-layer data and saved scalers. LSTM "
            "walk-forward results are loaded directly from S3."
        )

# ---------------------------------------------------------------------------
# Header + KPI row (always visible, above the tabs)
# ---------------------------------------------------------------------------
st.title("Next-Day Gold Price Forecasting")
if v2 is not None:
    lf, m2 = v2["latest_forecast"], v2["models"]
    up2, l2 = m2["drift"]["direction_from_sign"]["always_up_acc"], m2["lstm"]
    acc2 = l2["direction_from_head"]["acc"]
    st.caption(f"LSTM time-series model (walk-forward, return target) · data through {lf['as_of_date']} · forecast generated {lf['generated_at'][:10]}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"${lf['latest_close']:,.2f}")
    c2.metric("Next-Day Forecast", f"${lf['predicted_next_close']:,.2f}", f"{lf['predicted_next_close'] - lf['latest_close']:+,.2f}")
    c2.caption(f"90% range: \\${lf['interval_90_close'][0]:,.2f} – \\${lf['interval_90_close'][1]:,.2f}")
    c3.metric("LSTM Direction Accuracy (test)", f"{acc2:.1%}", f"{(acc2 - up2) * 100:+.1f} pt vs always-up ({up2:.1%})")
    c4.metric("LSTM MAE (test)", f"${l2['mae_usd']:,.2f}", f"{l2['mae_usd'] - l2['naive_mae_usd']:+.2f} vs no-change", delta_color="inverse")

    st.info(
        "On daily data the LSTM cannot be told apart from 'always up' or from a no-change forecast (see Model "
        "Evaluation), so read the forecast as roughly today's close within the range shown, not as a trading "
        "signal. The much larger errors of the original price-level pipeline came from a scale problem and are "
        "kept for the record under Model Evaluation > Original pipeline."
    )
else:
    st.caption(f"LSTM time-series model · pipeline data current as of {latest_row['date'].date()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"${latest_row['close']:,.2f}")
    c2.metric("Next-Day Forecast", f"${next_day_pred:,.2f}", f"{delta:+,.2f}")
    c2.caption(f"90% range: \\${forecast_lo:,.2f} – \\${forecast_hi:,.2f}")
    c3.metric("LSTM Directional Accuracy", f"{dir_acc['LSTM Walk-Forward']:.1f}%")
    c4.metric("LSTM MAE (test period)", f"${results['LSTM Walk-Forward']['MAE']:,.2f}")

    st.info(
        "The model tends to under-predict during sharp upward price moves (see the "
        "Prediction Error panel in the Model Evaluation tab). Treat the forecast "
        "above as a directional signal rather than a precise price target."
    )

with st.expander("Daily automation forecast (Lambda inference-refresh)", expanded=False):
    st.caption("Note: this automation runs the ORIGINAL price-level model; it has not been updated to the corrected return-target model.")
    st.caption(
        "The KPI above is computed fresh on every page load from the canonical "
        "Gold-layer data and model in S3. Separately, a scheduled Lambda "
        "(`automation/lambda_function.py`) runs once daily to keep a forecast "
        "ready between full pipeline re-runs — shown here for comparison."
    )
    if lambda_forecast is None:
        st.caption(
            "No automation output found yet at "
            f"`s3://{S3_BUCKET}/{FORECAST_KEY}` — either the Lambda hasn't been "
            "deployed/run yet, or it hasn't found a new trading day to process. "
            "This is expected until it's set up; the KPI above is unaffected."
        )
    else:
        generated_at = pd.Timestamp(lambda_forecast["generated_at"]).tz_localize(None)
        age_hours = (pd.Timestamp.now(tz="UTC").tz_localize(None) - generated_at).total_seconds() / 3600
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Automation forecast (as of " + lambda_forecast["as_of_date"] + ")",
                   f"${lambda_forecast['predicted_next_close']:,.2f}",
                   f"{lambda_forecast['delta']:+,.2f}")
        lc2.metric("Agrees with on-the-fly forecast?",
                   "Yes" if abs(lambda_forecast["predicted_next_close"] - next_day_pred) < 1.0 else "Differs")
        lc3.metric("Generated", f"{age_hours:.1f}h ago")
        if age_hours > 48:
            st.warning(
                "This automation output is over 48 hours old — the scheduled "
                "Lambda may not be running (check EventBridge Scheduler / Lambda "
                "logs in AWS). Not necessarily an error on a weekend/market "
                "holiday, when no new close posts.",
                icon="⚠️",
            )

def render_v2_eval(v2: dict, p: pd.DataFrame) -> None:
    proto, models = v2["protocol"], v2["models"]
    up = models["drift"]["direction_from_sign"]["always_up_acc"]
    st.subheader("Corrected evaluation: walk-forward LSTM on returns")
    st.caption(
        f"Same walk-forward scheme as the original notebooks (retrain every {proto['retrain_every_days']} trading days on the latest "
        f"{proto['train_len_days']} days), but predicting the next-day log-return from scale-free features, with scalers re-fit inside every "
        f"window. Test period {proto['test_start']} to {proto['test_end']} ({proto['n_test']} days); every model is scored on the same days."
    )

    pred_close = p["meta_close"] * np.exp(p["pred_ret_lstm"])
    err = p["meta_next_close"] - pred_close
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06,
        subplot_titles=("Next-day close: actual vs. LSTM forecast", "Forecast error (actual - forecast, USD)"),
    )
    fig.add_trace(go.Scatter(x=p["date"], y=p["meta_next_close"], name="Actual next close", line=dict(color=COLORS["actual"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=p["date"], y=pred_close, name="LSTM forecast", line=dict(color=COLORS["LSTM Walk-Forward"], width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=p["date"], y=err, marker_color=[ERROR_UNDER if e >= 0 else ERROR_OVER for e in err], showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="#8a8a86", line_width=1, row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified", margin=dict(t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Error (USD)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The x-axis is the day the forecast is made; both lines show the NEXT trading day's close. The forecast tracks the actual series "
        "almost exactly because it is close to a no-change forecast: the errors are day-to-day market moves, not a systematic lag."
    )

    st.divider()
    st.subheader("Direction accuracy vs. the 'always up' rule")
    st.caption(
        f"Gold rose on {up:.1%} of test days, so a model that always says 'up' already scores {up:.1%}. That, not 50%, is the bar to beat. "
        "Bars show accuracy with a 95% confidence interval."
    )
    entries = [("Always up", models["drift"]["direction_from_sign"]), ("Momentum rule", models["momentum_rule"]["direction_from_head"]),
               ("Ridge", models["ridge"]["direction_from_sign"]), ("HistGB classifier", models["hist_gb_clf"]["direction_from_head"]),
               ("LSTM", models["lstm"]["direction_from_head"])]
    colors = ["#8a8a86", COLORS["Naive Persistence"], COLORS["Linear Regression"], COLORS["AR(5) Baseline"], COLORS["LSTM Walk-Forward"]]
    acc = [d["acc"] * 100 for _, d in entries]
    fig_d = go.Figure(go.Bar(
        x=[n for n, _ in entries], y=acc, marker_color=colors, text=[f"{a:.1f}%" for a in acc], textposition="inside", insidetextanchor="start",
        textfont=dict(size=13, color="white"),
        error_y=dict(type="data", symmetric=False, array=[(d["ci95"][1] - d["acc"]) * 100 for _, d in entries], arrayminus=[(d["acc"] - d["ci95"][0]) * 100 for _, d in entries]),
    ))
    fig_d.add_hline(y=up * 100, line_dash="dash", line_color="#8a8a86")
    fig_d.update_layout(height=380, margin=dict(t=40, b=10), showlegend=False, yaxis=dict(range=[35, 72], title="Direction accuracy (%)"),
                        title=dict(text=f"Dashed line = always up ({up:.1%})", font=dict(size=13)))
    st.plotly_chart(fig_d, use_container_width=True)

    st.divider()
    st.subheader("Model comparison (identical test days)")
    labels = {"drift": "Rolling drift", "ridge": "Ridge", "hist_gb_reg": "HistGB regressor", "hist_gb_clf": "HistGB classifier", "lstm": "LSTM", "momentum_rule": "Momentum rule"}
    rows = {}
    for key, r in models.items():
        d = r.get("direction_from_head") or r["direction_from_sign"]
        rows[labels.get(key, key)] = {
            "MAE ($)": r.get("mae_usd"), "MAE skill vs no-change": r.get("mae_skill_vs_naive"), "DM p (vs no-change)": r.get("dm_p_vs_naive"),
            "Direction acc. (%)": d["acc"] * 100, "vs always-up (pt)": (d["acc"] - d["always_up_acc"]) * 100, "p (vs always-up)": d["p_vs_always_up"],
            "AUC": (r.get("direction_from_head") or {}).get("auc"), "IC (rank corr.)": r.get("ic_spearman"),
        }
    table = pd.DataFrame(rows).T.astype(float)  # None -> NaN so missing cells render as a dash
    fmt = {"MAE ($)": "{:.2f}", "MAE skill vs no-change": "{:+.4f}", "DM p (vs no-change)": "{:.3f}", "Direction acc. (%)": "{:.1f}",
           "vs always-up (pt)": "{:+.1f}", "p (vs always-up)": "{:.3f}", "AUC": "{:.3f}", "IC (rank corr.)": "{:+.3f}"}
    shown = table.apply(lambda col: col.map(lambda v: "–" if pd.isna(v) else fmt[col.name].format(v)))  # display copy; `table` keeps the numbers
    st.dataframe(shown, use_container_width=True)
    lstm = models["lstm"]
    hd = lstm["direction_from_head"]
    st.markdown(
        f"**Interpretation.** The LSTM's price error (MAE \\${lstm['mae_usd']:,.2f}) is essentially the no-change level (\\${lstm['naive_mae_usd']:,.2f}; skill "
        f"{lstm['mae_skill_vs_naive']:+.2%}, Diebold-Mariano p = {lstm['dm_p_vs_naive']:.2f}). Its direction accuracy ({hd['acc']:.1%}, 95% CI "
        f"{hd['ci95'][0]:.1%} to {hd['ci95'][1]:.1%}) is not distinguishable from always-up ({up:.1%}, p = {hd['p_vs_always_up']:.2f}). "
        f"With only {proto['n_test']} test days the interval is about +/-{(hd['ci95'][1] - hd['ci95'][0]) / 2 * 100:.0f} points, so small edges cannot be detected either way. "
        "The much larger errors reported by the original pipeline came from a price-level scale problem, not from the market."
    )
    d1, d2 = st.columns(2)
    d1.download_button("Download predictions (CSV)", data=p.to_csv(index=False).encode("utf-8"), file_name="daily_walk_forward_predictions.csv", mime="text/csv")
    d2.download_button("Download metrics table (CSV)", data=table.to_csv().encode("utf-8"), file_name="daily_walk_forward_metrics.csv", mime="text/csv")


tab_overview, tab_eval, tab_vol, tab_method = st.tabs([
    "Overview", "Model Evaluation", "Volatility & Features", "Methodology",
])

# ---------------------------------------------------------------------------
# TAB 1 — Overview: full historical price chart (+ volume panel, when present)
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Historical Gold Price (GC=F)")
    st.caption(
        "Full collected history. Use the range selector or drag the slider "
        "below the chart to zoom into a specific period."
    )

    # Volume gets its own panel here (the conventional price+volume combo
    # chart) purely for context/reference. Whether it's a genuine model
    # FEATURE is a separate question, answered in Methodology — this panel
    # doesn't repeat that diagnostic, it just shows what the raw series
    # looks like, same as the price panel above it.
    has_volume = "volume" in df.columns
    fig_hist = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    ) if has_volume else go.Figure()
    subplot_kw = {"row": 1, "col": 1} if has_volume else {}

    fig_hist.add_trace(go.Scatter(
        x=df["date"], y=df["close"], name="Close",
        line=dict(color=COLORS["actual"], width=1.5),
    ), **subplot_kw)
    fig_hist.add_vline(
        x=pd.Timestamp(SPLIT_DATE), line_dash="dash", line_color="#8a8a86",
        annotation_text="Train / Test Split", annotation_position="top left",
        **subplot_kw,
    )

    range_axis = dict(
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
    )

    if has_volume:
        fig_hist.add_trace(go.Bar(
            x=df["date"], y=pd.to_numeric(df["volume"], errors="coerce"),
            name="Volume", marker_color=COLORS["volume"], marker_line_width=0,
            showlegend=False,
        ), row=2, col=1)
        fig_hist.update_layout(
            height=560, hovermode="x unified", margin=dict(t=10, b=10),
            xaxis2=range_axis,
        )
        fig_hist.update_yaxes(title_text="Close Price (USD)", row=1, col=1)
        fig_hist.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig_hist.update_layout(
            height=460, hovermode="x unified", margin=dict(t=10, b=10),
            yaxis_title="Close Price (USD)", xaxis=range_axis,
        )

    st.plotly_chart(fig_hist, use_container_width=True)

    overview_caption = (
        "The price series shows a sustained long-term uptrend with a sharp, "
        "historically unprecedented acceleration during the 2025–2026 test "
        "period — the central challenge discussed throughout Model Evaluation."
    )
    if has_volume:
        overview_caption += (
            " Volume is shown for reference only — see Methodology for why "
            "it isn't part of the model's feature set (the long, mostly-empty "
            "stretches are a known gap in Yahoo Finance's continuous-futures "
            "volume reporting for GC=F, not a bug in this pipeline)."
        )
    st.caption(overview_caption)

# ---------------------------------------------------------------------------
# TAB 2 — Model Evaluation: actual vs predicted, error panel, comparison
# ---------------------------------------------------------------------------
with tab_eval:
    if v2 is not None:
        render_v2_eval(v2, v2_preds)
    _orig = (
        st.expander("Original pipeline (price-level target, notebooks 03-06): kept for the record", expanded=False)
        if v2 is not None else nullcontext()
    )
    with _orig:
        if v2 is not None:
            st.caption(
                "These charts come from the original notebooks: the model predicted the next-day close PRICE from price-level inputs scaled "
                "with training-period statistics. In 2025-26 gold traded far above that range, so inputs and targets left what the scalers and "
                "model had seen and the LSTM's MAE reached about \\$639 against about \\$51 for trivial baselines. The interpretation text at the "
                "bottom of this section describes that pipeline and no longer reflects the current conclusion. Directional accuracy here "
                "compares day-to-day changes of the predictions with those of the actual prices, which is not the standard definition."
            )
        st.subheader("Actual vs. Predicted (LSTM Walk-Forward)")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
            vertical_spacing=0.06,
            subplot_titles=("Price", "Prediction Error (Actual − Predicted)"),
        )
        fig.add_trace(go.Scatter(
            x=dates_wf, y=wf_actual, name="Actual",
            line=dict(color=COLORS["actual"], width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates_wf, y=wf_pred, name="LSTM Predicted",
            line=dict(color=COLORS["LSTM Walk-Forward"], width=2, dash="dash"),
        ), row=1, col=1)

        error_colors = [ERROR_UNDER if e >= 0 else ERROR_OVER for e in pred_error]
        fig.add_trace(go.Bar(
            x=dates_wf, y=pred_error, name="Error", marker_color=error_colors,
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=0, line_color="#8a8a86", line_width=1, row=2, col=1)

        fig.update_layout(
            height=560, hovermode="x unified", margin=dict(t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            xaxis2=dict(rangeslider=dict(visible=True)),
        )
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Error (USD)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Red bars ({ERROR_UNDER}) mark days the model under-predicted (actual > "
            f"predicted); blue bars mark over-prediction. The red bars dominate "
            f"during the 2025–2026 rally — visual confirmation that the LSTM "
            f"systematically lags the surge rather than erring randomly."
        )

        st.divider()

        st.subheader("Metrics for a Selected Period")
        st.caption(
            "Narrow the LSTM's own metrics to a sub-window of the test period — "
            "e.g. isolate the 2025–2026 surge to see how much it drives the "
            "headline MAE/RMSE above, versus quieter stretches."
        )
        range_min, range_max = dates_wf.min().date(), dates_wf.max().date()
        range_pick = st.date_input(
            "Date range", value=(range_min, range_max),
            min_value=range_min, max_value=range_max,
        )
        if isinstance(range_pick, tuple) and len(range_pick) == 2:
            r_start, r_end = pd.Timestamp(range_pick[0]), pd.Timestamp(range_pick[1])
            range_mask = np.asarray((dates_wf >= r_start) & (dates_wf <= r_end))
            if range_mask.sum() >= 2:
                r_actual, r_pred = wf_actual[range_mask], wf_pred[range_mask]
                r_metrics = get_metrics(r_actual, r_pred)
                r_dir_acc = directional_accuracy(r_actual, r_pred)
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Days in range", f"{int(range_mask.sum()):,}")
                rc2.metric("MAE", f"${r_metrics['MAE']:,.2f}")
                rc3.metric("RMSE", f"${r_metrics['RMSE']:,.2f}")
                rc4.metric("Directional Accuracy", f"{r_dir_acc:.1f}%")
            else:
                st.caption("Select a range with at least 2 days to compute metrics.")

        st.divider()

        st.subheader("Model Comparison")
        st.caption(
            "All four models are evaluated on the identical chronological test "
            "window. MAE/RMSE measure error magnitude in USD; Directional "
            "Accuracy measures whether the model correctly calls up-vs-down "
            "moves — the practically relevant question for a trading signal."
        )
        mcol1, mcol2, mcol3 = st.columns(3)

        for col, metric_key, title, fmt in [
            (mcol1, "MAE", "MAE (USD) — lower is better", "${:,.1f}"),
            (mcol2, "RMSE", "RMSE (USD) — lower is better", "${:,.1f}"),
        ]:
            fig_b = go.Figure(go.Bar(
                x=model_names,
                y=[results[m][metric_key] for m in model_names],
                marker_color=[COLORS[m] for m in model_names],
                text=[fmt.format(results[m][metric_key]) for m in model_names],
                textposition="outside",
            ))
            fig_b.update_layout(title=title, height=360, margin=dict(t=40, b=10), showlegend=False)
            col.plotly_chart(fig_b, use_container_width=True)

        fig_d = go.Figure(go.Bar(
            x=model_names,
            y=[dir_acc[m] for m in model_names],
            marker_color=[COLORS[m] for m in model_names],
            text=[f"{dir_acc[m]:.1f}%" for m in model_names],
            textposition="outside",
        ))
        fig_d.add_hline(y=50, line_dash="dash", line_color="#8a8a86",
                         annotation_text="50% = random guess", annotation_position="top left")
        fig_d.update_layout(title="Directional Accuracy (%) — higher is better",
                             height=360, margin=dict(t=40, b=10), showlegend=False)
        mcol3.plotly_chart(fig_d, use_container_width=True)

        metrics_table = pd.DataFrame({
            "MAE ($)": {m: round(results[m]["MAE"], 2) for m in model_names},
            "RMSE ($)": {m: round(results[m]["RMSE"], 2) for m in model_names},
            "Directional Accuracy (%)": {m: round(dir_acc[m], 2) for m in model_names},
        })
        st.dataframe(metrics_table, use_container_width=True)

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Download metrics table (CSV)",
            data=metrics_table.to_csv().encode("utf-8"),
            file_name="model_comparison_metrics.csv",
            mime="text/csv",
        )
        wf_export = pd.DataFrame({
            "date": dates_wf, "actual": wf_actual, "lstm_predicted": wf_pred,
            "error": pred_error,
        })
        dl2.download_button(
            "Download actual vs. predicted (CSV)",
            data=wf_export.to_csv(index=False).encode("utf-8"),
            file_name="lstm_walk_forward_actual_vs_predicted.csv",
            mime="text/csv",
        )

        st.markdown(
            "**Interpretation.** The three baselines post lower MAE/RMSE than the "
            "LSTM, but none exceed 45% directional accuracy — worse than chance "
            "at calling the next day's direction. Their low error is a byproduct "
            "of predicting little change on a series that moves slowly most "
            "days, not genuine skill. The LSTM is the only model to clear 50% "
            "directional accuracy; its higher absolute error reflects a "
            "documented, expected limitation (smoother predictions that lag an "
            "unprecedented price surge — see the error panel above), not a "
            "failure to learn."
        )

# ---------------------------------------------------------------------------
# TAB 3 — Volatility & Feature Explorer
# ---------------------------------------------------------------------------
with tab_vol:
    st.subheader("30-Day Rolling Volatility")
    st.caption(
        "Standard deviation of daily returns over a trailing 30-day window — "
        "higher values mark periods of larger, less predictable daily "
        "price swings, often around macroeconomic news or geopolitical events."
    )
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df["date"], y=df["volatility_30"], name="30-Day Volatility",
        line=dict(color=COLORS["LSTM Walk-Forward"], width=1.5),
        fill="tozeroy", fillcolor="rgba(42, 120, 214, 0.12)",
    ))
    fig_vol.update_layout(
        height=380, hovermode="x unified", margin=dict(t=10, b=10),
        yaxis_title="Rolling Std. Dev. of Daily Return",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    st.subheader("Daily Return Distribution")
    st.caption(
        "Shape of the day-over-day percentage change (return) the model "
        "actually trains on — plotted, not just described, because a "
        "distribution's shape (fat tails, symmetry, how tightly it clusters "
        "around zero) says more about non-stationarity than a mean/std pair "
        "alone."
    )
    return_pct = (df["return"] * 100).dropna()
    fig_ret = go.Figure(go.Histogram(
        x=return_pct, nbinsx=120,
        marker_color=COLORS["return_dist"],
        marker_line_width=0,
    ))
    fig_ret.add_vline(x=0, line_dash="dash", line_color="#8a8a86", line_width=1)
    fig_ret.update_layout(
        height=360, margin=dict(t=10, b=10), bargap=0.02,
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
    )
    st.plotly_chart(fig_ret, use_container_width=True)
    st.caption(
        f"Mean {return_pct.mean():+.3f}% · Std {return_pct.std():.3f}% — "
        "tightly centered on zero with a visible peak (more days near-flat "
        "than a normal distribution would predict) and a few multi-percent "
        "tails in both directions. This is exactly the non-stationarity fix "
        "described in Methodology: expressing moves as a percentage keeps "
        "this distribution's shape stable regardless of whether gold is "
        "trading at \\$400 or \\$5,300, which the raw price series cannot do."
    )

    st.divider()

    st.subheader("Feature Explorer")
    st.caption(
        "The engineered features the model actually trains on (see "
        "Methodology for why each one was added)."
    )
    feat_choice = st.multiselect(
        "Select features to plot",
        options=list(FEATURE_LABELS.keys()),
        format_func=lambda k: FEATURE_LABELS[k],
        default=["ma7", "ma30"],
    )
    if feat_choice:
        fig_f = go.Figure()
        palette_cycle = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#4A3AA7"]
        for i, feat in enumerate(feat_choice):
            fig_f.add_trace(go.Scatter(
                x=df["date"], y=df[feat], name=FEATURE_LABELS[feat],
                line=dict(color=palette_cycle[i % len(palette_cycle)]),
            ))
        fig_f.update_layout(height=380, hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.caption("Select at least one feature above to plot it.")

# ---------------------------------------------------------------------------
# TAB 4 — Methodology (the "why", not just the "what")
# ---------------------------------------------------------------------------
with tab_method:
    st.subheader("Methodology")

    if v2 is not None:
        st.markdown("**What changed after the review (v2), and why**")
        st.write(
            "The original notebooks predicted the next-day close PRICE from price-level inputs (close, moving averages, USD momentum) "
            "scaled with statistics from the training period. Gold rose far above that range in 2025-26, so inputs and targets left the "
            "range the scalers and the model had seen, and the LSTM's error exploded (MAE about \\$639 vs about \\$51 for trivial baselines). "
            "The fix: every feature is a return, ratio or oscillator (a test multiplies all prices by a constant and checks that no feature "
            "changes; another edits future prices and checks that past features do not), the target is the next-day log-return converted "
            "back to a price, and the scalers are re-fit inside every walk-forward window."
        )
        st.write(
            "Evaluation was also tightened: results are compared with the 'always up' rate (gold trended up, so that rule already scores "
            "above 50%), reported with 95% confidence intervals, and price errors are tested against a no-change forecast (Diebold-Mariano). "
            "The old directional-accuracy metric compared day-to-day changes of the predictions with those of the actual prices; the new one "
            "compares the sign of the predicted return with the sign of the actual next-day return. Static-training variants (window 20 or 60 "
            "days, 1- and 5-day horizons, training from 2012) and hourly-data experiments (1, 4 and 24 bars, bid and mid prices) reach the same "
            "conclusion; their results are in the repository (daily/results, hourly/results)."
        )
        st.divider()
        st.caption(
            "The sections below describe the original pipeline (60-day window, scalers fit on the training period, price-level target) "
            "and are kept for the record; the changes in v2 are summarized above."
        )

    st.markdown("**Why trading volume isn't a model feature, even though it's in the raw data**")
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        n_total = len(vol)
        n_unreliable = int((vol.isna() | (vol <= 0)).sum())
        pct_unreliable = n_unreliable / n_total * 100
        st.write(
            f"Volume is part of the raw OHLCV data `yfinance` returns and it "
            f"survives into this Gold-layer table untouched, but "
            f"**{pct_unreliable:.1f}% of trading days ({n_unreliable:,} of "
            f"{n_total:,}) show zero or missing volume** — long, multi-year "
            f"stretches with essentially nothing reported, punctuated by "
            f"isolated spikes (chart below). This is a known characteristic "
            f"of Yahoo Finance's continuous-futures volume reporting for "
            f"GC=F specifically — inconsistent across contract rolls and "
            f"much of the older history — not a defect introduced by this "
            f"pipeline's own cleaning. It's also a concrete example of why a "
            f"missing-value check alone isn't sufficient: `isnull().sum()` "
            f"only counts nulls, and would report this column as far "
            f"healthier than it actually is if most of the gaps are stored "
            f"as literal zeros rather than nulls — the problem only becomes "
            f"visible by plotting the distribution, which is exactly why "
            f"`03_Feature_Engineering.ipynb` derives every feature from "
            f"`close` alone (§ Feature Engineering in the main README) "
            f"rather than including volume."
        )
        fig_volq = go.Figure(go.Bar(
            x=df["date"], y=vol, marker_color=COLORS["volume"],
            marker_line_width=0,
        ))
        fig_volq.update_layout(
            height=300, margin=dict(t=10, b=10),
            yaxis_title="Volume", xaxis=dict(rangeslider=dict(visible=True)),
        )
        st.plotly_chart(fig_volq, use_container_width=True)
    else:
        st.info(
            "The `volume` column isn't present in this Gold-layer table — "
            "it was excluded from feature engineering regardless (every "
            "engineered feature derives from `close` alone; see the main "
            "README's Feature Engineering section for the full reasoning)."
        )

    st.markdown("**Why a chronological split, not a random one**")
    st.write(
        f"The dataset is split at {SPLIT_DATE} — every row on or before that "
        "date is training data, everything after is test data. A random "
        "shuffle-then-split would let the model train on days that occur "
        "*after* a test day it is being evaluated on, silently inflating "
        "accuracy. This is a textbook instance of data leakage in time-series "
        "modeling, and a chronological split is the direct fix."
    )

    st.markdown("**Why the scalers are fit on the training set only**")
    st.write(
        "Both the feature scaler and the target scaler are `StandardScaler` "
        "instances fit exclusively on the training partition, then applied "
        "unchanged to the test partition. Fitting on the combined or full "
        "dataset would leak the test set's own mean and variance into "
        "preprocessing — a subtler, easy-to-miss form of the same leakage "
        "problem the chronological split addresses."
    )

    st.markdown("**Why a 60-day sliding window**")
    st.write(
        f"Each training sample is built from the prior {SEQ_LEN} days of "
        "engineered features to predict the next day's close. Sixty days "
        "gives the model roughly a quarter of trading history per "
        "prediction — enough for the slower-moving features (the 30- and "
        "60-day moving averages) to carry meaningful signal within the "
        "window itself."
    )

    st.markdown("**Why walk-forward retraining instead of training once**")
    st.write(
        "The test period reaches gold prices well above anything seen "
        "during training. A model trained once on historical data has no "
        "basis for predicting price behavior at levels it never observed. "
        "Walk-forward retraining re-fits the model every 7 simulated "
        "trading days on the most recent 365-day window, letting it "
        "gradually absorb new price levels as they actually occur, rather "
        "than staying frozen at whatever it learned once."
    )

    st.markdown("**Why Directional Accuracy is reported alongside MAE/RMSE**")
    st.write(
        "MAE and RMSE measure error magnitude in price terms, but say "
        "nothing about whether the model calls the right direction. A model "
        "can score well on MAE simply because prices rarely move much "
        "day-to-day, while being no better than a coin flip at predicting "
        "up-vs-down. Directional Accuracy exposes that distinction directly "
        "— see Model Evaluation for how sharply the two metrics disagree "
        "on which model is actually \"better\"."
    )

    st.markdown("**Known limitation**")
    if v2 is not None:
        lstm2 = v2["models"]["lstm"]
        st.write(
            "No forecasting edge was found. After correcting the scale problem the LSTM's price error is at the no-change level and its "
            f"direction accuracy ({lstm2['direction_from_head']['acc']:.1%}) is not distinguishable from always-up "
            f"({v2['models']['drift']['direction_from_sign']['always_up_acc']:.1%}). With {v2['protocol']['n_test']} test days the confidence "
            "interval is wide (about +/-5 points), features come from price alone (no macro or news data), and one instrument was tested, so "
            "the forecast should be read as 'about today's close within the interval shown', not as a trading signal. The earlier claim that "
            "the model under-predicts the 2025-26 rally while calling direction better than the baselines came from the price-level scale "
            "problem and from a non-standard direction metric."
        )
    else:
        st.write(
            "The LSTM produces smoother predictions than the actual series and "
            "systematically under-predicts the magnitude of sudden upward "
            "moves during the 2025–2026 rally (visualized in the Prediction "
            "Error panel, Model Evaluation tab). It still calls direction "
            "correctly more often than the baselines or random chance — it "
            "understates *how much* the price will move, not *whether* it will "
            "rise."
        )
