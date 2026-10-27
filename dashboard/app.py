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
    initial_sidebar_state="auto",  # open on desktop, collapsed on narrow screens (an open drawer would cover the forecast)
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
def load_v2(results_mtime: float):
    """Corrected daily evaluation (daily/walk_forward.py): walk-forward LSTM on log-returns, judged against the
    always-up rate. Returns (results, predictions) or (None, None) when the files are not in the repository."""
    try:
        res = json.loads((V2_DIR / "walk_forward_results.json").read_text())
        preds = pd.read_csv(V2_DIR / "wf_predictions.csv", parse_dates=["date"])
        return res, preds
    except Exception:
        return None, None


FEATURE_SEARCH = [  # label, results sub-folder under daily/results ("" = the main price-only run)
    ("Price history only", ""),
    ("+ cross-asset/macro (6 series)", "with_macro"),
    ("+ NFP calendar", "with_macro_calendar"),
    ("+ FOMC + CPI calendar", "with_macro_calendar_full"),
    ("+ CFTC COT positioning", "with_cot"),
]


@st.cache_data(show_spinner=False)
def feature_search_rows(stamp: float) -> list[dict]:
    """One row per feature-search experiment, read from the committed results so a re-run updates the dashboard
    (nothing here is typed in by hand). `stamp` (latest file mtime) is only there to invalidate the cache."""
    rows = []
    for label, folder in FEATURE_SEARCH:
        try:
            res = json.loads((V2_DIR / folder / "walk_forward_results.json").read_text())
            preds = pd.read_csv(V2_DIR / folder / "wf_predictions.csv")
        except Exception:
            continue
        lstm, hd = res["models"]["lstm"], res["models"]["lstm"]["direction_from_head"]
        rows.append(dict(label=label, n=res["protocol"]["n_test"], acc=hd["acc"], auc=hd.get("auc"), p=hd["p_vs_always_up"],
                         r2=lstm.get("r2_returns"), up_share=float((preds["p_up_lstm"] > 0.5).mean())))
    return rows


def feature_search_stamp() -> float:
    files = [V2_DIR / folder / "walk_forward_results.json" for _, folder in FEATURE_SEARCH]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


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
_v2_file = V2_DIR / "walk_forward_results.json"
v2, v2_preds          = load_v2(_v2_file.stat().st_mtime if _v2_file.exists() else 0.0)  # mtime in the cache key: a refresh shows up without a restart

# 30-day rolling volatility, computed on the fly for the dashboard view (the
# model itself trains on the 7-day volatility_7 feature — see Feature Explorer)
df["volatility_30"] = df["return"].rolling(30).std()

# ---------------------------------------------------------------------------
# Sidebar — project credits on top, then data freshness; technical details live in an expander
# ---------------------------------------------------------------------------
ASSETS = Path(__file__).resolve().parent / "assets"
LOGOS = [(ASSETS / "mfu_logo.png", 72), (ASSETS / "adt_logo.png", 96)]  # (file, display width in px): the crest is tall, the ADT mark round
TEAM = [("Mr. Achira Lueablae", "6631501120"), ("Mr. Aphiwat Chioewvijit", "6631501121")]
ADVISOR = "Asst. Prof. Pattaramon Vuttipittayamongkol"
COMMITTEE = ["Asst. Prof. Khwunta Kirimasthong", "Asst. Prof. Surapong Uttama"]

with st.sidebar:
    with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="medium"):
        for logo, width in LOGOS:
            if logo.exists():
                with st.container(width=width):  # image(width=...) would downscale the file itself and look soft on high-DPI screens
                    st.image(str(logo), width="stretch")
    st.markdown("**Next-Day Gold Price  \nForecasting with LSTM**", text_alignment="center")
    st.caption("Bachelor of Engineering  \nMajor Computer Engineering  \nSchool of Applied Digital Technology  \nMae Fah Luang University · 2026", text_alignment="center")
    st.markdown(
        "**Developed by**\n\n" + "\n".join(f"- {name}  \n  :gray[Student ID {sid}]" for name, sid in TEAM)
        + f"\n\n**Advisor**\n\n- {ADVISOR}"
        + "\n\n**Committee**\n\n" + "\n".join(f"- {name}" for name in COMMITTEE)
    )
    st.divider()
    if v2 is not None:
        st.caption(f"Data through **{v2['latest_forecast']['as_of_date']}** · generated {v2['latest_forecast']['generated_at'][:10]}")
    else:
        st.caption(f"Data through **{latest_row['date'].date()}**")
    with st.expander("Model and data details"):
        st.markdown("**Data**")
        st.text(f"GC=F (COMEX Gold Futures)\n{df['date'].min().date()} – {df['date'].max().date()}\n{len(df):,} rows")
        st.markdown("**Model**")
        if v2 is not None:
            pr = v2["protocol"]
            st.text(
                f"LSTM({pr['lstm_units']}), 2 heads: return + direction\nInput window: {pr['window']} days\n"
                f"Retrained every {pr['retrain_every_days']} days\non a {pr['train_len_days']}-day window\n"
                f"Target: next-day log-return\nTest: {pr['test_start']} – {pr['test_end']}"
            )
        else:
            st.text(f"2-layer LSTM (64, 32 units)\nInput window: {SEQ_LEN} days\nRetrained every {RETRAIN_EVERY_DAYS} days\n365-day rolling window")
            st.text(f"Last retrained: {model_last_retrained.date()} ({days_since_retrain} day(s) ago)")
            if days_since_retrain > RETRAIN_EVERY_DAYS:
                st.warning(f"Overdue for retrain by {days_since_retrain - RETRAIN_EVERY_DAYS} day(s).", icon="⚠️")

# ---------------------------------------------------------------------------
# Forecast-first header: the next-day forecast is the project's headline output,
# so it gets the biggest number on the page plus a chart showing it in context.
# ---------------------------------------------------------------------------
WINDOWS = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252}


def next_trading_day(d) -> pd.Timestamp:
    return pd.Timestamp(d) + pd.offsets.BDay(1)


def model_verdict(lstm: dict, up: float) -> tuple[str, str]:
    """('better' | 'worse' | 'same') for price error vs. no-change and for direction vs. always-up, at p < 0.05."""
    hd = lstm["direction_from_head"]
    p_price, p_dir = lstm["dm_p_vs_naive"], hd["p_vs_always_up"]
    price = "better" if p_price < 0.05 and lstm["mae_skill_vs_naive"] > 0 else "worse" if p_price < 0.05 and lstm["mae_skill_vs_naive"] < 0 else "same"
    direction = "better" if p_dir < 0.05 and hd["acc"] > up else "worse" if p_dir < 0.05 and hd["acc"] < up else "same"
    return price, direction


def pct_miss(p: pd.DataFrame) -> tuple[float, float]:
    """Mean absolute percentage miss of the LSTM forecast and of a "tomorrow = today" guess, on the same test days."""
    actual = p["meta_next_close"]
    lstm = (p["meta_close"] * np.exp(p["pred_ret_lstm"]) - actual).abs() / actual
    naive = (p["meta_close"] - actual).abs() / actual
    return float(lstm.mean() * 100), float(naive.mean() * 100)


def edge_color(edge: str, diff: float) -> str:
    """Chip colour for "model error minus baseline error": green / red only when the difference is statistically clear (p < 0.05),
    orange (a bit worse) or blue (a bit better) when it is within noise."""
    return {"better": "green", "worse": "red"}.get(edge) or ("orange" if diff > 0 else "blue")


def forecast_inputs() -> dict:
    if v2 is not None:
        lf = v2["latest_forecast"]
        hist = v2_preds[["date", "meta_close"]].rename(columns={"meta_close": "close"})
        as_of = pd.Timestamp(lf["as_of_date"])
        if as_of > hist["date"].max():
            hist = pd.concat([hist, pd.DataFrame({"date": [as_of], "close": [lf["latest_close"]]})], ignore_index=True)
        return dict(hist=hist.sort_values("date"), as_of=as_of, last_close=lf["latest_close"], fc=lf["predicted_next_close"],
                    lo=lf["interval_90_close"][0], hi=lf["interval_90_close"][1])
    return dict(hist=df[["date", "close"]], as_of=latest_row["date"], last_close=float(latest_row["close"]), fc=next_day_pred, lo=forecast_lo, hi=forecast_hi)


def render_hero(fi: dict) -> None:
    """Built only from native Streamlit elements (no raw HTML/CSS), so it cannot break on a different Streamlit
    version or a stale browser tab. Horizontal containers wrap by content width, not viewport width: with the sidebar
    open on a small window the stats drop below the price instead of squeezing the price onto several lines."""
    fc, last, lo, hi = fi["fc"], fi["last_close"], fi["lo"], fi["hi"]
    change, pct = fc - last, (fc / last - 1) * 100
    up = change >= 0
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute", gap="medium"):
            with st.container(width="content"):
                st.caption(f"FORECAST FOR {next_trading_day(fi['as_of']).strftime('%a %d %b %Y').upper()}")
                st.markdown(f"# \\${fc:,.2f}")
                st.markdown(f"**:{'green' if up else 'red'}[{'▲' if up else '▼'} {'+' if up else '−'}\\${abs(change):,.2f} ({pct:+.2f}%)]** vs. last close")
            with st.container(horizontal=True, width="content", gap="medium"):
                with st.container(width="content"):
                    st.caption("90% LIKELY RANGE")
                    st.markdown(f"#### \\${lo:,.0f}–\\${hi:,.0f}")
                with st.container(width="content"):
                    st.caption(f"LAST CLOSE · {pd.Timestamp(fi['as_of']).strftime('%d %b').upper()}")
                    st.markdown(f"#### \\${last:,.2f}")
        st.caption("Actual closes fell inside the range on 90% of test days. Research forecast, not trading advice.")


def forecast_chart(fi: dict, window_days: int) -> go.Figure:
    h = fi["hist"].tail(window_days)
    fc, lo, hi, last = fi["fc"], fi["lo"], fi["hi"], fi["last_close"]
    last_date, fc_date = pd.Timestamp(fi["as_of"]), next_trading_day(fi["as_of"])
    blue = COLORS["LSTM Walk-Forward"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["close"], mode="lines", name="Actual close", line=dict(color=COLORS["actual"], width=2.4),
                             hovertemplate="%{x|%a %d %b}<br>$%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[last_date, fc_date], y=[last, fc], mode="lines", line=dict(color=blue, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[fc_date], y=[fc], mode="markers",
        marker=dict(symbol="diamond", size=16, color=blue, line=dict(color="white", width=1.5)),
        error_y=dict(type="data", symmetric=False, array=[hi - fc], arrayminus=[fc - lo], color=blue, thickness=3, width=9),
        hovertemplate=f"Forecast for %{{x|%a %d %b}}<br>${fc:,.2f}<br>90% range ${lo:,.0f} to ${hi:,.0f}<extra></extra>",
    ))
    span = max((fc_date - h["date"].min()).days, 10)
    y_lo, y_hi = min(h["close"].min(), lo), max(h["close"].max(), hi)
    pad = (y_hi - y_lo) * 0.10
    fig.update_layout(
        height=336, margin=dict(l=8, r=8, t=8, b=8), showlegend=False, hovermode="closest",
        xaxis=dict(showgrid=False, range=[h["date"].min(), fc_date + pd.Timedelta(days=int(span * 0.16))]),
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(128,128,128,.18)", range=[y_lo - pad, y_hi + pad]),
    )
    return fig


fi = forecast_inputs()
st.markdown("### Next-Day Gold Price Forecast")
st.caption("XAU/USD · LSTM model, retrained every 7 trading days")
render_hero(fi)
win = st.radio("History shown", list(WINDOWS), index=1, horizontal=True, label_visibility="collapsed", key="hero_window")
st.plotly_chart(forecast_chart(fi, WINDOWS[win]), width="stretch", config={"displayModeBar": False})
st.caption("Line: actual daily close · ◆ forecast for the next trading day, bar = 90% likely range")

# Error in price terms is what a price forecast is judged by. Fixed-width metrics inside a wrapping container sit in one row on a wide page
# and stack on a narrow one (st.columns only stacks by viewport width, which the open sidebar defeats).
if v2 is not None:
    lstm_v2, up_v2 = v2["models"]["lstm"], v2["models"]["drift"]["direction_from_sign"]["always_up_acc"]
    pct_lstm, pct_naive = pct_miss(v2_preds)
    price_edge, _ = model_verdict(lstm_v2, up_v2)
    miss_color = edge_color(price_edge, lstm_v2["mae_usd"] - lstm_v2["naive_mae_usd"])
    chip_key = " Chip colour: green = clearly better than that guess, red = clearly worse, orange or blue = no clear difference (within noise)."
    with st.container(horizontal=True, gap="medium"):
        st.metric("Typical daily miss", f"${lstm_v2['mae_usd']:,.2f}", f"{lstm_v2['mae_usd'] - lstm_v2['naive_mae_usd']:+.2f} vs. “tomorrow = today”", delta_color=miss_color, width=230,
                  help="Average gap between the forecast and the actual next-day close on the test period (MAE). A guess of “tomorrow's price = today's price” scores almost the same." + chip_key)
        st.metric("Typical miss, % of price", f"{pct_lstm:.2f}%", f"{pct_lstm - pct_naive:+.3f} pt vs. “tomorrow = today”", delta_color=miss_color, width=230,
                  help="The same miss as a share of the price (mean absolute percentage error). A guess of “tomorrow's price = today's price” scores almost the same." + chip_key)
        st.metric("Data through", pd.Timestamp(fi["as_of"]).strftime("%d %b"), f"{pd.Timestamp(fi['as_of']).year} · {v2['protocol']['n_test']} test days", delta_color="blue", delta_arrow="off", width=230)
    if price_edge == "better":
        st.caption("Some edge over a simple baseline was measured. See **How reliable?** for the evidence and caveats.")
    else:
        st.caption("Read this as roughly today's close within the range shown: on this data the model does not beat the simple “tomorrow = today” guess. See **How reliable?**.")
else:
    with st.container(horizontal=True, gap="medium"):
        st.metric("Typical daily miss (MAE)", f"${results['LSTM Walk-Forward']['MAE']:,.2f}", width=230)
        st.metric("Data through", pd.Timestamp(fi["as_of"]).strftime("%d %b"), str(pd.Timestamp(fi["as_of"]).year), delta_color="blue", delta_arrow="off", width=230)
    st.caption("Treat this as a rough guide, not a precise price target: the model under-predicts sharp upward moves.")


def render_track_record(v2: dict, p: pd.DataFrame) -> None:
    st.subheader("Forecast vs. actual")
    win_tr = st.radio("Window", ["1M", "3M", "6M", "All"], index=1, horizontal=True, label_visibility="collapsed", key="track_window")
    q = p.tail({"1M": 22, "3M": 66, "6M": 132, "All": len(p)}[win_tr])
    pred_close = q["meta_close"] * np.exp(q["pred_ret_lstm"])
    err = q["meta_next_close"] - pred_close
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06,
        subplot_titles=("Next-day close: actual vs. forecast", "Miss (actual − forecast, USD)"),
    )
    fig.add_trace(go.Scatter(x=q["date"], y=q["meta_next_close"], name="Actual next close", line=dict(color=COLORS["actual"], width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=q["date"], y=pred_close, name="Forecast", line=dict(color=COLORS["LSTM Walk-Forward"], width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=q["date"], y=err, marker_color=[ERROR_UNDER if e >= 0 else ERROR_OVER for e in err], showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="#8a8a86", line_width=1, row=2, col=1)
    fig.update_layout(height=500, hovermode="x unified", margin=dict(t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Each point is a forecast made the day before (x-axis = the day it was made). Both lines show the NEXT trading day's close.")
    st.download_button("Download predictions (CSV)", data=p.to_csv(index=False).encode("utf-8"), file_name="daily_walk_forward_predictions.csv", mime="text/csv")


def render_scorecard(v2: dict, p: pd.DataFrame) -> None:
    proto, models = v2["protocol"], v2["models"]
    up = models["drift"]["direction_from_sign"]["always_up_acc"]
    lstm = models["lstm"]
    hd = lstm["direction_from_head"]
    price_edge, dir_edge = model_verdict(lstm, up)
    phrase = {"better": "beats", "worse": "is worse than", "same": "cannot be told apart from"}
    pct_lstm, pct_naive = pct_miss(p)

    st.subheader("How reliable is the forecast?")
    if price_edge == "better":
        st.success("**Verdict: some measurable edge.** Check the details below before relying on it.")
    else:
        st.info("**Verdict: no reliable edge.** The price forecast does not beat a simple “tomorrow = today” guess, so treat it as “about today's close, within the range shown”.")
    r2_line = (
        f"- **Return fit:** R² {lstm['r2_returns']:+.3f} (a constant zero-return forecast scores {lstm['naive_r2_returns']:+.3f}); rank correlation IC {lstm['ic_spearman']:+.3f} (p = {lstm['ic_p']:.2f}).\n"
        if "r2_returns" in lstm else ""
    )
    st.markdown(
        f"- **Price error:** typical miss \\${lstm['mae_usd']:,.2f} per day ({pct_lstm:.2f}% of the price; RMSE \\${lstm['rmse_usd']:,.2f}). The model {phrase[price_edge]} a “tomorrow = today” guess (\\${lstm['naive_mae_usd']:,.2f}, {pct_naive:.2f}%; Diebold-Mariano p = {lstm['dm_p_vs_naive']:.2f}).\n"
        f"{r2_line}"
        f"- **Sample size:** {proto['n_test']} test days ({proto['test_start']} to {proto['test_end']}), so small differences between models are within noise."
    )

    st.divider()
    st.markdown("**Typical daily miss by model (USD, lower is better)**")
    st.caption(f"The dashed line is a “tomorrow = today” guess (\\${lstm['naive_mae_usd']:,.2f}). A model has to land clearly below it to add value.")
    mae_entries = [("Rolling drift", "drift", "#8a8a86"), ("Ridge", "ridge", COLORS["Linear Regression"]),
                   ("HistGB regressor", "hist_gb_reg", COLORS["AR(5) Baseline"]), ("LSTM", "lstm", COLORS["LSTM Walk-Forward"])]
    maes = [models[k]["mae_usd"] for _, k, _ in mae_entries]
    fig_m = go.Figure(go.Bar(
        x=[n for n, _, _ in mae_entries], y=maes, marker_color=[c for _, _, c in mae_entries], text=[f"${m:,.2f}" for m in maes],
        textposition="inside", insidetextanchor="start", textfont=dict(size=13, color="white"),
    ))
    fig_m.add_hline(y=lstm["naive_mae_usd"], line_dash="dash", line_color="#8a8a86", annotation_text="“tomorrow = today”", annotation_position="top left")
    fig_m.update_layout(height=320, margin=dict(t=20, b=10), showlegend=False, yaxis=dict(range=[0, max(maes) * 1.12], title="Typical miss (USD)"))
    st.plotly_chart(fig_m, width="stretch", config={"displayModeBar": False})

    with st.expander("Secondary check: up/down calls (a classification-style view)"):
        st.caption(
            "Calling the next day's direction treats the forecast as a yes/no classifier, which is not how a price forecast is judged, so the error measures "
            "above are the main yardstick. It is shown because the proposal's ~70% target was phrased this way and because it exposes how the model behaves."
        )
        st.markdown(
            f"**{hd['acc']:.1%}** of up/down calls were correct (95% CI {hd['ci95'][0]:.1%} to {hd['ci95'][1]:.1%}). The model {phrase[dir_edge]} “always up” "
            f"({up:.1%}; p = {hd['p_vs_always_up']:.2f}). Gold rose on {up:.1%} of test days, so always saying “up” already scores that; 50% is not the bar to beat. "
            f"With {proto['n_test']} test days the interval is about ±{(hd['ci95'][1] - hd['ci95'][0]) / 2 * 100:.0f} points."
        )
        st.caption("Bars show 95% confidence intervals; the dashed line is “always up”.")
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
        fig_d.add_hline(y=up * 100, line_dash="dash", line_color="#8a8a86", annotation_text=f"always up {up:.1%}", annotation_position="top left")
        fig_d.update_layout(height=360, margin=dict(t=20, b=10), showlegend=False, yaxis=dict(range=[35, 72], title="Correct up/down calls (%)"))
        st.plotly_chart(fig_d, width="stretch", config={"displayModeBar": False})

    with st.expander("Detailed metrics for every model (MAE, AUC, IC, R², p-values)"):
        labels = {"drift": "Rolling drift", "ridge": "Ridge", "hist_gb_reg": "HistGB regressor", "hist_gb_clf": "HistGB classifier", "lstm": "LSTM", "momentum_rule": "Momentum rule"}
        rows = {}
        for key, r in models.items():
            d = r.get("direction_from_head") or r["direction_from_sign"]
            rows[labels.get(key, key)] = {
                "MAE ($)": r.get("mae_usd"), "MAE skill vs no-change": r.get("mae_skill_vs_naive"), "DM p (vs no-change)": r.get("dm_p_vs_naive"),
                "Direction acc. (%)": d["acc"] * 100, "vs always-up (pt)": (d["acc"] - d["always_up_acc"]) * 100, "p (vs always-up)": d["p_vs_always_up"],
                "AUC": (r.get("direction_from_head") or {}).get("auc"), "IC (rank corr.)": r.get("ic_spearman"), "R² (returns)": r.get("r2_returns"),
            }
        table = pd.DataFrame(rows).T.astype(float)  # None -> NaN so missing cells render as a dash
        fmt = {"MAE ($)": "{:.2f}", "MAE skill vs no-change": "{:+.4f}", "DM p (vs no-change)": "{:.3f}", "Direction acc. (%)": "{:.1f}",
               "vs always-up (pt)": "{:+.1f}", "p (vs always-up)": "{:.3f}", "AUC": "{:.3f}", "IC (rank corr.)": "{:+.3f}", "R² (returns)": "{:+.4f}"}
        shown = table.apply(lambda col: col.map(lambda v: "–" if pd.isna(v) else fmt[col.name].format(v)))  # display copy; `table` keeps the numbers
        st.dataframe(shown, width="stretch")
        st.download_button("Download metrics table (CSV)", data=table.to_csv().encode("utf-8"), file_name="daily_walk_forward_metrics.csv", mime="text/csv")


tab_track, tab_data, tab_rel, tab_method = st.tabs([
    "Track record", "Data explorer", "How reliable?", "Method",
])

with tab_track:
    if v2 is not None:
        render_track_record(v2, v2_preds)
    else:
        st.info("The forecast track record is built from daily/results, which is not available in this deployment.")

# ---------------------------------------------------------------------------
# Data explorer — full historical price chart (+ volume panel, when present)
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Historical Gold Price (GC=F)")
    st.caption("Drag the slider or use 1Y / 5Y / All to zoom.")

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

    st.plotly_chart(fig_hist, width="stretch")

    st.caption(
        "Long-term uptrend with a sharp acceleration in 2025–2026. "
        + ("Volume is shown for reference only (see Method for why it is not a model feature)." if has_volume else "")
    )

# ---------------------------------------------------------------------------
# How reliable? — verdict and metrics, plus the original pipeline kept for the record
# ---------------------------------------------------------------------------
with tab_rel:
    if v2 is not None:
        render_scorecard(v2, v2_preds)
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
        st.plotly_chart(fig, width="stretch")

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
            col.plotly_chart(fig_b, width="stretch")

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
        mcol3.plotly_chart(fig_d, width="stretch")

        metrics_table = pd.DataFrame({
            "MAE ($)": {m: round(results[m]["MAE"], 2) for m in model_names},
            "RMSE ($)": {m: round(results[m]["RMSE"], 2) for m in model_names},
            "Directional Accuracy (%)": {m: round(dir_acc[m], 2) for m in model_names},
        })
        st.dataframe(metrics_table, width="stretch")

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
# Data explorer (continued) — volatility & feature explorer
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("30-Day Rolling Volatility")
    st.caption("Standard deviation of daily returns over a trailing 30-day window. Higher = bigger, less predictable swings.")
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
    st.plotly_chart(fig_vol, width="stretch")

    st.divider()

    st.subheader("Daily Return Distribution")
    st.caption("Day-over-day percentage change, the quantity the model actually predicts.")
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
    st.plotly_chart(fig_ret, width="stretch")
    st.caption(f"Mean {return_pct.mean():+.3f}% · Std {return_pct.std():.3f}% — tightly centred on zero with fat tails, and stable whether gold trades at \\$400 or \\$5,300.")

    st.divider()

    st.subheader("Feature Explorer")
    st.caption("The engineered features the model trains on.")
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
        st.plotly_chart(fig_f, width="stretch")
    else:
        st.caption("Select at least one feature above to plot it.")

# ---------------------------------------------------------------------------
# TAB 4 — Methodology (the "why", not just the "what")
# ---------------------------------------------------------------------------
with tab_method:
    st.subheader("Methodology")
    st.caption("Why the pipeline is built this way — click any item to expand.")

    if v2 is not None:
        lstm2 = v2["models"]["lstm"]
        r2_clause = f" and its R² on daily returns is {lstm2['r2_returns']:+.3f}" if "r2_returns" in lstm2 else ""
        st.markdown(
            f"**Bottom line:** no forecasting edge was found. The LSTM's typical daily miss (\\${lstm2['mae_usd']:,.2f}) is level with a "
            f"“tomorrow = today” guess (\\${lstm2['naive_mae_usd']:,.2f}){r2_clause}. "
            "Read the forecast as roughly today's close, not a trading signal."
        )

        with st.expander("What changed after the review, and why"):
            st.write(
                "The original notebooks predicted the next-day close PRICE from price-level inputs, scaled with training-period "
                "statistics. Gold rose far above that range in 2025-26, so the LSTM's error exploded (MAE about \\$639 vs \\$51 for "
                "trivial baselines). Fix: every feature is a return/ratio/oscillator, the target is a log-return, and scalers are "
                "re-fit inside every walk-forward window."
            )
            st.write(
                "Evaluation was also tightened: price errors are tested against a no-change forecast (Diebold-Mariano), and up/down "
                "calls are compared against the 'always up' rate (gold trended up, so that rule already scores above 50%) with 95% "
                "confidence intervals."
            )

        with st.expander("Also tested, still null: macro, calendar and positioning features"):
            fs_rows = feature_search_rows(feature_search_stamp())
            if fs_rows:
                p_min = min(fs_rows, key=lambda r: r["p"])
                aucs = [r["auc"] for r in fs_rows if r["auc"] is not None]
                up_lo, up_hi = min(r["up_share"] for r in fs_rows), max(r["up_share"] for r in fs_rows)
                st.caption(
                    f"{len(fs_rows) - 1} additions to the LSTM, same protocol. AUC ranges from {min(aucs):.3f} to {max(aucs):.3f}; the smallest p-value against "
                    f"'always up' is {p_min['p']:.2f} ({'not significant' if p_min['p'] >= 0.05 else 'below 0.05, see Track record before reading anything into it'}). "
                    f"Checked directly, the model predicts 'up' on {up_lo:.0%}-{up_hi:.0%} of days whichever features it gets, "
                    "which is why accuracy can move while AUC does not."
                )
                if len({r["n"] for r in fs_rows}) > 1:
                    st.warning("These rows were evaluated on different test windows (see the Test days column), so they are not strictly comparable.", icon="⚠️")
                st.dataframe(pd.DataFrame([
                    {"Feature set": r["label"], "Test days": r["n"], "Direction acc.": f"{r['acc']:.1%}", "AUC": "–" if r["auc"] is None else f"{r['auc']:.3f}",
                     "p vs always-up": f"{r['p']:.2f}", "R² (returns)": "–" if r["r2"] is None else f"{r['r2']:+.3f}", "Predicts 'up' on": f"{r['up_share']:.0%} of days"}
                    for r in fs_rows]).set_index("Feature set"), width="stretch")
                st.caption("Full results: `daily/results/with_macro/`, `with_macro_calendar/`, `with_macro_calendar_full/`, `with_cot/`.")
            else:
                st.caption("Feature-search results are not available in this deployment.")

        st.caption("Sections below describe the original pipeline (60-day window, price-level target), kept for the record.")

    with st.expander("Why trading volume isn't a model feature"):
        if "volume" in df.columns:
            vol = pd.to_numeric(df["volume"], errors="coerce")
            n_total = len(vol)
            n_unreliable = int((vol.isna() | (vol <= 0)).sum())
            pct_unreliable = n_unreliable / n_total * 100
            st.write(
                f"**{pct_unreliable:.1f}% of trading days ({n_unreliable:,} of {n_total:,}) show zero or missing volume** — a known "
                "gap in Yahoo Finance's continuous-futures volume reporting for GC=F, not a defect in this pipeline. Every engineered "
                "feature is derived from `close` alone instead."
            )
            fig_volq = go.Figure(go.Bar(x=df["date"], y=vol, marker_color=COLORS["volume"], marker_line_width=0))
            fig_volq.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="Volume", xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig_volq, width="stretch")
        else:
            st.caption("The `volume` column isn't present in this table — excluded from feature engineering regardless.")

    with st.expander("Why a chronological split, not a random one"):
        st.write(
            f"Split at {SPLIT_DATE}: everything before is training, everything after is test. A random shuffle would let the model "
            "train on days *after* a test day it's evaluated on — a textbook data leak."
        )

    with st.expander("Why the scalers are fit on the training set only"):
        st.write(
            "Both scalers are fit exclusively on the training partition. Fitting on the full dataset would leak the test set's own "
            "mean/variance into preprocessing — a subtler form of the same leakage the chronological split prevents."
        )

    with st.expander("Why a 60-day sliding window"):
        st.write(
            f"Each sample uses the prior {SEQ_LEN} days of features — enough for the slower moving averages (30-, 60-day) to carry "
            "meaningful signal within the window itself."
        )

    with st.expander("Why walk-forward retraining instead of training once"):
        st.write(
            "The test period reaches gold prices never seen in training. Retraining every 7 days on the latest 365-day window lets "
            "the model absorb new price levels as they occur, instead of staying frozen."
        )

    with st.expander("Why up/down accuracy is only a secondary check"):
        up_share = float((v2_preds["p_up_lstm"] > 0.5).mean()) if v2_preds is not None else None
        st.write(
            "Up/down accuracy treats the forecast as a yes/no classifier. A price forecast is judged by how far off it is (MAE, RMSE, "
            "% error) and by R², each against a “tomorrow = today” baseline. Direction is still reported as a secondary check, for two "
            "reasons: the proposal's ~70% target was phrased as direction accuracy, and the check shows how the model behaves."
            + (f" It says “up” on {up_share:.0%} of test days, so its up/down score mostly tracks the share of up days in the test period." if up_share is not None else "")
        )

    st.divider()
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
