"""The always-visible top of the page: title and a short briefing, the forecast (the product) with its range, the chart that sets past forecasts beside what really
happened, and the accuracy scorecard. Everything below this (the tabs) is the evidence."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .cards import NOTES, card_row, metric_card
from .data import next_trading_day, run_age_hours
from .logic import edge_color, price_verdict
from .theme import COLORS, ERROR_OVER, ERROR_UNDER, finish, signed_usd, usd

WINDOWS = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252}
VERDICT_BADGE = {"better": "Better than a simple guess", "same": "Not clearly better than a simple guess", "worse": "Worse than a simple guess"}
BAND = "rgba(42,120,214,0.14)"  # the 90% range of past forecasts (the model blue, very light)
BRIEFING = ("An LSTM neural network forecasts the **next trading day's gold price** (XAU/USD) and a 90% range around it.  \n"
            "Every forecast is saved **before** the day happens, then scored against the real close.  \n"
            "It is judged against a simple guess, “tomorrow = today”: the model has to beat that to be useful.")


def _age_text(hours: float | None) -> str:
    if hours is None:
        return "recently"
    return "just now" if hours < 1 else f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def freshness(cur: dict) -> tuple[str, str, str]:
    """(label, colour, icon) of the status badge."""
    if cur["source"] != "live":
        return "Backtest data", "gray", ":material/history:"
    age = run_age_hours(cur)
    if age is not None and age > 72:
        return f"Not updated {_age_text(age)}", "orange", ":material/warning:"
    return f"Live · updated {_age_text(age)}", "green", ":material/bolt:"


def _day(ts: pd.Timestamp, year: bool = False) -> str:
    """'Monday 9 November' (no leading zero, portable across platforms)."""
    return f"{ts:%A} {ts.day} {ts:%B}" + (f" {ts.year}" if year else "")


def forecast_chart(hist: pd.DataFrame, fh: pd.DataFrame, cur: dict, days: int) -> go.Figure:
    """Top: actual close (orange) with the forecasts made the day before (blue, with their 90% range) and the new forecast. Bottom: how far off each one was (actual - forecast)."""
    h = hist.tail(days)
    fc, lo, hi, last = cur["predicted_next_close"], cur["interval_90_close"][0], cur["interval_90_close"][1], cur["latest_close"]
    last_date, fc_date = pd.Timestamp(cur["as_of_date"]), next_trading_day(cur["as_of_date"])
    blue, orange = COLORS["model"], COLORS["actual"]
    past = fh[fh["for_date"] >= h["date"].min()]
    dots = len(h) <= 70  # one marker per day only while the days are far enough apart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.05, subplot_titles=("", "Actual − Forecast ($)"))
    # 90% range of every forecast, the new one included (drawn first so it sits behind the lines)
    ranged = pd.concat([past.dropna(subset=["lo", "hi"])[["for_date", "lo", "hi"]], pd.DataFrame({"for_date": [fc_date], "lo": [lo], "hi": [hi]})])
    fig.add_trace(go.Scatter(x=ranged["for_date"], y=ranged["hi"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ranged["for_date"], y=ranged["lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=BAND, name="90% range", hoverinfo="skip", legendrank=3), row=1, col=1)
    fig.add_trace(go.Scatter(x=past["for_date"], y=past["forecast"], mode="lines+markers" if dots else "lines", name="Forecast", legendrank=2,
                             line=dict(color=blue, width=1.3, dash="dash"), marker=dict(size=5, color=blue), hovertemplate="$%{y:,.2f}<extra>Forecast</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=h["close"], mode="lines", name="Actual", legendrank=1, line=dict(color=orange, width=2.4), hovertemplate="$%{y:,.2f}<extra>Actual</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[last_date], y=[last], mode="markers", marker=dict(size=9, color=orange, line=dict(color="white", width=1.5)), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[last_date, fc_date], y=[last, fc], mode="lines", line=dict(color=blue, width=2, dash="dot"), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[fc_date], y=[fc], mode="markers+text", text=[f"${fc:,.0f}"], textposition="middle right", textfont=dict(color=blue, size=13), cliponaxis=False, name="New forecast", legendrank=4,
        marker=dict(symbol="diamond", size=15, color=blue, line=dict(color="white", width=1.5)),
        error_y=dict(type="data", symmetric=False, array=[hi - fc], arrayminus=[fc - lo], color=blue, thickness=2.5, width=8),
        hovertemplate=f"${fc:,.2f} · 90% range ${lo:,.0f} to ${hi:,.0f}<extra>Forecast for {fc_date:%a %d %b}</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=past["for_date"], y=past["miss"], marker_color=[ERROR_UNDER if m >= 0 else ERROR_OVER for m in past["miss"]], marker_line_width=0, showlegend=False,
                         hovertemplate="%{y:+,.2f}<extra>Actual − Forecast</extra>"), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(128,128,128,.6)", line_width=1, row=2, col=1)
    live = past[past["source"] == "live"]
    if len(live) and past["source"].nunique() > 1:  # the stretch before this line is the backtest (the model replayed on past data), after it real forecasts
        fig.add_vline(x=live["for_date"].min() - pd.Timedelta(hours=12), line_dash="dot", line_color="rgba(128,128,128,.7)", row=1, col=1,
                      annotation_text="Live forecasts start", annotation_position="top left", annotation_font_size=11)
    span = max((fc_date - h["date"].min()).days, 10)
    y_lo, y_hi = min(h["close"].min(), lo, past["lo"].min() if past["lo"].notna().any() else lo), max(h["close"].max(), hi, past["hi"].max() if past["hi"].notna().any() else hi)
    pad = (y_hi - y_lo) * 0.08
    finish(fig, 340, legend=True, hover="x unified")
    fig.update_layout(margin=dict(r=64), legend=dict(x=0, xanchor="left", traceorder="normal"))  # room for the price label beside the forecast marker
    fig.update_xaxes(range=[h["date"].min(), fc_date + pd.Timedelta(days=max(1, int(span * 0.03)))], tickformat="%d %b", hoverformat="%a %d %b %Y")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", range=[y_lo - pad, y_hi + pad], row=1, col=1)
    step = next((c for c in (10, 20, 25, 50, 100, 200, 500, 1000) if c >= past["miss"].abs().max() / 2), 1000) if len(past) else 50
    fig.update_yaxes(tickvals=[-step, 0, step], ticktext=[f"−${step}", "$0", f"${step}"], row=2, col=1)  # a short panel: three ticks, so the labels never touch
    fig.update_annotations(font_size=12, xanchor="left", x=0, selector=dict(text="Actual − Forecast ($)"))
    return fig


@st.fragment
def _chart(hist: pd.DataFrame, fh: pd.DataFrame, cur: dict) -> None:
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.caption("**Price history and forecast**", help="Orange line: the actual daily close. Blue dashed line: the forecast made the day before, with its 90% range in light blue. "
                   "Blue diamond: the new forecast, with the bar showing its range. Bars below: how far off each forecast was (actual minus forecast; red = the price ended higher, blue = lower). "
                   "Before the dotted vertical line the forecasts come from the backtest (the model replayed on past data); after it they are real forecasts saved by the pipeline.")
        window = st.segmented_control("History shown", list(WINDOWS), default="3M", key="hero_window", label_visibility="collapsed") or "3M"
    st.plotly_chart(forecast_chart(hist, fh, cur, WINDOWS[window]), width="stretch", config={"displayModeBar": False})


def _range_block(lo: float, hi: float, reg: dict | None, log: pd.DataFrame | None) -> None:
    """The 90% range and how far it can be trusted, as one unit."""
    st.caption("LIKELY RANGE (90%)", help="The close should land inside this range 9 days out of 10.")
    st.markdown(f"### {usd(lo, 0)} – {usd(hi, 0)}")
    if not reg:
        return
    e = reg["interval"]["lstm_ewma_gaussian"]
    live = ""
    if log is not None and len(log):
        scored = log.dropna(subset=["actual_next_close"])
        if len(scored):
            live = f" Live so far: {int(scored['inside_90'].astype(str).str.lower().eq('true').sum())} of {len(scored)} days."
    ok = abs(e["coverage"] - 0.9) <= 0.02
    st.markdown(f":{'green' if ok else 'orange'}-badge[**Range reliability {e['coverage']:.1%}**] goal 90%",
                help=f"How often the real close landed inside the 90% range in testing: {round(e['coverage'] * e['n'])} of {e['n']} days, using only information known before each forecast.{live} 90% is ideal.")


def _scorecard(res: dict, reg: dict | None) -> None:
    """Accuracy on the backtest window against the simple guess, with the technical names (MAE, RMSE, ...) visible; the verdict is the coloured pill beside the title."""
    lstm = res["models"]["lstm"]
    verdict = price_verdict(lstm)
    diff = lstm["mae_usd"] - lstm["naive_mae_usd"]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"**Accuracy · {res['protocol']['n_test']} backtest days**", width="content", help="Every forecast was made without seeing the day it is judged on. Each number is set against the simple guess "
                    "“tomorrow = today” (the baseline) on the same days. For MAE, RMSE and error % lower is better; R² says how much of the daily price move the forecast explains (0 = nothing).")
        st.badge(VERDICT_BADGE[verdict], icon=":material/balance:", color=edge_color(verdict, diff),
                 help=f"The simple guess “tomorrow = today” misses by {usd(lstm['naive_mae_usd'])} a day on average; this model by {usd(lstm['mae_usd'])} ({signed_usd(diff)}). "
                      f"The pill is green or red only when the difference is clear (p < 0.05); here p = {lstm['dm_p_vs_naive']:.2f}.")
    with card_row():
        metric_card("MAE · average miss", usd(lstm["mae_usd"]), f"simple guess {usd(lstm['naive_mae_usd'])}", NOTES["mae"],
                    help="Mean absolute error: the average gap between the forecast and the actual close, in dollars, per day.")
        metric_card("RMSE", usd(lstm["rmse_usd"]), f"simple guess {usd(lstm['naive_rmse_usd'])}", NOTES["rmse"],
                    help="Root mean squared error: like MAE, but the gaps are squared first, so big misses count extra and it shows how bad the worst days were.")
        if reg:
            m, nc = reg["models"]["lstm"], reg["no_change"]
            metric_card("Error % (MAPE)", f"{m['mape_pct']:.2f}%", f"simple guess {nc['mape_pct']:.2f}%", NOTES["mape"],
                        help="Mean absolute percentage error: the average miss as a share of the price, so it can be compared when gold is cheap or expensive.")
            metric_card("R² (out-of-sample)", f"{m['r2_os_vs_rolling_mean']:+.3f}".replace("-", "−"), f"simple guess {nc['r2_os_vs_rolling_mean']:+.3f}".replace("-", "−"), NOTES["r2"],
                        help="The share of the daily price move the forecast explains. 0 means no better than a plain average; below 0 means worse.")


def render(cur: dict, hist: pd.DataFrame, fh: pd.DataFrame, res: dict | None, reg: dict | None, log: pd.DataFrame | None) -> None:
    label, color, icon = freshness(cur)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("## Gold Price Forecast")
        trained = cur.get("model_trained_through")
        st.badge(label, icon=icon, color=color, help="The daily pipeline refreshes the forecast at 10:00 (Thailand time). "
                 f"Data through {cur['as_of_date']}" + (f"; model trained through {trained}." if trained else "."))
        st.badge("Research project", color="gray")
    st.caption(BRIEFING)

    fc, last, lo, hi = cur["predicted_next_close"], cur["latest_close"], cur["interval_90_close"][0], cur["interval_90_close"][1]
    change, pct = fc - last, (fc / last - 1) * 100
    up = change >= 0
    as_of = pd.Timestamp(cur["as_of_date"])
    target = next_trading_day(as_of)
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="small"):
            st.caption("GOLD CLOSING PRICE FORECAST (XAU/USD)", width="content", help=f"The model's forecast for the close of {_day(target, year=True)}, made from the {_day(as_of)} close.")
            st.markdown(f"### :material/event: Forecast for :blue[{_day(target, year=True)}]", width="content")  # top-right corner: which day this number is for
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="small"):
            with st.container(width="content", gap="xxsmall"):
                st.markdown(f"# {usd(fc)}")
                st.markdown(f":{'green' if up else 'red'}-badge[{'▲' if up else '▼'} {signed_usd(change)} ({pct:+.2f}%)]  \n:gray[:small[vs. the last close {usd(last)} · {as_of:%a} {as_of.day} {as_of:%b}]]")
            with st.container(width="content", border=True):
                _range_block(lo, hi, reg, log)
        st.divider()
        _chart(hist, fh, cur)
    if res is not None:
        _scorecard(res, reg)
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.caption("Read it as “about today's close, within the range”. Research only, not trading advice.")
            with st.popover("How to read this", icon=":material/help:"):
                st.markdown(
                    "- **Forecast:** the model's estimate of the next trading day's closing price.\n"
                    "- **Likely range and range reliability:** where the close should land 9 days out of 10, and how often it really did in testing (90% is ideal).\n"
                    "- **MAE, RMSE, error %:** how far off the forecast was over the test days, set against the simple guess “tomorrow = today” (the baseline). RMSE counts big misses extra.\n"
                    "- **Better than a simple guess?** A model is only useful if it does clearly better than just repeating today's price.\n\n"
                    "The tabs below show the forecasts next to what really happened, the real market data, and the tests behind each number."
                )
