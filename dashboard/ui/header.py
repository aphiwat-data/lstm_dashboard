"""The always-visible top of the page: title, the forecast (the product), its chart and a three-number trust strip."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import next_trading_day, run_age_hours
from .logic import VERDICT_WORDS, edge_color, pct_miss, price_verdict
from .theme import COLORS, finish, signed_usd, usd

WINDOWS = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252}


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


def forecast_chart(hist: pd.DataFrame, cur: dict, days: int) -> go.Figure:
    h = hist.tail(days)
    fc, lo, hi, last = cur["predicted_next_close"], cur["interval_90_close"][0], cur["interval_90_close"][1], cur["latest_close"]
    last_date, fc_date = pd.Timestamp(cur["as_of_date"]), next_trading_day(cur["as_of_date"])
    blue, orange = COLORS["model"], COLORS["actual"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["close"], mode="lines", line=dict(color=orange, width=2.4), hovertemplate="%{x|%a %d %b}<br><b>$%{y:,.2f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=[last_date], y=[last], mode="markers", marker=dict(size=9, color=orange, line=dict(color="white", width=1.5)), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[last_date, fc_date], y=[last, fc], mode="lines", line=dict(color=blue, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[fc_date], y=[fc], mode="markers+text", text=[f"${fc:,.0f}"], textposition="middle right", textfont=dict(color=blue, size=13), cliponaxis=False,
        marker=dict(symbol="diamond", size=15, color=blue, line=dict(color="white", width=1.5)),
        error_y=dict(type="data", symmetric=False, array=[hi - fc], arrayminus=[fc - lo], color=blue, thickness=2.5, width=8),
        hovertemplate=f"Forecast for %{{x|%a %d %b}}<br><b>${fc:,.2f}</b><br>90% range ${lo:,.0f} to ${hi:,.0f}<extra></extra>",
    ))
    span = max((fc_date - h["date"].min()).days, 10)
    y_lo, y_hi = min(h["close"].min(), lo), max(h["close"].max(), hi)
    pad = (y_hi - y_lo) * 0.10
    finish(fig, 280, hover="closest")
    fig.update_layout(margin=dict(r=64))  # room for the price label beside the forecast marker
    fig.update_xaxes(range=[h["date"].min(), fc_date + pd.Timedelta(days=max(1, int(span * 0.03)))], tickformat="%d %b")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", range=[y_lo - pad, y_hi + pad])
    return fig


@st.fragment
def _chart(hist: pd.DataFrame, cur: dict) -> None:
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.caption("**Price history and forecast**", help="Orange line: actual daily close. Blue diamond: tomorrow's forecast, with the bar showing its 90% range.")
        window = st.segmented_control("History shown", list(WINDOWS), default="3M", key="hero_window", label_visibility="collapsed") or "3M"
    st.plotly_chart(forecast_chart(hist, cur, WINDOWS[window]), width="stretch", config={"displayModeBar": False})


def _trust_strip(res: dict, preds: pd.DataFrame, reg: dict | None) -> None:
    lstm = res["models"]["lstm"]
    verdict = price_verdict(lstm)
    pct_l, pct_n = pct_miss(preds)
    diff = lstm["mae_usd"] - lstm["naive_mae_usd"]
    cov = reg["interval"]["lstm_ewma_gaussian"]["coverage"] if reg else None
    n_cov = reg["interval"]["lstm_ewma_gaussian"]["n"] if reg else None
    with st.container(horizontal=True, gap="small"):
        st.metric("Typical error", usd(lstm["mae_usd"]), f"{pct_l:.2f}% of the price", delta_color="off", delta_arrow="off", border=True, width="stretch",
                  help=f"Average gap between the forecast and the actual next-day close, measured over the whole test period ({res['protocol']['n_test']} trading days).")
        if cov is not None:
            st.metric("Range accuracy", f"{cov:.1%}", "Target 90%", delta_color="green" if abs(cov - 0.9) <= 0.02 else "orange", delta_arrow="off", border=True, width="stretch",
                      help=f"How often the actual close landed inside the 90% range over {n_cov} test days, using only information known before each forecast. 90% is ideal.")
        st.metric("Beats “tomorrow = today”?", VERDICT_WORDS[verdict], f"{signed_usd(diff)} typical error", delta_color=edge_color(verdict, diff), delta_arrow="off",
                  border=True, width="stretch", help=f"“Tomorrow = today” repeats today's price and is the standard benchmark. The chip is green or red only when the difference is clear (p < 0.05); here p = {lstm['dm_p_vs_naive']:.2f}.")


def render(cur: dict, hist: pd.DataFrame, res: dict | None, preds: pd.DataFrame | None, reg: dict | None) -> None:
    label, color, icon = freshness(cur)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("## Gold Price Forecast")
        trained = cur.get("model_trained_through")
        st.badge(label, icon=icon, color=color, help="The daily pipeline refreshes the forecast at 08:00 (Thailand time). "
                 f"Data through {cur['as_of_date']}" + (f"; model trained through {trained}." if trained else "."))
        st.badge("Research project", color="gray")
    st.caption("Tomorrow's XAU/USD close · LSTM model")

    fc, last, lo, hi = cur["predicted_next_close"], cur["latest_close"], cur["interval_90_close"][0], cur["interval_90_close"][1]
    change, pct = fc - last, (fc / last - 1) * 100
    up = change >= 0
    day = next_trading_day(cur["as_of_date"]).strftime("%A %d %B").upper()
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom", gap="small"):
            with st.container(width="content"):
                st.caption(f"FORECAST FOR {day}")
                st.markdown(f"# {usd(fc)}")
                st.markdown(f":{'green' if up else 'red'}-badge[{'▲' if up else '▼'} {signed_usd(change)} ({pct:+.2f}%)] vs. last close {usd(last)} on {pd.Timestamp(cur['as_of_date']).strftime('%d %b')}")
            with st.container(width="content"):
                st.caption("LIKELY RANGE (90%)", help="The close should land inside this range 9 days out of 10.")
                st.markdown(f"### {usd(lo, 0)} – {usd(hi, 0)}")
        st.divider()
        _chart(hist, cur)
    if res is not None and preds is not None:
        _trust_strip(res, preds, reg)
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.caption("Read it as “about today's close, within the range”. Research only, not trading advice.")
            with st.popover("How to read this", icon=":material/help:"):
                st.markdown(
                    "- **Forecast:** the model's estimate of tomorrow's closing price.\n"
                    "- **Range:** where the close should land 9 days out of 10; a wider range means more uncertainty.\n"
                    "- **Typical error:** how far off the forecast has been on average in testing.\n"
                    "- **Beats “tomorrow = today”?** The simplest forecast just repeats today's price. A model is only useful if it does clearly better.\n\n"
                    "The **How reliable?** tab shows the evidence behind each number."
                )
