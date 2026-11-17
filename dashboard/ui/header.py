"""The always-visible top of the page: the title with the status badge and a one-line description, the forecast (the product) with its day and its range, the chart that sets past forecasts beside
what really happened, and the accuracy scorecard. Everything below this (the tabs) is the evidence."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .cards import card_row, metric_card
from .data import next_trading_day, run_age_hours
from .theme import COLORS, ERROR_OVER, ERROR_UNDER, chart_title, finish, panel_title, signed_usd, usd

WINDOWS = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252}
BAND = "rgba(42,120,214,0.14)"  # the 90% range of past forecasts (the model blue, very light)
BRIEFING = "An LSTM neural network forecasts the next trading day's gold price (XAU/USD) with a 90% range."  # one line under the title; it must not mention the simple guess (owner)


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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.05, subplot_titles=("", "<b>Actual − Forecast ($)</b>"))
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
    span = max((fc_date - h["date"].min()).days, 10)
    y_lo, y_hi = min(h["close"].min(), lo, past["lo"].min() if past["lo"].notna().any() else lo), max(h["close"].max(), hi, past["hi"].max() if past["hi"].notna().any() else hi)
    pad = (y_hi - y_lo) * 0.08
    finish(fig, 340, legend=True, hover="x unified")
    fig.update_layout(margin=dict(r=64), legend=dict(x=0, xanchor="left", traceorder="normal"))  # room for the price label beside the forecast marker
    fig.update_xaxes(range=[h["date"].min(), fc_date + pd.Timedelta(days=max(1, int(span * 0.03)))], tickformat="%d %b", hoverformat="%a %d %b %Y")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", range=[y_lo - pad, y_hi + pad], row=1, col=1)
    step = next((c for c in (10, 20, 25, 50, 100, 200, 500, 1000) if c >= past["miss"].abs().max() / 2), 1000) if len(past) else 50
    fig.update_yaxes(tickvals=[-step, 0, step], ticktext=[f"−${step}", "$0", f"${step}"], row=2, col=1)  # a short panel: three ticks, so the labels never touch
    panel_title(fig, "Actual − Forecast ($)")
    return fig


@st.fragment
def _chart(hist: pd.DataFrame, fh: pd.DataFrame, cur: dict) -> None:
    live_from = fh.loc[fh["source"] == "live", "for_date"].min()
    since = f"Forecasts before {live_from.day} {live_from:%b} are replayed on past data; from then on they are real forecasts, saved before each day." if pd.notna(live_from) else ""
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        chart_title("Actual vs Forecast", help="Orange line: the real closing price each day. Blue dashed line: the forecast made the day before, with its range shaded light blue. "
                   "Blue diamond: the new forecast and its range. Bars at the bottom: how far off each forecast was (red: the price ended higher than forecast, blue: lower). " + since)
        window = st.segmented_control("History shown", list(WINDOWS), default="1M", key="hero_window", label_visibility="collapsed") or "1M"
    st.plotly_chart(forecast_chart(hist, fh, cur, WINDOWS[window]), width="stretch", config={"displayModeBar": False})


def _range_block(lo: float, hi: float, reg: dict | None, log: pd.DataFrame | None) -> None:
    """The 90% range and how far it can be trusted, as one unit."""
    st.caption("RANGE")
    st.markdown(f"### {usd(lo, 0)} – {usd(hi, 0)}")
    if not reg:
        return
    e = reg["interval"]["lstm_ewma_gaussian"]
    live = ""
    if log is not None and len(log):
        scored = log.dropna(subset=["actual_next_close"])
        if len(scored):
            live = f" So far in live use: {int(scored['inside_90'].astype(str).str.lower().eq('true').sum())} of {len(scored)} days."
    ok = abs(e["coverage"] - 0.9) <= 0.02
    st.markdown(f":{'green' if ok else 'orange'}-badge[**Range reliability {e['coverage']:.1%}**]",
                help=f"The range is where the close is expected to land on 9 days out of 10. In testing, the real close stayed inside it on {round(e['coverage'] * e['n'])} of {e['n']} days.{live}")


def _how_to_read() -> None:
    with st.popover("How to read this", icon=":material/help:"):
        st.markdown(
            "- **Forecast:** the model's estimate of the next trading day's closing price.\n"
            "- **Range and range reliability:** where the close should land 9 days out of 10, and how often it really did in testing.\n"
            "- **MAE, RMSE, error %:** how far off the forecast was on the test days."
        )


def _scorecard(res: dict, reg: dict | None) -> None:
    """Accuracy on the backtest window, with the technical names (MAE, RMSE, ...) visible; definitions sit in the tooltips. "How to read this" sits at the right end of the title row."""
    lstm = res["models"]["lstm"]
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        chart_title(f"Accuracy · {res['protocol']['n_test']} backtest days",
                    help=f"The model was tested on {res['protocol']['n_test']} past days. For each day it forecast the close without having seen it, so these numbers show how it would have done in real use.")
        _how_to_read()
    with card_row():
        metric_card("MAE · average miss", usd(lstm["mae_usd"]),
                    help="The average miss: on a typical day the forecast was about this many dollars away from the real close. Lower is better.")
        metric_card("RMSE", usd(lstm["rmse_usd"]),
                    help="Like the average miss, but big misses count more than small ones, so it shows how bad the worst days were. Lower is better.")
        if reg:
            m = reg["models"]["lstm"]
            metric_card("Error % (MAPE)", f"{m['mape_pct']:.2f}%",
                        help="The average miss as a percentage of the price, so it means the same whether gold is cheap or expensive. Lower is better.")
            metric_card("R² (out-of-sample)", f"{m['r2_os_vs_rolling_mean']:+.3f}".replace("-", "−"),
                        help="Like a test score for catching the daily ups and downs. 1 means every move was predicted; 0 means none was caught, no better than assuming a typical day.")


def render(cur: dict, hist: pd.DataFrame, fh: pd.DataFrame, res: dict | None, reg: dict | None, log: pd.DataFrame | None) -> None:
    label, color, icon = freshness(cur)
    trained = cur.get("model_trained_through")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("## Gold Price Forecast")
        st.badge(label, icon=icon, color=color, help="The forecast is refreshed every morning at 10:00 (Thailand time). "
                 f"It uses prices up to {cur['as_of_date']}" + (f"; the model was last trained on data up to {trained}." if trained else "."))
    st.caption(BRIEFING)

    fc, last, lo, hi = cur["predicted_next_close"], cur["latest_close"], cur["interval_90_close"][0], cur["interval_90_close"][1]
    change, pct = fc - last, (fc / last - 1) * 100
    up = change >= 0
    as_of = pd.Timestamp(cur["as_of_date"])
    target = next_trading_day(as_of)
    with st.container(border=True):
        with st.container(gap=None):  # no gap between the price row and the line: the line then sits as far from the row as from the chart title below it
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="small"):
                with st.container(width="content", gap="xxsmall"):
                    st.markdown(f"### :material/event: Forecast for :blue[{_day(target, year=True)}]")  # which day the number below is for
                    st.markdown(f"# {usd(fc)}")
                    st.markdown(f":{'green' if up else 'red'}-badge[{'▲' if up else '▼'} {signed_usd(change)} ({pct:+.2f}%)]", help=f"Compared with the last close: {usd(last)} on {as_of:%a} {as_of.day} {as_of:%b}.")
                with st.container(width="content", border=True):
                    _range_block(lo, hi, reg, log)
            st.divider()
        _chart(hist, fh, cur)
    if res is not None:
        _scorecard(res, reg)
        st.caption("Made only for a Computer Engineering senior project. This is not trading advice.")
