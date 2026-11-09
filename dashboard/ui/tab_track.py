"""Forecast vs. actual: the backtest first (the model replayed on past data, 449 days), then the live record (real forecasts saved before each day, scored afterwards)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .cards import NOTES, card_row, metric_card
from .theme import COLORS, ERROR_OVER, ERROR_UNDER, finish, usd

WINDOWS = {"1M": 22, "3M": 66, "6M": 132, "All": None}


@st.fragment
def _backtest(bt: pd.DataFrame, preds: pd.DataFrame) -> None:
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.subheader(f"Backtest · {len(preds)} test days", width="content", help="The model is retrained every 7 trading days and judged only on later days, as in real use. Each point is a forecast made the day before, "
                     "shown on the day it was for. Red bars: the price ended above the forecast; blue bars: below.")
        st.badge("Simulated on past data", icon=":material/history:", color="gray")
    window = st.segmented_control("Window", list(WINDOWS), default="3M", key="bt_window", label_visibility="collapsed") or "3M"
    q = bt if WINDOWS[window] is None else bt.tail(WINDOWS[window])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.07,
                        subplot_titles=("Actual vs. forecast price", "Actual − Forecast ($)"))
    fig.add_trace(go.Scatter(x=q["for_date"], y=q["actual"], name="Actual", line=dict(color=COLORS["actual"], width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=q["for_date"], y=q["forecast"], name="Forecast", line=dict(color=COLORS["model"], width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=q["for_date"], y=q["miss"], marker_color=[ERROR_UNDER if m >= 0 else ERROR_OVER for m in q["miss"]], showlegend=False, name="Actual − Forecast"), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(128,128,128,.6)", line_width=1, row=2, col=1)
    finish(fig, 470, legend=True)
    fig.update_layout(legend=dict(x=1, xanchor="right"))
    fig.update_xaxes(hoverformat="%a %d %b %Y")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_annotations(font_size=12, xanchor="left", x=0)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.download_button("Download CSV", data=preds.to_csv(index=False).encode("utf-8"), file_name="daily_walk_forward_predictions.csv", mime="text/csv",
                       icon=":material/download:")


def _live_chart(scored: pd.DataFrame, pending: pd.DataFrame) -> None:
    """Each real forecast with its 90% range, and the actual close beside it (a circle inside the range, a cross outside), then the forecast still waiting for its close."""
    blue, orange = COLORS["model"], COLORS["actual"]
    inside = scored["inside_90"].astype(str).str.lower().eq("true").to_numpy()
    label = lambda d: f"{pd.Timestamp(d):%a %d %b}"  # noqa: E731  (categories, so weekends leave no gaps)
    x = [label(d) for d in scored["next_date"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=scored["predicted_next_close"], mode="markers", name="Forecast and its 90% range", marker=dict(symbol="diamond", size=12, color=blue, line=dict(color="white", width=1.2)),
                             error_y=dict(type="data", symmetric=False, array=scored["hi_90"] - scored["predicted_next_close"], arrayminus=scored["predicted_next_close"] - scored["lo_90"], color=blue, thickness=2.2, width=7),
                             hovertemplate="%{y:$,.2f}<extra>Forecast</extra>"))
    for mask, name, symbol, line_w in ((inside, "Actual close · inside the range", "circle", 1.5), (~inside, "Actual close · outside the range", "x", 3)):
        if mask.any():
            fig.add_trace(go.Scatter(x=[a for a, m in zip(x, mask) if m], y=scored["actual_next_close"][mask], mode="markers", name=name,
                                     marker=dict(symbol=symbol, size=10 if symbol == "circle" else 11, color=orange, line=dict(color="white" if symbol == "circle" else orange, width=line_w)),
                                     hovertemplate="%{y:$,.2f}<extra>Actual</extra>"))
    lo, hi = list(scored["lo_90"]), list(scored["hi_90"])
    if len(pending):
        p = pending.iloc[-1]
        px_ = label(pd.Timestamp(p["as_of_date"]) + pd.offsets.BDay(1))
        fig.add_trace(go.Scatter(x=[px_], y=[p["predicted_next_close"]], mode="markers", name="Next forecast (not scored yet)", marker=dict(symbol="diamond-open", size=12, color=blue, line=dict(width=2)),
                                 error_y=dict(type="data", symmetric=False, array=[p["hi_90"] - p["predicted_next_close"]], arrayminus=[p["predicted_next_close"] - p["lo_90"]], color=blue, thickness=1.4, width=7),
                                 hovertemplate="%{y:$,.2f}<extra>Next forecast</extra>"))
        lo.append(p["lo_90"]), hi.append(p["hi_90"])
    pad = (max(hi) - min(lo)) * 0.06
    finish(fig, 300, legend=True, hover="closest")
    fig.update_layout(legend=dict(x=0, xanchor="left"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", range=[min(lo) - pad, max(hi) + pad])
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _live(log: pd.DataFrame | None) -> None:
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.subheader("Live record", width="content", help="Every day the pipeline saves its forecast before the day happens. Once the next close is known the forecast is scored here, and nothing can be edited afterwards.")
        st.badge("Real forecasts", icon=":material/bolt:", color="green")
    scored = log.dropna(subset=["actual_next_close"]) if log is not None and len(log) else None
    if scored is None or scored.empty:
        st.info("**Just started.** The first forecast is scored after the next close.", icon=":material/hourglass_top:")
        return
    miss_model = (scored["actual_next_close"] - scored["predicted_next_close"]).abs().mean()
    miss_naive = (scored["actual_next_close"] - scored["latest_close"]).abs().mean()
    inside = scored["inside_90"].astype(str).str.lower().eq("true")
    with card_row():
        metric_card("Forecasts scored", f"{len(scored)}", note=NOTES["scored"])
        metric_card("MAE · average miss", usd(miss_model), f"simple guess: {usd(miss_naive)}", NOTES["mae"],
                    help="Mean absolute error of the live forecasts so far, against the simple guess “tomorrow = today” on the same days.")
        metric_card("Range reliability", f"{int(inside.sum())} of {len(scored)} days", f"{inside.mean():.0%} · goal 90%", NOTES["reliability"],
                    help="With so few days this says little yet.")
    _live_chart(scored.tail(15).reset_index(drop=True), log[log["actual_next_close"].isna()])
    show = scored.tail(15).iloc[::-1].assign(inside=lambda d: d["inside_90"].astype(str).str.lower().eq("true"))
    table = pd.DataFrame({"Forecast made": show["as_of_date"], "Forecast for": pd.to_datetime(show["next_date"]), "Forecast": show["predicted_next_close"],
                          "Actual close": show["actual_next_close"], "Actual − Forecast": (show["actual_next_close"] - show["predicted_next_close"]), "In range?": show["inside"]})
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "Forecast made": st.column_config.DateColumn(format="DD MMM"), "Forecast for": st.column_config.DateColumn(format="DD MMM"),
        "Forecast": st.column_config.NumberColumn(format="$%.2f"), "Actual close": st.column_config.NumberColumn(format="$%.2f"),
        "Actual − Forecast": st.column_config.NumberColumn(format="%+.2f"), "In range?": st.column_config.CheckboxColumn()})


def render(preds: pd.DataFrame | None, log: pd.DataFrame | None, fh: pd.DataFrame) -> None:
    st.caption("The model's forecasts next to what really happened.")
    if preds is not None:
        _backtest(fh[fh["source"] == "backtest"].reset_index(drop=True), preds)
    else:
        st.info("The backtest results are not available in this deployment.")
    st.write("")
    _live(log)
