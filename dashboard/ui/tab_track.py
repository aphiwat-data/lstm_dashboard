"""Track record: the live record (forecasts logged by the daily pipeline, scored once the next close is known) and the walk-forward backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .theme import COLORS, ERROR_OVER, ERROR_UNDER, finish, usd

WINDOWS = {"1M": 22, "3M": 66, "6M": 132, "All": None}


def _live(log: pd.DataFrame | None) -> None:
    st.subheader("Live record", help="Every day the pipeline saves its forecast. Once the next close is known the forecast is scored here, and nothing can be edited afterwards.")
    scored = log.dropna(subset=["actual_next_close"]) if log is not None and len(log) else None
    if scored is None or scored.empty:
        st.info("**Just started.** The first forecast is scored after the next close.", icon=":material/hourglass_top:")
        return
    miss_model = (scored["actual_next_close"] - scored["predicted_next_close"]).abs().mean()
    miss_naive = (scored["actual_next_close"] - scored["latest_close"]).abs().mean()
    inside = scored["inside_90"].astype(str).str.lower().eq("true")
    with st.container(horizontal=True, gap="small"):
        st.metric("Forecasts scored", f"{len(scored)}", border=True, width="stretch")
        st.metric("Typical error", usd(miss_model), f"baseline: {usd(miss_naive)}", delta_color="off", delta_arrow="off", border=True, width="stretch")
        st.metric("Range hit rate", f"{int(inside.sum())} of {len(scored)}", f"{inside.mean():.0%} · target 90%", delta_color="off", delta_arrow="off", border=True, width="stretch")
    show = scored.tail(15).iloc[::-1].assign(inside=lambda d: d["inside_90"].astype(str).str.lower().eq("true"))
    table = pd.DataFrame({"Forecast made": show["as_of_date"], "Forecast for": pd.to_datetime(show["next_date"]), "Forecast": show["predicted_next_close"],
                          "Actual close": show["actual_next_close"], "Actual − Forecast": (show["actual_next_close"] - show["predicted_next_close"]), "In range?": show["inside"]})
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "Forecast made": st.column_config.DateColumn(format="DD MMM"), "Forecast for": st.column_config.DateColumn(format="DD MMM"),
        "Forecast": st.column_config.NumberColumn(format="$%.2f"), "Actual close": st.column_config.NumberColumn(format="$%.2f"),
        "Actual − Forecast": st.column_config.NumberColumn(format="%+.2f"), "In range?": st.column_config.CheckboxColumn()})


@st.fragment
def _backtest(preds: pd.DataFrame) -> None:
    st.subheader(f"Backtest · {len(preds)} test days", help="The model is retrained every 7 trading days and judged only on later days, as in real use. Each point is a forecast made the day before. "
                 "Red bars: the price ended above the forecast; blue bars: below.")
    window = st.segmented_control("Window", list(WINDOWS), default="3M", key="bt_window", label_visibility="collapsed") or "3M"
    q = preds if WINDOWS[window] is None else preds.tail(WINDOWS[window])
    forecast = q["meta_close"] * np.exp(q["pred_ret_lstm"])
    miss = q["meta_next_close"] - forecast
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.07,
                        subplot_titles=("Actual vs. forecast price", "Actual − Forecast ($)"))
    fig.add_trace(go.Scatter(x=q["date"], y=q["meta_next_close"], name="Actual", line=dict(color=COLORS["actual"], width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=q["date"], y=forecast, name="Forecast", line=dict(color=COLORS["model"], width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=q["date"], y=miss, marker_color=[ERROR_UNDER if m >= 0 else ERROR_OVER for m in miss], showlegend=False, name="Actual − Forecast"), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(128,128,128,.6)", line_width=1, row=2, col=1)
    finish(fig, 470, legend=True)
    fig.update_layout(legend=dict(x=1, xanchor="right"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_annotations(font_size=12, xanchor="left", x=0)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.download_button("Download CSV", data=preds.to_csv(index=False).encode("utf-8"), file_name="daily_walk_forward_predictions.csv", mime="text/csv",
                       icon=":material/download:")


def render(preds: pd.DataFrame | None, log: pd.DataFrame | None) -> None:
    _live(log)
    st.write("")
    if preds is not None:
        _backtest(preds)
    else:
        st.info("The backtest results are not available in this deployment.")
