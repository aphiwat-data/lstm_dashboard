"""Data explorer: how gold has traded, and how much it typically moves in a day."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .theme import COLORS, finish

PRICE_WINDOWS = {"1Y": 365, "5Y": 365 * 5, "10Y": 365 * 10, "All": None}


@st.fragment
def _price(px: pd.DataFrame) -> None:
    st.subheader("Gold price history", help=f"COMEX gold futures (GC=F), daily closing price since {px['date'].min().year}.")
    window = st.segmented_control("Window", list(PRICE_WINDOWS), default="5Y", key="px_window", label_visibility="collapsed") or "5Y"
    d = px if PRICE_WINDOWS[window] is None else px[px["date"] >= px["date"].max() - pd.Timedelta(days=PRICE_WINDOWS[window])]
    fig = go.Figure(go.Scatter(x=d["date"], y=d["close"], line=dict(color=COLORS["actual"], width=1.8), hovertemplate="%{x|%d %b %Y}<br><b>$%{y:,.2f}</b><extra></extra>"))
    finish(fig, 360, hover="x unified")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _daily_move(px: pd.DataFrame) -> None:
    st.subheader("How much gold moves per day", help="The typical size of the daily price change, averaged over the last 30 days. Bigger means a more jumpy market, and the likely range in the header gets wider.")
    move = px.set_index("date")["close"].pct_change().rolling(30).std() * 100
    fig = go.Figure(go.Scatter(x=move.index, y=move, line=dict(color=COLORS["model"], width=1.6), fill="tozeroy", fillcolor="rgba(42,120,214,0.12)",
                               hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}% per day<extra></extra>"))
    finish(fig, 280)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render(px: pd.DataFrame | None) -> None:
    if px is None:
        st.info("Price history could not be loaded right now. The forecast above is unaffected.", icon=":material/cloud_off:")
        return
    _price(px)
    st.divider()
    _daily_move(px)
