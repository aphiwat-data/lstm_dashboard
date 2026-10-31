"""Data explorer: price history, volatility (the range in the header is a volatility forecast), daily returns and the model's own indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .data import indicators
from .theme import COLORS, finish

TEST_START = pd.Timestamp("2025-01-09")
PRICE_WINDOWS = {"1Y": 365, "5Y": 365 * 5, "10Y": 365 * 10, "All": None}
INDICATORS = {"f_ret_5": "5-day return", "f_ret_20": "20-day return", "f_vol_20": "20-day volatility", "f_rsi_14": "Momentum (RSI 14)", "f_ma_gap_30": "Gap to 30-day average",
              "f_ma_gap_60": "Gap to 60-day average"}


@st.fragment
def _price(px: pd.DataFrame) -> None:
    st.subheader("Gold price history", help=f"COMEX gold futures (GC=F), daily close since {px['date'].min().year}. Volume is for context only: it is often missing, so the model uses prices alone.")
    window = st.segmented_control("Window", list(PRICE_WINDOWS), default="5Y", key="px_window", label_visibility="collapsed") or "5Y"
    d = px if PRICE_WINDOWS[window] is None else px[px["date"] >= px["date"].max() - pd.Timedelta(days=PRICE_WINDOWS[window])]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=d["date"], y=d["close"], name="Close", line=dict(color=COLORS["actual"], width=1.8), hovertemplate="%{x|%d %b %Y}<br><b>$%{y:,.2f}</b><extra></extra>"), row=1, col=1)
    if d["date"].min() < TEST_START:
        fig.add_vline(x=TEST_START, line_dash="dot", line_color="rgba(128,128,128,.7)", annotation_text="Test period starts", annotation_position="top left", annotation_font_size=12, row=1, col=1)
    vol = pd.to_numeric(d["volume"], errors="coerce")
    fig.add_trace(go.Bar(x=d["date"], y=vol, name="Volume", marker_color=COLORS["volume"], marker_line_width=0, hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>"), row=2, col=1)
    finish(fig, 430, hover="x unified")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Volume", title_font_size=11, row=2, col=1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _volatility(px: pd.DataFrame, cur: dict, vol: tuple | None) -> None:
    st.subheader("Volatility", help="How much the price typically moves in a day. The 90% range in the header is a volatility forecast: wider when markets are jumpy.")
    if vol is not None:
        vdf, comparison = vol
        lo, hi = cur["interval_90_close"]
        sigma = float(np.log(hi / lo) / (2 * 1.6449) * 100)
        with st.container(horizontal=True, gap="small"):
            st.metric("Expected daily move", f"±{sigma:.2f}%", border=True, width="stretch", help="One standard deviation behind today's range: a weighted average of recent forecast errors.")
            st.metric("Yearly equivalent", f"{sigma * np.sqrt(252):.1f}%", border=True, width="stretch", help="Daily move × √252 trading days.")
            st.metric("VIX (US stock market)", f"{vdf['vix_prev_close'].iloc[-1]:.1f}", border=True, width="stretch",
                      help="Shown for context. We tested it as an input and it did not improve the gold forecast.")
        st.markdown("**Daily move vs. expected range**", help="Bars: how far gold actually moved each test day. Line: how far the model expected it could move; 90% of days should stay below the line.")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=vdf["date"], y=vdf["abs_return_pct"], name="Actual daily move", marker_color=COLORS["naive"], opacity=0.5, marker_line_width=0, hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=vdf["date"], y=1.6449 * vdf["sigma_ewma_pct"], name="Expected limit (90% of days)", line=dict(color=COLORS["model"], width=2.4), hovertemplate="%{y:.2f}%<extra></extra>"))
        finish(fig, 300, legend=True)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with st.expander("Which volatility model works best?", icon=":material/query_stats:"):
            rows = [{"Model": r["model"], "Score": r["qlike"], "vs. this model": r["qlike_vs_ewma"], "p-value": r["dm_p_vs_ewma"]} for r in comparison["variance_forecasts"]]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
                "Score": st.column_config.NumberColumn(format="%.3f", help="QLIKE: lower is better"), "vs. this model": st.column_config.NumberColumn(format="%+.3f"),
                "p-value": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Lower score is better. The simple weighted average used here was not beaten.")
    r = px.set_index("date")["close"].pct_change()
    rolling = r.rolling(30).std() * np.sqrt(252) * 100
    fig2 = go.Figure(go.Scatter(x=rolling.index, y=rolling, line=dict(color=COLORS["model"], width=1.6), fill="tozeroy", fillcolor="rgba(42,120,214,0.12)", hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}%<extra></extra>"))
    finish(fig2, 260)
    fig2.update_yaxes(ticksuffix="%")
    st.markdown("**Long-run volatility** (30-day, yearly rate)")
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})


def _returns(px: pd.DataFrame) -> None:
    st.subheader("Daily changes", help="How many days gold moved by each percentage: tightly centred on zero with occasional large moves, whether gold trades at \\$400 or \\$5,000. "
                 "This daily change is what the model predicts.")
    ret = (px["close"].pct_change() * 100).dropna()
    fig = go.Figure(go.Histogram(x=ret, nbinsx=120, marker_color=COLORS["dist"], marker_line_width=0, hovertemplate="%{x:.2f}%<br>%{y} days<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(128,128,128,.7)", line_width=1)
    finish(fig, 280, hover="closest")
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(title_text="Days", title_font_size=11)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"Average {ret.mean():+.3f}% · typical swing {ret.std():.2f}%")


@st.fragment
def _indicators(px: pd.DataFrame) -> None:
    feats = indicators(px.set_index("date")["close"])
    if feats is None:
        return
    st.subheader("What the model looks at", help="The model sees only numbers computed from past prices: recent returns, volatility, momentum and distance from moving averages. Shown: the last two years.")
    key = st.segmented_control("Indicator", list(INDICATORS), format_func=INDICATORS.get, default="f_ret_20", key="indicator", label_visibility="collapsed") or "f_ret_20"
    s = feats[key].dropna().tail(504)
    pct = key.startswith(("f_ret", "f_vol", "f_ma"))
    y = s * 100 if pct else s
    fig = go.Figure(go.Scatter(x=s.index, y=y, line=dict(color=COLORS["model"], width=1.6), hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}" + ("%" if pct else "") + "<extra></extra>"))
    finish(fig, 280)
    fig.update_yaxes(ticksuffix="%" if pct else "")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render(px: pd.DataFrame | None, cur: dict, vol: tuple | None) -> None:
    if px is None:
        st.info("Price history could not be loaded right now. The forecast and its evaluation above are unaffected.", icon=":material/cloud_off:")
        return
    _price(px)
    st.divider()
    _volatility(px, cur, vol)
    st.divider()
    _returns(px)
    st.divider()
    _indicators(px)
