"""Colours, number formats and the shared Plotly style for every chart.

Chart colours are the project's CVD-validated palette: one fixed hue per entity, never cycled (orange = the real price, blue = the model).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "actual": "#EB6834",   # orange: the real price series
    "model": "#2A78D6",    # blue: the LSTM
    "naive": "#8A8F98",    # gray: the simple guess (repeating the last close)
    "ridge": "#E87BA4",    # magenta
    "hgb": "#EDA100",      # yellow
    "good": "#1BAF7A",     # aqua
    "volume": "#8B7FD1",   # soft purple
    "dist": "#1BAF7A",
}
ERROR_OVER, ERROR_UNDER = "#2A78D6", "#E34948"  # model too high (cool) / too low (warm)
GRID = "rgba(128,128,128,0.16)"
# Plotly measures text once with whatever font exists at first paint and caches it, so a web font that loads a moment later would clip legends: charts use the system UI font.
CHART_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


def chart_title(text: str, help: str | None = None) -> None:
    """The heading of one chart or block: the same size as a section heading (h3), so every chart can be found at a glance; the longer explanation sits in the tooltip."""
    st.subheader(text, help=help, anchor=False, width="content")


def panel_title(fig: go.Figure, text: str) -> go.Figure:
    """Style the title of one subplot panel (the label above a chart panel): bold, a little larger and darker than the axis text."""
    fig.update_annotations(selector=dict(text=f"<b>{text}</b>"), font=dict(size=13, color="#344054", family=CHART_FONT), xanchor="left", x=0)
    return fig


def usd(x: float, nd: int = 2) -> str:
    """Dollar amount that is safe inside Streamlit markdown and st.metric (a bare pair of $ would start LaTeX)."""
    return f"\\${x:,.{nd}f}"


def signed_usd(x: float, nd: int = 2) -> str:
    return f"{'+' if x >= 0 else '−'}\\${abs(x):,.{nd}f}"


def finish(fig: go.Figure, height: int = 340, legend: bool = False, hover: str = "x unified") -> go.Figure:
    """Shared look: light grid, no frame, horizontal legend on top, generous hit targets."""
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=30 if legend else 6, b=4), showlegend=legend, hovermode=hover, bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=12)),
        font=dict(size=12, family=CHART_FONT), hoverlabel=dict(font=dict(size=12, family=CHART_FONT)),
    )
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, ticks="", automargin=True)
    return fig
