"""Metric cards: the number, one chip for the comparison and one small line saying what the metric measures and when to use it."""
from __future__ import annotations

import streamlit as st

# One short line per metric, reused wherever the metric appears: what it measures, and the use case.
NOTES = {
    "mae": "Average miss in dollars. Best for the everyday error.",
    "rmse": "Like MAE, but big misses count extra. Best for spotting bad days.",
    "mape": "Miss as a share of the price. Best for comparing price levels.",
    "r2": "Does it explain the daily moves? 0 = no better than average.",
    "scored": "Real forecasts whose actual close is now known.",
    "reliability": "How often the close stayed inside the 90% range. Goal 90%.",
    "daily_move": "How far gold usually moves in a day. Sets the range width.",
    "yearly_pace": "The daily move scaled to a year. Best for comparing assets.",
    "vix": "US stock-market nervousness. Context only: it did not help.",
}


def card_row():
    """A row of cards. They keep equal heights (vertical_alignment="distribute" stretches them) and wrap when the screen is narrow."""
    return st.container(horizontal=True, gap="small", vertical_alignment="distribute")


def metric_card(label: str, value: str, chip: str | None = None, note: str | None = None, *, color: str = "off", help: str | None = None) -> None:
    """A bordered card: label, value, an optional chip (colour = how it compares) and the small note underneath."""
    with st.container(border=True, width="stretch", gap="xxsmall"):
        st.metric(label, value, chip, delta_color=color, delta_arrow="off", help=help)
        if note:
            st.caption(note)
