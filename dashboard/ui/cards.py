"""Metric cards: a bordered card with the label, the number and, where it helps, one chip. Definitions and comparisons sit in the tooltip."""
from __future__ import annotations

import streamlit as st


def card_row():
    """A row of cards. They keep equal heights (vertical_alignment="distribute" stretches them) and wrap when the screen is narrow."""
    return st.container(horizontal=True, gap="small", vertical_alignment="distribute")


def metric_card(label: str, value: str, chip: str | None = None, *, color: str = "off", help: str | None = None) -> None:
    """A bordered card: label, value and an optional chip (colour = how it compares)."""
    with st.container(border=True, width="stretch", gap="xxsmall"):
        st.metric(label, value, chip, delta_color=color, delta_arrow="off", help=help)
