"""Sidebar: who made this (always visible, as requested), then the status of the data and a link to the code."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from .data import REPO_URL
from .header import freshness

ASSETS = Path(__file__).resolve().parents[1] / "assets"
LOGOS = [(ASSETS / "mfu_logo.png", 66), (ASSETS / "adt_logo.png", 90)]  # (file, display width px): the crest is tall, the ADT mark round
TEAM = [("Mr. Achira Lueablae", "6631501120"), ("Mr. Aphiwat Chioewvijit", "6631501121")]
ADVISOR = "Asst. Prof. Pattaramon Vuttipittayamongkol"
COMMITTEE = ["Asst. Prof. Khwunta Kirimasthong", "Asst. Prof. Surapong Uttama"]


def render(cur: dict | None) -> None:
    with st.sidebar:
        with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="medium"):
            for logo, width in LOGOS:
                if logo.exists():
                    with st.container(width=width):  # image(width=...) would downscale the file itself and look soft on high-DPI screens
                        st.image(str(logo), width="stretch")
        st.markdown("**Next-Day Gold Price  \nForecasting with LSTM**", text_alignment="center")
        st.caption("Bachelor of Engineering, Computer Engineering  \nSchool of Applied Digital Technology  \nMae Fah Luang University · 2026", text_alignment="center")
        st.divider()
        st.caption("DEVELOPED BY")
        st.markdown("\n\n".join(f"**{name}**  \n:gray[Student ID {sid}]" for name, sid in TEAM))
        st.caption("ADVISOR")
        st.markdown(f"**{ADVISOR}**")
        st.caption("COMMITTEE")
        st.markdown("\n\n".join(f"**{name}**" for name in COMMITTEE))
        st.divider()
        if cur is not None:
            label, color, icon = freshness(cur)
            st.caption("STATUS")
            st.badge(label, icon=icon, color=color)
            trained = cur.get("model_trained_through")
            st.caption(f"Data through **{cur['as_of_date']}**" + (f"  \nModel trained through **{trained}**" if trained else ""))
        st.link_button("View the source code", REPO_URL, icon=":material/code:", width="stretch")
