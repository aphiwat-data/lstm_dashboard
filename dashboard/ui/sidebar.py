"""Sidebar: who made this (always visible, as requested). Kept short enough to fit one screen without scrolling; the source link lives on the About tab."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "assets"
LOGOS = [(ASSETS / "mfu_logo.png", 44), (ASSETS / "adt_logo.png", 62)]  # (file, display width px): the crest is tall, the ADT mark round
TEAM = [("Mr. Achira Lueablae", "6631501120"), ("Mr. Aphiwat Chioewvijit", "6631501121")]
ADVISOR = "Asst. Prof. Pattaramon Vuttipittayamongkol"
COMMITTEE = ["Asst. Prof. Khwunta Kirimasthong", "Asst. Prof. Surapong Uttama"]


def _credit(label: str, lines: list[str]) -> None:
    """A small label over its names, as one element so the sidebar does not add a gap after every line. Names are small so they fit the narrower sidebar of newer Streamlit."""
    st.markdown(f":gray[:small[{label}]]  \n" + "  \n".join(lines))


def render() -> None:
    with st.sidebar, st.container(gap="xxsmall"):
        with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="medium"):
            for logo, width in LOGOS:
                if logo.exists():
                    with st.container(width=width):  # image(width=...) would downscale the file itself and look soft on high-DPI screens
                        st.image(str(logo), width="stretch")
        st.markdown("**Next-Day Gold Price  \nForecasting with LSTM**  \n:gray[:small[B.Eng. Computer Engineering  \nSchool of Applied Digital Technology  \nMae Fah Luang University · 2026]]", text_alignment="center")
        st.divider()
        _credit("DEVELOPED BY", [f":small[**{name}**  \n:gray[{sid}]]" for name, sid in TEAM])
        _credit("ADVISOR", [f":small[**{ADVISOR}**]"])
        _credit("COMMITTEE", [f":small[**{name}**]" for name in COMMITTEE])
