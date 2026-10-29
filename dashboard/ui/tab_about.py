"""About: what the project is, how the daily pipeline works, what else was tried (one line) and a short glossary."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .data import REPO_URL, feature_rows, horizons

STEPS = [
    ("Collect", "the latest gold price from Yahoo Finance, daily at 08:00 (Thailand time)"),
    ("Check", "automatic data-quality checks"),
    ("Forecast", "an LSTM reads the last 20 days (retrained weekly)"),
    ("Add the range", "a 90% range from the model's recent errors"),
    ("Publish and score", "saved to Amazon S3, scored when the real close arrives"),
]
GLOSSARY = [
    ("LSTM", "A neural network built for sequences. Here it reads the last 20 days and predicts the next day's price change."),
    ("Typical error", "The average gap between the forecast and the actual close, in dollars. Lower is better."),
    ("Baseline (tomorrow = today)", "The simplest forecast: repeat today's close. Also called the naive, no-change or persistence forecast. A model is only useful if it clearly beats it."),
    ("90% range and hit rate", "The range where the close should land 9 days out of 10. The hit rate is how often it really did; 90% is ideal."),
    ("Backtest", "Replaying the model on past days it had not seen, to see how it would have done."),
]


def _ideas_tried(res: dict) -> tuple[int, int, int]:
    """(ideas tested, clearly better than the baseline, clearly worse) over the other models, the extra inputs and the longer horizons."""
    verdicts = []
    for key in ("ridge", "hist_gb_reg"):
        m = res["models"][key]
        verdicts.append((m["dm_p_vs_naive"], m["mae_usd"] - m["naive_mae_usd"]))
    verdicts += [(r["dm_p"], r["mae"] - r["naive_mae"]) for r in feature_rows()[1:]]
    verdicts += [(r["models"]["lstm"]["dm_p_vs_naive"], r["models"]["lstm"]["mae_usd"] - r["models"]["lstm"]["naive_mae_usd"]) for r in horizons().values()]
    better = sum(p < 0.05 and d < 0 for p, d in verdicts)
    worse = sum(p < 0.05 and d > 0 for p, d in verdicts)
    return len(verdicts), better, worse


def render(res: dict | None) -> None:
    st.subheader("About this project")
    st.markdown("A senior project at Mae Fah Luang University: a daily forecast of **tomorrow's gold price (XAU/USD)** with an LSTM, a 90% range, and an honest test against a simple baseline.")
    st.write("")
    st.subheader("How the forecast is made", help="A fully automatic pipeline on AWS (Lambda and S3) runs every day, without anyone pressing a button.")
    with st.container(border=True):
        for i, (title, text) in enumerate(STEPS, 1):
            st.markdown(f":blue-badge[{i}] **{title}** · {text}")
    if res is not None:
        n, better, worse = _ideas_tried(res)
        st.write("")
        st.subheader("What else we tried", help="Other models (Ridge, gradient boosting), extra data (other markets, jobs-report days, Fed and inflation days, trader positioning) and forecasts 5 and 20 days ahead. "
                     "The numbers are in the project README on GitHub.")
        st.markdown(f"We tested {n} other ideas. " + ("None clearly beat the baseline" if not better else f"{better} clearly beat the baseline") + (f"; {worse} were clearly worse." if worse else "."))
    with st.expander("Glossary", icon=":material/menu_book:"):
        st.markdown("\n".join(f"- **{term}.** {text}" for term, text in GLOSSARY))
    st.divider()
    with st.container(horizontal=True, gap="small"):
        st.link_button("Source code", REPO_URL, icon=":material/code:")
        st.link_button("Full write-up (README)", f"{REPO_URL}/blob/main/README.md", icon=":material/description:")
    st.caption("Research project, not financial advice · Prices: Yahoo Finance (GC=F)")
