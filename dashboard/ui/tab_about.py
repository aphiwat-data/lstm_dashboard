"""About: what the project is, how the daily pipeline works, everything that was tried (the scoreboard), the method in plain words and a glossary."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import REPO_URL, feature_rows, horizons
from .theme import COLORS, ERROR_UNDER, finish

STEPS = [
    ("Collect", "the latest gold price from Yahoo Finance, daily at 08:00 (Thailand time)"),
    ("Check", "automatic data-quality checks"),
    ("Forecast", "an LSTM reads the last 20 days (retrained weekly)"),
    ("Add the range", "a 90% range from the model's recent errors"),
    ("Publish and score", "saved to Amazon S3, scored when the real close arrives"),
]
SHORT = {"+ macro and cross-asset prices": "LSTM + other markets", "+ NFP release days": "LSTM + jobs-report days", "+ FOMC and CPI release days": "LSTM + Fed and inflation days", "+ CFTC positioning": "LSTM + trader positioning"}
GLOSSARY = [
    ("LSTM", "A neural network built for sequences. Here it reads the last 20 days and predicts the next day's price change."),
    ("Typical error (MAE)", "The average gap between the forecast and the actual close, in dollars. Lower is better."),
    ("Baseline (“tomorrow = today”)", "The benchmark that simply repeats today's close, also called the naive or no-change forecast. A price forecast is only useful if it clearly beats this."),
    ("p-value", "How likely a difference this large would be to appear by luck alone. Below 0.05 counts as a clear difference; above it, the two are treated as equal."),
    ("R² (out-of-sample)", "The share of the daily price move the forecast explains. 0 means no better than a simple average; below 0 means worse."),
    ("Rank correlation (IC)", "Whether days with a higher forecast really had higher actual moves. 0 means no relationship, 1 a perfect match."),
    ("90% range and coverage", "Where the close should land 9 days out of 10. Coverage is how often it really did; 90% is ideal."),
    ("Walk-forward test", "Retrain on the past, test on the days that follow, then move forward and repeat, so the model never sees the days it is judged on."),
    ("EWMA", "Exponentially weighted moving average: recent values count more than old ones. Used to size the range."),
    ("VIX", "The “fear gauge” of the US stock market. Tested as an input; it did not help."),
]


def _intro() -> None:
    st.subheader("About this project")
    st.markdown("A senior project at Mae Fah Luang University: a daily forecast of **tomorrow's gold price (XAU/USD)** with an LSTM, a 90% range, and an honest test against simple baselines.")


def _pipeline() -> None:
    st.subheader("How the forecast is made", help="A fully automatic pipeline on AWS (Lambda and S3) runs every day, without anyone pressing a button.")
    with st.container(border=True):
        for i, (title, text) in enumerate(STEPS, 1):
            st.markdown(f":blue-badge[{i}] **{title}** · {text}")


def _experiments(res: dict) -> pd.DataFrame:
    models, rows = res["models"], []
    for key, label in (("lstm", "LSTM (this model)"), ("ridge", "Ridge (linear model)"), ("hist_gb_reg", "Gradient boosting (trees)")):
        m = models[key]
        rows.append(("Other models" if key != "lstm" else "Main model", label, label, m["mae_usd"], m["naive_mae_usd"], m["dm_p_vs_naive"], res["protocol"]["n_test"]))
    for r in feature_rows()[1:]:
        rows.append(("Extra inputs", SHORT.get(r["label"], r["label"]), "LSTM " + r["label"], r["mae"], r["naive_mae"], r["dm_p"], r["n"]))
    for h, r in horizons().items():
        m = r["models"]["lstm"]
        rows.append(("Longer horizon", f"LSTM, {h} days ahead", f"LSTM, price {h} trading days ahead", m["mae_usd"], m["naive_mae_usd"], m["dm_p_vs_naive"], r["protocol"]["n_test"]))
    d = pd.DataFrame(rows, columns=["group", "short", "label", "mae", "naive", "p", "n"])
    d["diff_pct"] = (d["mae"] / d["naive"] - 1) * 100
    d["verdict"] = ["No clear difference" if p >= 0.05 else "Better" if x < 0 else "Worse" for p, x in zip(d["p"], d["diff_pct"])]
    return d


def _scoreboard(res: dict | None) -> None:
    st.subheader("What we tried", help="Every idea was judged the same way: the typical price error against “tomorrow = today”, on the same test days. The same holds for the range: "
                 "five ways of drawing it, a GARCH model, the VIX and a network trained on the range itself all failed to beat a simple weighted average of recent errors.")
    if res is None:
        st.info("The evaluation results are not available in this deployment.")
        return
    d = _experiments(res)
    lim = max(5, math.ceil(d["diff_pct"].abs().max() * 1.6))
    colors = [COLORS["naive"] if v == "No clear difference" else COLORS["good"] if v == "Better" else ERROR_UNDER for v in d["verdict"]]
    fig = go.Figure(go.Bar(y=d["short"], x=d["diff_pct"], orientation="h", marker_color=colors, text=["0.0%" if round(x, 1) == 0 else f"{x:+.1f}%" for x in d["diff_pct"]], textposition="outside", cliponaxis=False,
                           customdata=d[["mae", "naive", "p"]], hovertemplate="%{y}<br>Typical error $%{customdata[0]:.2f} vs. $%{customdata[1]:.2f} for “tomorrow = today”<br>p = %{customdata[2]:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color="rgba(128,128,128,.7)", line_width=1)
    finish(fig, 60 + 34 * len(d), hover="closest")
    fig.update_xaxes(range=[-lim, lim], ticksuffix="%", title_text="Error vs. baseline (left = better)", title_font_size=12, showgrid=True, gridcolor="rgba(128,128,128,0.16)")
    fig.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    better, worse = (d["verdict"] == "Better").sum(), (d["verdict"] == "Worse").sum()
    st.caption("Grey: no real difference" + (" · Red: clearly worse" if worse else "") + (" · Green: clearly better" if better else "") + (". Nothing came out clearly better." if not better else "."))
    if d.loc[d["group"] != "Longer horizon", "n"].nunique() > 1:
        st.warning("These rows were evaluated on different test windows (see the table), so they are not strictly comparable.", icon=":material/warning:")
    with st.expander("See the numbers", icon=":material/table:"):
        table = pd.DataFrame({"Experiment": d["label"], "Typical error": d["mae"], "Baseline": d["naive"], "Difference": d["diff_pct"], "p-value": d["p"], "Verdict": d["verdict"], "Test days": d["n"]})
        st.dataframe(table, width="stretch", hide_index=True, column_config={
            "Typical error": st.column_config.NumberColumn(format="$%.2f"), "Baseline": st.column_config.NumberColumn(format="$%.2f"),
            "Difference": st.column_config.NumberColumn(format="%+.1f%%"), "p-value": st.column_config.NumberColumn(format="%.2f"), "Test days": st.column_config.NumberColumn(format="%d")})
        st.caption("Longer-horizon p-values are adjusted for overlapping forecasts.")


def _method(res: dict | None, reg: dict | None, px: pd.DataFrame | None) -> None:
    cov = f", so its {reg['interval']['lstm_ewma_gaussian']['coverage']:.1%} hit rate is honest" if reg else ""
    up = f" The model says “up” on {reg['models']['lstm']['says_up_share']:.0%} of test days, so its score mostly follows how often gold rose." if reg else ""
    if px is not None and "volume" in px:
        vol = pd.to_numeric(px["volume"], errors="coerce")
        bad = int((vol.isna() | (vol <= 0)).sum())
        volume = f"{bad / len(vol):.0%} of days show zero or missing volume in Yahoo Finance's data, so every input comes from the closing price alone."
    else:
        volume = "Volume is often zero or missing in Yahoo Finance's data, so every input comes from the closing price alone."
    with st.expander("The method in plain words", icon=":material/lightbulb:"):
        st.markdown(
            "**Why a change, not a price?** The first version predicted the price and broke when gold reached levels it had never seen (error about \\$640 against about \\$51 for “tomorrow = today”). "
            "Now the model predicts the day's percentage change and converts it: forecast = today's close × e^(predicted change).\n\n"
            "**Why is every test on unseen days?** The model is retrained every 7 trading days on the latest 365 days and judged only on the days that follow. Inputs use past data only; automated tests check it.\n\n"
            f"**How is the 90% range computed?** Forecast ± 1.645 × the typical recent miss (a weighted average of past errors, decay 0.94). It uses only errors known before each forecast{cov}.\n\n"
            f"**Why isn't volume an input?** {volume}\n\n"
            f"**Why is up/down only a side check?** A price forecast is judged by how far off it is, not by a yes/no call.{up}")


def _glossary() -> None:
    with st.expander("Glossary", icon=":material/menu_book:"):
        st.markdown("\n".join(f"- **{term}.** {text}" for term, text in GLOSSARY))


def _footer() -> None:
    st.divider()
    with st.container(horizontal=True, gap="small"):
        st.link_button("Source code", REPO_URL, icon=":material/code:")
        st.link_button("Full write-up (README)", f"{REPO_URL}/blob/main/README.md", icon=":material/description:")
    st.caption("Research project, not financial advice · Prices: Yahoo Finance (GC=F)")


def render(res: dict | None, reg: dict | None, px: pd.DataFrame | None) -> None:
    _intro()
    st.write("")
    _pipeline()
    st.write("")
    _scoreboard(res)
    st.write("")
    _method(res, reg, px)
    _glossary()
    _footer()
