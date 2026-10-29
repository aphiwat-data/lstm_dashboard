"""About: what the project is, how the daily pipeline works, everything that was tried (the scoreboard), the method in plain words and a glossary."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import REPO_URL, feature_rows, horizons
from .logic import PHRASE, price_verdict
from .theme import COLORS, ERROR_UNDER, finish, usd

STEPS = [
    ("Collect", "Every day at 08:00 (Thailand time) a scheduled AWS Lambda function downloads the latest settled gold futures bar from Yahoo Finance."),
    ("Check", "The new bar goes through data-quality checks. If something looks wrong, no forecast is published that day."),
    ("Forecast", "An LSTM neural network reads the last 20 days (returns, volatility, momentum, moving averages) and predicts tomorrow's price change. Its weights are retrained every week."),
    ("Add the range", "A 90% range is drawn around the forecast from the model's own recent errors, so it widens when the market gets jumpy."),
    ("Publish and score", "The forecast is saved to Amazon S3 and shown on this page. When the real close arrives it is scored, and the result joins the live record."),
]
SHORT = {"+ macro and cross-asset prices": "+ macro prices", "+ NFP release days": "+ NFP days", "+ FOMC and CPI release days": "+ FOMC and CPI days", "+ CFTC positioning": "+ positioning (COT)"}
GLOSSARY = [
    ("LSTM", "A neural network built for sequences. Here it reads the last 20 days and predicts the next day's price change."),
    ("Typical error (MAE)", "The average gap between the forecast and the actual close, in dollars. Lower is better."),
    ("“Tomorrow = today”", "The benchmark that simply repeats today's close. A price forecast is only useful if it clearly beats this."),
    ("p-value", "How likely a difference this large would be to appear by luck alone. Below 0.05 counts as a clear difference; above it, the two are treated as equal."),
    ("R² (out-of-sample)", "The share of the daily price move the forecast explains. 0 means no better than a simple average; below 0 means worse."),
    ("Rank correlation (IC)", "Whether days with a higher forecast really had higher actual moves. 0 means no relationship, 1 a perfect match."),
    ("90% range and coverage", "Where the close should land 9 days out of 10. Coverage is how often it really did; 90% is ideal."),
    ("Walk-forward test", "Retrain on the past, test on the days that follow, then move forward and repeat, so the model never sees the days it is judged on."),
    ("EWMA", "Exponentially weighted moving average: recent values count more than old ones. Used to size the range."),
    ("VIX", "The “fear gauge” of the US stock market. Tested as an input; it did not help."),
]


def _intro(res: dict | None, reg: dict | None) -> None:
    st.subheader("About this project")
    st.markdown("A senior project in Computer Engineering at Mae Fah Luang University. It forecasts the **next trading day's closing price of gold (XAU/USD)** with an LSTM neural network, "
                "publishes the forecast with a 90% range every day, and then checks, as strictly as it can, how good that forecast really is.")
    if res is None:
        return
    lstm = res["models"]["lstm"]
    text = (f"**The honest result.** Over {res['protocol']['n_test']} test days the model's typical error was {usd(lstm['mae_usd'])}, against {usd(lstm['naive_mae_usd'])} for a “tomorrow = today” guess. "
            f"Statistically, the model {PHRASE[price_verdict(lstm)]} that simple guess.")
    if reg:
        e = reg["interval"]["lstm_ewma_gaussian"]
        text += f" What does hold up is the range: the actual close landed inside it on {e['coverage']:.1%} of days (target 90%), so it is a trustworthy measure of uncertainty."
    st.info(text, icon=":material/flag:")


def _pipeline() -> None:
    st.subheader("How the forecast is made")
    st.caption("A fully automatic pipeline on AWS runs every day, without anyone pressing a button.")
    with st.container(border=True):
        for i, (title, text) in enumerate(STEPS, 1):
            st.markdown(f":blue-badge[{i}] **{title}.** {text}")


def _experiments(res: dict) -> pd.DataFrame:
    models, rows = res["models"], []
    for key, label in (("lstm", "LSTM (this model)"), ("ridge", "Ridge regression"), ("hist_gb_reg", "Gradient boosting")):
        m = models[key]
        rows.append(("Other models" if key != "lstm" else "Main model", label, label, m["mae_usd"], m["naive_mae_usd"], m["dm_p_vs_naive"], res["protocol"]["n_test"]))
    for r in feature_rows()[1:]:
        rows.append(("Extra inputs", SHORT.get(r["label"], r["label"]), "LSTM " + r["label"], r["mae"], r["naive_mae"], r["dm_p"], r["n"]))
    for h, r in horizons().items():
        m = r["models"]["lstm"]
        rows.append(("Longer horizon", f"{h}-day forecast", f"LSTM, price {h} trading days ahead", m["mae_usd"], m["naive_mae_usd"], m["dm_p_vs_naive"], r["protocol"]["n_test"]))
    d = pd.DataFrame(rows, columns=["group", "short", "label", "mae", "naive", "p", "n"])
    d["diff_pct"] = (d["mae"] / d["naive"] - 1) * 100
    d["verdict"] = ["No clear difference" if p >= 0.05 else "Better" if x < 0 else "Worse" for p, x in zip(d["p"], d["diff_pct"])]
    return d


def _scoreboard(res: dict | None) -> None:
    st.subheader("What we tried")
    st.caption("Every idea was judged the same way: the typical price error against “tomorrow = today”, on the same test days.")
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
    fig.update_xaxes(range=[-lim, lim], ticksuffix="%", title_text="Error vs. “tomorrow = today” (left = better)", title_font_size=12, showgrid=True, gridcolor="rgba(128,128,128,0.16)")
    fig.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    better, worse = (d["verdict"] == "Better").sum(), (d["verdict"] == "Worse").sum()
    st.caption("Grey bars are differences too small to tell apart from luck." + (" Red bars are clearly worse than “tomorrow = today”." if worse else "") + (" Green bars are clearly better." if better else "")
               + (" Nothing came out clearly better." if not better else "")
               + " The same holds for the range: five ways of drawing it, a GARCH model, the VIX and a network trained on the range itself all failed to beat a simple weighted average of recent errors.")
    if d.loc[d["group"] != "Longer horizon", "n"].nunique() > 1:
        st.warning("These rows were evaluated on different test windows (see the table), so they are not strictly comparable.", icon=":material/warning:")
    with st.expander("See the numbers", icon=":material/table:"):
        table = pd.DataFrame({"Experiment": d["label"], "Typical error": d["mae"], "“Tomorrow = today”": d["naive"], "Difference": d["diff_pct"], "p-value": d["p"], "Verdict": d["verdict"], "Test days": d["n"]})
        st.dataframe(table, width="stretch", hide_index=True, column_config={
            "Typical error": st.column_config.NumberColumn(format="$%.2f"), "“Tomorrow = today”": st.column_config.NumberColumn(format="$%.2f"),
            "Difference": st.column_config.NumberColumn(format="%+.1f%%"), "p-value": st.column_config.NumberColumn(format="%.2f"), "Test days": st.column_config.NumberColumn(format="%d")})
        st.caption("Longer-horizon rows use overlapping forecasts, so their p-values are adjusted for that. Details in the README.")


def _method(res: dict | None, reg: dict | None, px: pd.DataFrame | None) -> None:
    st.subheader("The method in plain words")
    with st.expander("Why does the model predict a change instead of a price?"):
        st.markdown("The first version predicted the price directly. When gold climbed to levels it had never seen in training, its error exploded (about \\$640, against about \\$51 for “tomorrow = today”), "
                    "because a neural network cannot extrapolate beyond the range it was trained on. The current model predicts the day's percentage change from scale-free inputs "
                    "(returns, ratios and oscillators) and turns it into a price: **forecast = today's close × e^(predicted change)**.")
    with st.expander("Why is every test on days the model has not seen?"):
        st.markdown("Every 7 trading days the model is retrained on the latest 365 days, then judged on the days that follow, exactly as it would be used in real life. "
                    "The scalers are fitted inside each training window and every input uses past data only; automated tests check that nothing from the future leaks in.")
    with st.expander("How is the 90% range computed?"):
        cov = f" Its hit rate of {reg['interval']['lstm_ewma_gaussian']['coverage']:.1%} is therefore an honest, out-of-sample number." if reg else ""
        st.markdown("From the model's own past errors: an exponentially weighted average of recent squared errors (decay 0.94, fixed before testing) gives a typical size for tomorrow's miss, "
                    "and the range is the forecast ± 1.645 times that size. It uses only errors known before each forecast." + cov)
    with st.expander("Why isn't trading volume an input?"):
        if px is not None and "volume" in px:
            vol = pd.to_numeric(px["volume"], errors="coerce")
            bad = int((vol.isna() | (vol <= 0)).sum())
            st.markdown(f"**{bad / len(vol):.0%} of days ({bad:,} of {len(vol):,}) show zero or missing volume** in Yahoo Finance's gold futures data, a known gap in the source. "
                        "So every input is computed from the closing price alone.")
        else:
            st.markdown("Volume is often zero or missing in Yahoo Finance's gold futures data, so every input is computed from the closing price alone.")
    with st.expander("Why is up/down accuracy only a side check?"):
        up = f" It says “up” on {reg['models']['lstm']['says_up_share']:.0%} of test days, so its score mostly follows how often gold rose." if reg else ""
        st.markdown("A price forecast is judged by how far off it is, not by a yes/no call, so the price error comes first. Direction is still reported as a labelled secondary check, "
                    "because the original proposal phrased its target that way and because it shows how the model behaves." + up)


def _glossary() -> None:
    with st.expander("Glossary", icon=":material/menu_book:"):
        st.markdown("\n".join(f"- **{term}.** {text}" for term, text in GLOSSARY))


def _footer() -> None:
    st.divider()
    with st.container(horizontal=True, gap="small"):
        st.link_button("Source code", REPO_URL, icon=":material/code:")
        st.link_button("Full write-up (README)", f"{REPO_URL}/blob/main/README.md", icon=":material/description:")
    st.caption("A research and education project. Nothing here is financial advice. Prices: COMEX gold futures (GC=F) via Yahoo Finance.")


def render(res: dict | None, reg: dict | None, px: pd.DataFrame | None) -> None:
    _intro(res, reg)
    st.write("")
    _pipeline()
    st.write("")
    _scoreboard(res)
    st.write("")
    _method(res, reg, px)
    _glossary()
    _footer()
