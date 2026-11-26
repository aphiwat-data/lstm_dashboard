"""How reliable? The evidence in plain words, then two comparison charts: average miss by model and range reliability by method."""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import MODELS
from .logic import edge_color, price_verdict
from .theme import COLORS, chart_title, finish, highlight, signed_usd, usd


def _row(label: str, value: str, chip: str, color: str, line: str) -> None:
    """One line of evidence: the number on the left, one short line on the right (it wraps under the number on a phone)."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.metric(label, value, chip, delta_color=color, delta_arrow="off", width=240)
        with st.container():
            st.markdown(line)


def _evidence(res: dict, reg: dict | None, verdict: str, key: str) -> None:
    lstm = res["models"][key]
    st.subheader("The evidence")
    with st.container(border=True):
        _row("MAE · average miss", usd(lstm["mae_usd"]), f"{signed_usd(lstm['mae_usd'] - lstm['naive_mae_usd'])} vs. baseline", edge_color(verdict, lstm["mae_usd"] - lstm["naive_mae_usd"]),
             "On a typical day the forecast was about this many dollars away from the real close.")
        if reg:
            m, iv = reg["models"][key], reg["interval"]
            signal = m["r2_os_vs_rolling_mean"] > 0 and m["ic_p"] < 0.05
            e = iv[f"{key}_ewma_gaussian"]
            st.divider()
            _row("R² · predicts moves?", f"{m['r2_os_vs_rolling_mean']:+.3f}", "Some predictive power" if signal else "No predictive power", "green" if signal else "orange",
                 "A test score for catching the daily ups and downs: 1 means every move was predicted, 0 means none was.")
            st.divider()
            _row("Range reliability", f"{e['coverage']:.1%}", "goal 90%", "green" if abs(e["coverage"] - 0.9) <= 0.02 else "orange",
                 "How often the real close stayed inside the range. The aim is 9 days out of 10.")


def _mae_chart(res: dict, key: str) -> None:
    models = res["models"]
    naive = models[key]["naive_mae_usd"]
    name, short = MODELS[key]["name"], MODELS[key]["short"]
    # (name for the hover, label under the bar, value, colour, hatch, is this model): this model first, its name and number in bold blue so they cannot be missed
    entries = [(f"{name} (this model)", f"{highlight(short, 14)}<br>{highlight('(this model)')}", models[key]["mae_usd"], COLORS["model"], "", True),
               ("Simple guess (repeat last close)", "Simple guess<br>(repeat last close)", naive, COLORS["naive"], "/", False)]
    if key != "lstm":  # the page's model is not the LSTM: show the LSTM and the other price-only models as the comparison
        entries.append(("LSTM (price only)", "LSTM<br>(price only)", models["lstm"]["mae_usd"], COLORS["lstm"], "", False))
    entries += [("Ridge (price only)" if key != "lstm" else "Ridge (linear model)", "Ridge<br>(price only)" if key != "lstm" else "Ridge<br>(linear model)", models["ridge"]["mae_usd"], COLORS["ridge"], "", False),
                ("Gradient boosting (price only)" if key != "lstm" else "Gradient boosting (tree model)", "Gradient boosting<br>(price only)" if key != "lstm" else "Gradient boosting<br>(tree model)",
                 models["hist_gb_reg"]["mae_usd"], COLORS["hgb"], "", False)]
    vals = [e[2] for e in entries]
    fig = go.Figure(go.Bar(x=[f"{label}<br>" + (highlight(f"${v:,.2f}", 14) if mine else f"<b>${v:,.2f}</b>") for _, label, v, _, _, mine in entries], customdata=[e[0] for e in entries], y=vals,
                           marker=dict(color=[e[3] for e in entries], pattern_shape=[e[4] for e in entries], pattern_fgcolor="rgba(255,255,255,0.55)"),
                           hovertemplate="%{customdata}<br>%{y:$,.2f}<extra></extra>"))
    fig.add_hline(y=naive, layer="below", line_dash="dash", line_color=COLORS["naive"], line_width=1.2)
    finish(fig, 300, hover="closest")
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(range=[0, max(vals) * 1.2], tickprefix="$")
    candidates = [(name, key)] + [(n, k) for n, k in (("LSTM", "lstm"), ("Ridge", "ridge"), ("Gradient boosting", "hist_gb_reg")) if k != key]
    better = [n for n, k in candidates if models[k]["dm_p_vs_naive"] < 0.05 and models[k]["mae_usd"] < naive]
    chart_title("Average miss (MAE) by model", help="Lower is better. A model has to land clearly below the dashed line to add value. " + (f"{', '.join(better)} {'does' if len(better) == 1 else 'do'}." if better else "None does."))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _plain(label: str) -> str:
    """A chart label without its markup (for the hover text)."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", label)).strip()


def _range_chart(compare: list | None, vix: list | None, key: str) -> None:
    if not compare:
        return
    wanted = [("EWMA Gaussian, lambda 0.94 (current)", f"{highlight(MODELS[key]['short'], 14)}<br>{highlight('(this model)')}"), ("Empirical quantiles, all past errors", "All past errors"), ("GARCH(1,1), Gaussian", "GARCH<br>(volatility model)"), ("LSTM quantile (pinball loss)", "LSTM learns<br>the range")]
    by = {r["method"]: r for r in compare}
    rows = [(label, by[key]["coverage"]) for key, label in wanted if key in by]
    v = next((r for r in vix if r["method"].startswith("EWMA + VIX (previous day)")), None) if vix else None
    if v:
        rows.append(("LSTM + VIX", v["coverage"]))
    colors = [COLORS["model"] if i == 0 else COLORS["naive"] for i in range(len(rows))]
    fig = go.Figure(go.Bar(x=[f"{r[0]}<br>" + (highlight(f"{r[1]:.1%}", 14) if i == 0 else f"<b>{r[1]:.1%}</b>") for i, r in enumerate(rows)], customdata=[_plain(r[0]) for r in rows], y=[r[1] * 100 for r in rows], marker_color=colors,
                           hovertemplate="%{customdata}<br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=90, layer="below", line_dash="dash", line_color=COLORS["actual"], line_width=1.2)
    finish(fig, 300, hover="closest")
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(range=[50, 100], ticksuffix="%")
    used, others = rows[0][1], [r[1] for r in rows[1:]]
    chart_title("Range reliability by method", help=f"How often the close landed inside the 90% range. Dashed line: the 90% goal. We compared {len(rows)} ways to draw the range." + (" The first one, this model's range (sized from its recent errors), landed closest to 90%." if all(abs(used - 0.9) <= abs(o - 0.9) for o in others) else "")
                + (" The others were too narrow." if all(o < used for o in others) else "") + (" “LSTM + VIX” also uses the VIX to size the range." if v else ""))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render(res: dict | None, preds: pd.DataFrame | None, reg: dict | None, compare: list | None, vix: list | None, key: str) -> None:
    if res is None or preds is None:
        st.info("The evaluation results are not available in this deployment.")
        return
    _evidence(res, reg, price_verdict(res["models"][key]), key)
    st.write("")
    _mae_chart(res, key)
    st.write("")
    _range_chart(compare, vix, key)
