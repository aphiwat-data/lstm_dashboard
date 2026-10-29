"""How reliable? The verdict first, then two numbers of evidence in plain words."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .logic import PHRASE, edge_color, pct_miss, price_verdict
from .theme import COLORS, finish, signed_usd, usd


def _verdict(verdict: str, coverage: float | None) -> None:
    range_note = " The range, however, can be trusted." if coverage is not None and abs(coverage - 0.9) <= 0.02 else ""
    if verdict == "better":
        st.success("**Better than the baseline.** Its typical error is clearly lower than just repeating today's price.", icon=":material/check_circle:")
    elif verdict == "worse":
        st.error("**Worse than the baseline.** Its typical error is clearly higher than just repeating today's price.", icon=":material/cancel:")
    else:
        st.info("**Not better than the baseline.** It does about as well as repeating today's price, so read the forecast as *about today's close*." + range_note, icon=":material/balance:")


def _row(label: str, value: str, chip: str, color: str, line: str, help: str) -> None:
    """One line of evidence: the number on the left, one short line on the right (it wraps under the number on a narrow screen); the detail sits in the tooltip."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.metric(label, value, chip, delta_color=color, delta_arrow="off", width=240, help=help)
        with st.container():
            st.markdown(line)


def _evidence(res: dict, preds: pd.DataFrame, reg: dict | None, verdict: str) -> None:
    lstm = res["models"]["lstm"]
    diff = lstm["mae_usd"] - lstm["naive_mae_usd"]
    st.subheader("The evidence", help=f"Based on {res['protocol']['n_test']} test days ({res['protocol']['test_start']} to {res['protocol']['test_end']}); small differences are within noise.")
    with st.container(border=True):
        _row("Typical error", usd(lstm["mae_usd"]), f"{signed_usd(diff)} vs. baseline", edge_color(verdict, diff),
             f"The baseline (“tomorrow = today”) misses by {usd(lstm['naive_mae_usd'])}.",
             f"The baseline just repeats today's price; it misses by {pct_miss(preds)[1]:.2f}% of the price. The model {PHRASE[verdict]} it (p = {lstm['dm_p_vs_naive']:.2f}; below 0.05 counts as a clear difference).")
        if reg:
            iv = reg["interval"]
            e, nc = iv["lstm_ewma_gaussian"], iv["no_change_ewma_gaussian"]
            st.divider()
            _row("Range hit rate", f"{e['coverage']:.1%}", "Target 90%", "green" if abs(e["coverage"] - 0.9) <= 0.02 else "orange",
                 f"The range held on {round(e['coverage'] * e['n'])} of {e['n']} test days.",
                 f"Ideal: 90%. Its width, {usd(e['mean_width_usd'], 0)}, is the same as the baseline's range ({usd(nc['mean_width_usd'], 0)}): the model does not narrow the uncertainty.")


def _error_chart(res: dict) -> None:
    lstm = res["models"]["lstm"]
    names = ["Baseline (tomorrow = today)", "This model (LSTM)"]
    vals = [lstm["naive_mae_usd"], lstm["mae_usd"]]
    fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker=dict(color=[COLORS["naive"], COLORS["model"]], pattern_shape=["/", ""], pattern_fgcolor="rgba(255,255,255,0.55)"),
                           text=[f"${v:,.2f}" for v in vals], textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:$,.2f}<extra></extra>"))
    finish(fig, 150, hover="closest")
    fig.update_layout(bargap=0.45)
    fig.update_xaxes(range=[0, max(vals) * 1.25], tickprefix="$", showgrid=True, gridcolor="rgba(128,128,128,0.16)")
    fig.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    st.markdown("**Typical daily error** (lower is better)", help="How far the forecast misses the next day's close on average. A model has to land clearly below the baseline to add value.")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render(res: dict | None, preds: pd.DataFrame | None, reg: dict | None) -> None:
    if res is None or preds is None:
        st.info("The evaluation results are not available in this deployment.")
        return
    verdict = price_verdict(res["models"]["lstm"])
    st.subheader("Can this forecast be relied on?")
    _verdict(verdict, reg["interval"]["lstm_ewma_gaussian"]["coverage"] if reg else None)
    _evidence(res, preds, reg, verdict)
    st.write("")
    _error_chart(res)
