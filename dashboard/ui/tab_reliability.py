"""How reliable? The verdict first, then the evidence in plain words; statistics are one click deeper."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .logic import PHRASE, direction_verdict, edge_color, pct_miss, price_verdict
from .theme import COLORS, finish, signed_usd, usd

MODEL_LABELS = {"drift": "Rolling average", "ridge": "Ridge", "hist_gb_reg": "Gradient boosting", "hist_gb_clf": "Gradient boosting (up/down)", "lstm": "LSTM", "momentum_rule": "Momentum rule"}


def _verdict(lstm: dict, verdict: str) -> None:
    if verdict == "better":
        st.success("**There is a measurable edge.** The model's typical error is clearly lower than the “tomorrow = today” forecast. Check the details before relying on it.", icon=":material/check_circle:")
    elif verdict == "worse":
        st.error("**The model is worse than the simple baseline.** Its typical error is clearly higher than a forecast that repeats today's price.", icon=":material/cancel:")
    else:
        st.info("**No reliable edge.** The model's price forecast is level with the simple forecast “tomorrow = today”, so read it as *about today's close, within the range shown*. "
                "The range itself is trustworthy: it has held about as often as promised.", icon=":material/balance:")


def _row(label: str, value: str, chip: str, color: str, lines: list[str]) -> None:
    """One line of evidence: the number on the left, what it means on the right (wraps under the number on a phone)."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.metric(label, value, chip, delta_color=color, delta_arrow="off", width=240)
        with st.container():
            for line in lines:
                st.markdown(line)


def _evidence(res: dict, preds: pd.DataFrame, reg: dict | None, verdict: str) -> None:
    lstm = res["models"]["lstm"]
    pct_n = pct_miss(preds)[1]
    st.subheader("The evidence")
    with st.container(border=True):
        _row("Price error", usd(lstm["mae_usd"]), f"{signed_usd(lstm['mae_usd'] - lstm['naive_mae_usd'])} vs. “tomorrow = today”", edge_color(verdict, lstm["mae_usd"] - lstm["naive_mae_usd"]),
             [f"A forecast that simply repeats today's price misses by {usd(lstm['naive_mae_usd'])} ({pct_n:.2f}% of the price).",
              f"The model {PHRASE[verdict]} it (p = {lstm['dm_p_vs_naive']:.2f}; below 0.05 counts as a clear difference)."])
        if reg:
            m, iv = reg["models"]["lstm"], reg["interval"]
            signal = m["r2_os_vs_rolling_mean"] > 0 and m["ic_p"] < 0.05
            e, nc = iv["lstm_ewma_gaussian"], iv["no_change_ewma_gaussian"]
            st.divider()
            _row("Explains the daily move?", f"{m['r2_os_vs_rolling_mean']:+.3f}", "Some signal" if signal else "No signal", "green" if signal else "orange",
                 ["Out-of-sample R²: 0 means no better than a simple average, below 0 is worse.", f"Rank correlation with the actual moves: {m['ic_spearman']:+.3f} (p = {m['ic_p']:.2f})."])
            st.divider()
            _row("Range accuracy", f"{e['coverage']:.1%}", "Target 90%", "green" if abs(e["coverage"] - 0.9) <= 0.02 else "orange",
                 [f"The 90% range held on {e['n']} test days (ideal: 90%).",
                  f"Its width, {usd(e['mean_width_usd'], 0)}, is the same as a no-change range ({usd(nc['mean_width_usd'], 0)}): the model does not narrow the uncertainty."])
    st.caption(f"Based on {res['protocol']['n_test']} test days ({res['protocol']['test_start']} to {res['protocol']['test_end']}); small differences are within noise.")


def _mae_chart(res: dict) -> None:
    models = res["models"]
    naive = models["lstm"]["naive_mae_usd"]
    entries = [("“Tomorrow = today”", naive, COLORS["naive"]), ("Ridge", models["ridge"]["mae_usd"], COLORS["ridge"]),
               ("Gradient boosting", models["hist_gb_reg"]["mae_usd"], COLORS["hgb"]), ("LSTM", models["lstm"]["mae_usd"], COLORS["model"])]
    vals = [v for _, v, _ in entries]
    fig = go.Figure(go.Bar(x=[n for n, _, _ in entries], y=vals, marker=dict(color=[c for *_, c in entries], pattern_shape=["/", "", "", ""], pattern_fgcolor="rgba(255,255,255,0.55)"),
                           text=[f"${v:,.2f}" for v in vals], textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>%{y:$,.2f}<extra></extra>"))
    fig.add_hline(y=naive, line_dash="dash", line_color=COLORS["naive"], line_width=1.2)
    finish(fig, 300, hover="closest")
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(range=[0, max(vals) * 1.2], tickprefix="$")
    st.markdown("**Typical daily error by model** (lower is better)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    better = [name for name, key in (("Ridge", "ridge"), ("Gradient boosting", "hist_gb_reg"), ("LSTM", "lstm")) if models[key]["dm_p_vs_naive"] < 0.05 and models[key]["mae_usd"] < naive]
    st.caption("A model has to land clearly below the dashed line to add value. " + (f"{', '.join(better)} {'does' if len(better) == 1 else 'do'}." if better else "None does."))


def _range_chart(compare: list | None, vix: list | None) -> None:
    if not compare:
        return
    wanted = [("EWMA Gaussian, lambda 0.94 (current)", "Used here"), ("Empirical quantiles, all past errors", "Past errors"), ("GARCH(1,1), Gaussian", "GARCH"), ("LSTM quantile (pinball loss)", "LSTM range")]
    by = {r["method"]: r for r in compare}
    rows = [(label, by[key]["coverage"]) for key, label in wanted if key in by]
    if vix:
        v = next((r for r in vix if r["method"].startswith("EWMA + VIX (previous day)")), None)
        if v:
            rows.append(("With VIX", v["coverage"]))
    colors = [COLORS["model"] if i == 0 else COLORS["naive"] for i in range(len(rows))]
    fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] * 100 for r in rows], marker_color=colors, text=[f"{r[1]:.1%}" for r in rows], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=90, line_dash="dash", line_color=COLORS["actual"], line_width=1.2)
    finish(fig, 300, hover="closest")
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(range=[50, 100], ticksuffix="%")
    st.markdown("**How often each range held** (dashed line: the 90% target)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    used, others = rows[0][1], [r[1] for r in rows[1:]]
    st.caption(f"We compared {len(rows)} ways to draw the 90% range." + (" The one used here landed closest to 90%." if all(abs(used - 0.9) <= abs(o - 0.9) for o in others) else "")
               + (" The others were too narrow." if all(o < used for o in others) else ""))


def _direction(res: dict) -> None:
    models = res["models"]
    lstm = models["lstm"]
    hd = lstm["direction_from_head"]
    up = models["drift"]["direction_from_sign"]["always_up_acc"]
    v = direction_verdict(lstm, up)
    with st.expander("Secondary check: calling the direction (up or down)", icon=":material/swap_vert:"):
        st.caption("Calling up or down treats the forecast as a yes/no question, which is not how a price forecast is judged, so it is only a side check. It also shows how the model behaves.")
        st.markdown(f"The model called the direction correctly on **{hd['acc']:.1%}** of days (95% interval {hd['ci95'][0]:.1%} to {hd['ci95'][1]:.1%}). "
                    f"Gold rose on {up:.1%} of test days, so always saying “up” already scores {up:.1%}; the model {PHRASE[v]} that (p = {hd['p_vs_always_up']:.2f}). "
                    "The model says “up” on about nine days in ten, so its score mostly follows how often gold rose.")
        entries = [("Always up", models["drift"]["direction_from_sign"]), ("Momentum rule", models["momentum_rule"]["direction_from_head"]), ("Ridge", models["ridge"]["direction_from_sign"]),
                   ("Gradient boosting", models["hist_gb_clf"]["direction_from_head"]), ("LSTM", models["lstm"]["direction_from_head"])]
        acc = [d["acc"] * 100 for _, d in entries]
        fig = go.Figure(go.Bar(x=[n for n, _ in entries], y=acc, marker_color=[COLORS["naive"], COLORS["good"], COLORS["ridge"], COLORS["hgb"], COLORS["model"]],
                               text=[f"{a:.1f}%" for a in acc], textposition="outside", cliponaxis=False,
                               error_y=dict(type="data", symmetric=False, array=[(d["ci95"][1] - d["acc"]) * 100 for _, d in entries], arrayminus=[(d["acc"] - d["ci95"][0]) * 100 for _, d in entries], thickness=1.2, width=4)))
        fig.add_hline(y=up * 100, line_dash="dash", line_color=COLORS["naive"], annotation_text=f"Always up {up:.1%}", annotation_position="top left", annotation_font_size=12)
        finish(fig, 320, hover="closest")
        fig.update_yaxes(range=[35, 75], ticksuffix="%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Bars show 95% confidence intervals.")


def _details(res: dict, reg: dict | None) -> None:
    with st.expander("All the statistics, model by model", icon=":material/table:"):
        rows = {}
        for key, r in res["models"].items():
            d = r.get("direction_from_head") or r["direction_from_sign"]
            m = (reg["models"].get(key) if reg else None) or {}
            rows[MODEL_LABELS.get(key, key)] = {"Typical error ($)": r.get("mae_usd"), "Typical error (%)": m.get("mape_pct"), "RMSE ($)": r.get("rmse_usd"),
                                                "vs. no change (p)": r.get("dm_p_vs_naive"), "R² out-of-sample": m.get("r2_os_vs_rolling_mean"), "Rank correlation": r.get("ic_spearman"),
                                                "Direction right (%)": d["acc"] * 100, "vs. always up (p)": d["p_vs_always_up"], "AUC": (r.get("direction_from_head") or {}).get("auc")}
        table = pd.DataFrame(rows).T.astype(float)
        st.dataframe(table, width="stretch", column_config={
            "Typical error ($)": st.column_config.NumberColumn(format="$%.2f"), "Typical error (%)": st.column_config.NumberColumn(format="%.3f%%"), "RMSE ($)": st.column_config.NumberColumn(format="$%.2f"),
            "vs. no change (p)": st.column_config.NumberColumn(format="%.3f"), "R² out-of-sample": st.column_config.NumberColumn(format="%+.3f"), "Rank correlation": st.column_config.NumberColumn(format="%+.3f"),
            "Direction right (%)": st.column_config.NumberColumn(format="%.1f%%"), "vs. always up (p)": st.column_config.NumberColumn(format="%.3f"), "AUC": st.column_config.NumberColumn(format="%.3f")})
        st.caption("p-values below 0.05 count as a clear difference. “Rolling average” predicts the recent average return every day.")
        st.download_button("Download the table (CSV)", data=table.to_csv().encode("utf-8"), file_name="daily_walk_forward_metrics.csv", mime="text/csv", icon=":material/download:")


def render(res: dict | None, preds: pd.DataFrame | None, reg: dict | None, compare: list | None, vix: list | None) -> None:
    if res is None or preds is None:
        st.info("The evaluation results are not available in this deployment.")
        return
    lstm = res["models"]["lstm"]
    verdict = price_verdict(lstm)
    st.subheader("Can this forecast be relied on?")
    _verdict(lstm, verdict)
    _evidence(res, preds, reg, verdict)
    st.write("")
    _mae_chart(res)
    st.write("")
    _range_chart(compare, vix)
    st.write("")
    _direction(res)
    _details(res, reg)
