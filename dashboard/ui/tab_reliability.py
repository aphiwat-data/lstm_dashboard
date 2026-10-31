"""How reliable? The verdict first, then the evidence in plain words; statistics are one click deeper."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .logic import PHRASE, direction_verdict, edge_color, pct_miss, price_verdict
from .theme import COLORS, finish, signed_usd, usd

MODEL_LABELS = {"drift": "Recent average return", "ridge": "Ridge", "hist_gb_reg": "Gradient boosting", "hist_gb_clf": "Gradient boosting (up/down)", "lstm": "LSTM (this model)", "momentum_rule": "Follow last move"}


def _verdict(lstm: dict, verdict: str) -> None:
    if verdict == "better":
        st.success("**A measurable edge.** The typical error is clearly lower than “tomorrow = today”.", icon=":material/check_circle:")
    elif verdict == "worse":
        st.error("**Worse than the simple baseline.** The typical error is clearly higher than repeating today's price.", icon=":material/cancel:")
    else:
        st.info("**Not better than the baseline.** Level with “tomorrow = today”, so read it as *about today's close*. The range, however, is trustworthy.", icon=":material/balance:")


def _row(label: str, value: str, chip: str, color: str, line: str, help: str) -> None:
    """One line of evidence: the number on the left, one short line on the right (it wraps under the number on a phone); the detail sits in the tooltip."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.metric(label, value, chip, delta_color=color, delta_arrow="off", width=240, help=help)
        with st.container():
            st.markdown(line)


def _evidence(res: dict, preds: pd.DataFrame, reg: dict | None, verdict: str) -> None:
    lstm = res["models"]["lstm"]
    pct_n = pct_miss(preds)[1]
    st.subheader("The evidence", help=f"Based on {res['protocol']['n_test']} test days ({res['protocol']['test_start']} to {res['protocol']['test_end']}); small differences are within noise.")
    with st.container(border=True):
        _row("Typical error", usd(lstm["mae_usd"]), f"{signed_usd(lstm['mae_usd'] - lstm['naive_mae_usd'])} vs. baseline", edge_color(verdict, lstm["mae_usd"] - lstm["naive_mae_usd"]),
             f"The simple guess “tomorrow = today” misses by {usd(lstm['naive_mae_usd'])}.",
             f"That guess misses by {pct_n:.2f}% of the price. The model {PHRASE[verdict]} it (p = {lstm['dm_p_vs_naive']:.2f}; below 0.05 counts as a clear difference).")
        if reg:
            m, iv = reg["models"]["lstm"], reg["interval"]
            signal = m["r2_os_vs_rolling_mean"] > 0 and m["ic_p"] < 0.05
            e, nc = iv["lstm_ewma_gaussian"], iv["no_change_ewma_gaussian"]
            st.divider()
            _row("Predicts daily moves?", f"{m['r2_os_vs_rolling_mean']:+.3f}", "Some predictive power" if signal else "No predictive power", "green" if signal else "orange",
                 "Below 0 means no better than a simple average.",
                 f"Out-of-sample R²: 0 means no better than a simple average, below 0 is worse. Rank correlation with the actual moves: {m['ic_spearman']:+.3f} (p = {m['ic_p']:.2f}).")
            st.divider()
            _row("Range hit rate", f"{e['coverage']:.1%}", "Target 90%", "green" if abs(e["coverage"] - 0.9) <= 0.02 else "orange",
                 f"The range held on {round(e['coverage'] * e['n'])} of {e['n']} test days.",
                 f"Ideal: 90%. Its width, {usd(e['mean_width_usd'], 0)}, is the same as a no-change range ({usd(nc['mean_width_usd'], 0)}): the model does not narrow the uncertainty.")


def _mae_chart(res: dict) -> None:
    models = res["models"]
    naive = models["lstm"]["naive_mae_usd"]
    entries = [("Baseline<br>(tomorrow = today)", naive, COLORS["naive"]), ("Ridge<br>(linear model)", models["ridge"]["mae_usd"], COLORS["ridge"]),
               ("Gradient boosting<br>(tree model)", models["hist_gb_reg"]["mae_usd"], COLORS["hgb"]), ("LSTM<br>(this model)", models["lstm"]["mae_usd"], COLORS["model"])]
    vals = [v for _, v, _ in entries]
    fig = go.Figure(go.Bar(x=[f"{n}<br><b>${v:,.2f}</b>" for n, v, _ in entries], customdata=[n.replace("<br>", " ") for n, _, _ in entries], y=vals, marker=dict(color=[c for *_, c in entries], pattern_shape=["/", "", "", ""], pattern_fgcolor="rgba(255,255,255,0.55)"),
                           hovertemplate="%{customdata}<br>%{y:$,.2f}<extra></extra>"))
    fig.add_hline(y=naive, layer="below", line_dash="dash", line_color=COLORS["naive"], line_width=1.2)
    finish(fig, 300, hover="closest")
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(range=[0, max(vals) * 1.2], tickprefix="$")
    better = [name for name, key in (("Ridge", "ridge"), ("Gradient boosting", "hist_gb_reg"), ("LSTM", "lstm")) if models[key]["dm_p_vs_naive"] < 0.05 and models[key]["mae_usd"] < naive]
    st.markdown("**Typical daily error by model**", help="Lower is better. A model has to land clearly below the dashed line to add value. " + (f"{', '.join(better)} {'does' if len(better) == 1 else 'do'}." if better else "None does."))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _range_chart(compare: list | None, vix: list | None) -> None:
    if not compare:
        return
    wanted = [("EWMA Gaussian, lambda 0.94 (current)", "Recent errors<br>(this model)"), ("Empirical quantiles, all past errors", "All past errors"), ("GARCH(1,1), Gaussian", "GARCH<br>(volatility model)"), ("LSTM quantile (pinball loss)", "LSTM learns<br>the range")]
    by = {r["method"]: r for r in compare}
    rows = [(label, by[key]["coverage"]) for key, label in wanted if key in by]
    if vix:
        v = next((r for r in vix if r["method"].startswith("EWMA + VIX (previous day)")), None)
        if v:
            rows.append(("Recent errors<br>+ VIX", v["coverage"]))
    colors = [COLORS["model"] if i == 0 else COLORS["naive"] for i in range(len(rows))]
    fig = go.Figure(go.Bar(x=[f"{r[0]}<br><b>{r[1]:.1%}</b>" for r in rows], customdata=[r[0].replace("<br>", " ") for r in rows], y=[r[1] * 100 for r in rows], marker_color=colors,
                           hovertemplate="%{customdata}<br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=90, layer="below", line_dash="dash", line_color=COLORS["actual"], line_width=1.2)
    finish(fig, 300, hover="closest")
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(range=[50, 100], ticksuffix="%")
    used, others = rows[0][1], [r[1] for r in rows[1:]]
    st.markdown("**How often the 90% range held, by way of drawing it**", help=f"Dashed line: the 90% target. We compared {len(rows)} ways to draw the range." + (" The first one, used in this model, landed closest to 90%." if all(abs(used - 0.9) <= abs(o - 0.9) for o in others) else "")
                + (" The others were too narrow." if all(o < used for o in others) else ""))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _direction(res: dict) -> None:
    models = res["models"]
    lstm = models["lstm"]
    hd = lstm["direction_from_head"]
    up = models["drift"]["direction_from_sign"]["always_up_acc"]
    v = direction_verdict(lstm, up)
    with st.expander("Side check: does it get up or down right?", icon=":material/swap_vert:"):
        st.markdown(f"Right on **{hd['acc']:.1%}** of days; always saying “up” scores **{up:.1%}**. The model {PHRASE[v]} that (p = {hd['p_vs_always_up']:.2f}).",
                    help="Calling up or down is a yes/no question, not how a price forecast is judged, so this is only a side check. The model says “up” on about nine days in ten, "
                         f"so its score mostly follows how often gold rose. 95% interval: {hd['ci95'][0]:.1%} to {hd['ci95'][1]:.1%}; the bars show the same intervals.")
        entries = [("Always guess up", models["drift"]["direction_from_sign"]), ("Follow last move", models["momentum_rule"]["direction_from_head"]), ("Ridge (linear)", models["ridge"]["direction_from_sign"]),
                   ("Gradient boosting (trees)", models["hist_gb_clf"]["direction_from_head"]), ("LSTM (this model)", models["lstm"]["direction_from_head"])]
        acc = [d["acc"] * 100 for _, d in entries]
        fig = go.Figure(go.Bar(x=[f"{n}<br><b>{a:.1f}%</b>" for (n, _), a in zip(entries, acc)], y=acc, marker_color=[COLORS["naive"], COLORS["good"], COLORS["ridge"], COLORS["hgb"], COLORS["model"]],
                               error_y=dict(type="data", symmetric=False, array=[(d["ci95"][1] - d["acc"]) * 100 for _, d in entries], arrayminus=[(d["acc"] - d["ci95"][0]) * 100 for _, d in entries], thickness=1.2, width=4)))
        fig.add_hline(y=up * 100, line_dash="dash", line_color=COLORS["naive"], layer="below")
        finish(fig, 320, hover="closest")
        fig.update_yaxes(range=[35, 75], ticksuffix="%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _details(res: dict, reg: dict | None) -> None:
    with st.expander("All the statistics, model by model", icon=":material/table:"):
        rows = {}
        for key, r in res["models"].items():
            d = r.get("direction_from_head") or r["direction_from_sign"]
            m = (reg["models"].get(key) if reg else None) or {}
            rows[MODEL_LABELS.get(key, key)] = {"Typical error ($)": r.get("mae_usd"), "Typical error (%)": m.get("mape_pct"), "RMSE ($)": r.get("rmse_usd"),
                                                "vs. baseline (p)": r.get("dm_p_vs_naive"), "R² out-of-sample": m.get("r2_os_vs_rolling_mean"), "Rank correlation": r.get("ic_spearman"),
                                                "Direction right (%)": d["acc"] * 100, "vs. always up (p)": d["p_vs_always_up"], "AUC": (r.get("direction_from_head") or {}).get("auc")}
        table = pd.DataFrame(rows).T.astype(float)
        st.dataframe(table, width="stretch", column_config={
            "Typical error ($)": st.column_config.NumberColumn(format="$%.2f"), "Typical error (%)": st.column_config.NumberColumn(format="%.3f%%"), "RMSE ($)": st.column_config.NumberColumn(format="$%.2f"),
            "vs. baseline (p)": st.column_config.NumberColumn(format="%.3f"), "R² out-of-sample": st.column_config.NumberColumn(format="%+.3f"), "Rank correlation": st.column_config.NumberColumn(format="%+.3f"),
            "Direction right (%)": st.column_config.NumberColumn(format="%.1f%%"), "vs. always up (p)": st.column_config.NumberColumn(format="%.3f"), "AUC": st.column_config.NumberColumn(format="%.3f")})
        st.caption("p below 0.05 = a clear difference.")
        st.download_button("Download CSV", data=table.to_csv().encode("utf-8"), file_name="daily_walk_forward_metrics.csv", mime="text/csv", icon=":material/download:")


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
