"""
One-page summary of the hourly XAU/USD findings, built only from hourly/results/*.json.

Usage: python make_summary_figure.py            # writes results/summary_hourly_findings.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parent / "results"
GRAY, GREEN, BLUE, RED, YELLOW = "#8a8a86", "#1BAF7A", "#2A78D6", "#E34948", "#EDA100"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117", "savefig.facecolor": "#0e1117",
        "text.color": "white", "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white", "font.size": 11,
    })


def panel_horizons(ax) -> None:
    files = {1: "model_results.json", 4: "model_results_h4.json", 24: "model_results_h24.json"}
    labels, base, gb, gru = [], [], [], []
    for h, f in files.items():
        t = load(f)["test"]
        head = t["hist_gb_clf"]["direction_from_head"]
        n_eff = head.get("n_eff", head["n"])  # older h=1 results predate the n_eff field (no label overlap at h=1)
        labels.append(f"{h} bar{'s' if h > 1 else ''}\n(n_eff {n_eff:,})")
        base.append(t["drift_only"]["direction_from_sign"]["always_up_acc"])
        gb.append(t["hist_gb_clf"]["direction_from_head"])
        gru.append(t["gru_ensemble"]["direction_from_head"])
    x, w = np.arange(len(labels)), 0.26
    ax.bar(x - w, np.array(base) * 100, w, color=GRAY, label="always up (bar to beat)")
    for off, rows, color, name in ((0, gb, GREEN, "HistGB classifier"), (w, gru, BLUE, "GRU, 2 seeds")):
        acc = np.array([r["acc"] for r in rows]) * 100
        err = np.array([[r["acc"] - r["ci95"][0], r["ci95"][1] - r["acc"]] for r in rows]).T * 100
        ax.bar(x + off, acc, w, color=color, label=name, yerr=err, ecolor="white", capsize=3)
    ax.set_ylim(46, 62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_title("A. No model beats 'always up' at any horizon (test 2025+, 95% CI)", fontsize=12, loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False)


def panel_spread(ax) -> None:
    s = load("spread_artifact_summary.json")["spread_bp_by_hour"]
    hours = np.arange(24)
    ax.plot(hours, s["all_years"], color=YELLOW, lw=2, marker="o", ms=4, label="all years")
    ax.plot(hours, s["2025+"], color=BLUE, lw=2, marker="o", ms=4, label="2025+")
    ax.axvspan(19.5, 22.5, color=RED, alpha=0.15)
    ax.text(19.3, max(s["all_years"]) * 0.96, "daily break", ha="right", color=RED, fontsize=10)
    ax.set_xlabel("UTC hour of bar")
    ax.set_ylabel("Median bid-ask spread (bp)")
    ax.set_title("B. Spread widens 2-3x around the daily break", fontsize=12, loc="left")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=9)


def panel_drift(ax) -> None:
    t = load("spread_artifact_summary.json")["next_bar_return_by_hour"]["ALL YEARS"]
    hours = list(range(16, 24))
    idx = [t["bar_open"].index(h) for h in hours]
    bid, mid = [t["bid_mean_bp"][i] for i in idx], [t["mid_mean_bp"][i] for i in idx]
    x, w = np.arange(len(hours)), 0.38
    ax.bar(x - w / 2, bid, w, color=RED, label="bid price")
    ax.bar(x + w / 2, mid, w, color=BLUE, label="mid price")
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_ylim(-1.8, 3.3)
    j = hours.index(19)
    ax.annotate(f"hour 19: bid {bid[j]:+.2f} bp (t={t['bid_t'][idx[j]]:.1f})\nmid {mid[j]:+.2f} bp (t={t['mid_t'][idx[j]]:.1f})",
                xy=(x[j] - w / 2, bid[j]), xytext=(x[0] - 0.35, -1.55), fontsize=9, color="white",
                arrowprops=dict(arrowstyle="->", color="white"))
    ax.set_xticks(x, hours)
    ax.set_xlabel("UTC hour of the current bar")
    ax.set_ylabel("Mean next-bar return (bp), 2004-2026")
    ax.set_title("C. The 'drift' at hour 19 exists only in bid prices", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def panel_edge(ax) -> None:
    covs = [0.2, 0.1, 0.05]
    sets = {"bid (same bars)": ("baseline_results_bidcommon.json", RED), "mid": ("baseline_results_mid.json", BLUE)}
    x, w = np.arange(len(covs)), 0.38
    for k, (name, (f, color)) in enumerate(sets.items()):
        rows = {r["coverage"]: r for r in load(f)["test"]["hist_gb"]["selective"]}
        edge = np.array([(rows[c]["acc"] - rows[c]["always_up_acc_same_rows"]) * 100 for c in covs])
        err = np.array([(rows[c]["ci95"][1] - rows[c]["ci95"][0]) / 2 * 100 for c in covs])
        ax.bar(x + (k - 0.5) * w, edge, w, color=color, label=name, yerr=err, ecolor="white", capsize=3)
        for i, c in enumerate(covs):
            ax.text(x[i] + (k - 0.5) * w, -8.3, f"n={rows[c]['n']:,}", ha="center", fontsize=8, color="#bbbbbb")
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_ylim(-9, 12)
    ax.set_xticks(x, [f"top {int(c * 100)}% most confident" for c in covs])
    ax.set_ylabel("Accuracy minus 'always up' (pt)")
    ax.set_title("D. The high-confidence 'edge' vanishes on mid prices (test)", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def main() -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, fn in zip(axes.ravel(), (panel_horizons, panel_spread, panel_drift, panel_edge)):
        fn(ax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Hourly XAU/USD direction: no tradable edge found; an apparent ~60% was largely a bid-spread artifact",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005,
             "Dukascopy h1 bars 2004 - Aug 2026 | train <=2023, val 2024, test 2025+ | bar to beat = always-up rate (51.5% / 52.9% / 55.1% at 1 / 4 / 24 bars)\n"
             "Bid-vs-mid panels use bars present in both feeds (ask missing 4 months); D: HistGB on 21 gold-only features; CI for h>1 uses n_eff = n / h",
             ha="center", fontsize=9, color="#aaaaaa")
    plt.tight_layout(rect=(0, 0.04, 1, 0.965))
    out = RES / "summary_hourly_findings.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
