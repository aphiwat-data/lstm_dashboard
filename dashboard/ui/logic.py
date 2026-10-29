"""Small pieces of judgement shared by several sections (kept free of Streamlit calls)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_verdict(lstm: dict) -> str:
    """'better' | 'worse' | 'same' for the LSTM's price error against the baseline "tomorrow = today" (Diebold-Mariano p < 0.05)."""
    p, skill = lstm["dm_p_vs_naive"], lstm["mae_skill_vs_naive"]
    return "better" if p < 0.05 and skill > 0 else "worse" if p < 0.05 and skill < 0 else "same"


def pct_miss(preds: pd.DataFrame) -> tuple[float, float]:
    """Mean absolute percentage miss of the LSTM forecast and of a "tomorrow = today" guess on the same days."""
    actual = preds["meta_next_close"]
    lstm = (preds["meta_close"] * np.exp(preds["pred_ret_lstm"]) - actual).abs() / actual
    naive = (preds["meta_close"] - actual).abs() / actual
    return float(lstm.mean() * 100), float(naive.mean() * 100)


def edge_color(verdict: str, diff: float) -> str:
    """Chip colour: green/red only when the difference is statistically clear, orange (a bit worse) or blue (a bit better) within noise."""
    return {"better": "green", "worse": "red"}.get(verdict) or ("orange" if diff > 0 else "blue")


VERDICT_WORDS = {"better": "Yes", "same": "Not clearly", "worse": "No, worse"}
PHRASE = {"better": "beats", "worse": "is worse than", "same": "cannot be told apart from"}
