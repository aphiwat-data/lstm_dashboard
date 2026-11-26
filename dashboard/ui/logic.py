"""Small pieces of judgement for the evidence rows (kept free of Streamlit calls)."""
from __future__ import annotations


def price_verdict(model: dict) -> str:
    """'better' | 'worse' | 'same' for this model's price error against the simple guess of repeating the last close (Diebold-Mariano p < 0.05)."""
    p, skill = model["dm_p_vs_naive"], model["mae_skill_vs_naive"]
    return "better" if p < 0.05 and skill > 0 else "worse" if p < 0.05 and skill < 0 else "same"


def edge_color(verdict: str, diff: float) -> str:
    """Chip colour: green/red only when the difference is statistically clear, orange (a bit worse) or blue (a bit better) within noise."""
    return {"better": "green", "worse": "red"}.get(verdict) or ("orange" if diff > 0 else "blue")
