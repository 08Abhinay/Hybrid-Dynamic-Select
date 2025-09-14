from __future__ import annotations

from typing import List, Dict, Any


def pred_class(preds: List[Dict[str, Any]]) -> int:
    """Return the index of the highest-probability class.

    Accepts HF pipeline outputs of the form:
      [ {"label": str, "score": float}, ... ]
    or batched form: [[...]].
    """
    if preds and isinstance(preds[0], list):
        preds = preds[0]

    pred_idx = 0
    best = preds[0]["score"]
    for i in range(1, len(preds)):
        cur = preds[i]["score"]
        if cur > best:
            pred_idx = i
            best = cur
    return pred_idx


def get_prob(preds: List[Dict[str, Any]]) -> float:
    """Return the maximum probability from pipeline output.

    Accepts single or batched form.
    """
    if preds and isinstance(preds[0], list):
        preds = preds[0]
    return max(p["score"] for p in preds)


def get_attack_class(atk_prob: float) -> int:
    """Convert attack probability to a predicted class index (binary).

    Returns 1 if atk_prob < 0.5 else 0 to match original logic.
    """
    return 1 if atk_prob < 0.5 else 0

