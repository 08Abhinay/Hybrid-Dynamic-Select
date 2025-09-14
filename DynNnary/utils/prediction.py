from __future__ import annotations

from typing import Any, Dict, List


def pred_class(preds: List[Dict[str, Any]] | List[List[Dict[str, Any]]]) -> int:
    if preds and isinstance(preds[0], list):
        preds = preds[0]  # type: ignore[assignment]
    pred = 0
    score = preds[0]["score"]
    for i in range(1, len(preds)):
        cur = preds[i]["score"]
        if cur > score:
            pred = i
            score = cur
    return pred


def get_prob(preds: List[Dict[str, Any]] | List[List[Dict[str, Any]]]) -> float:
    if preds and isinstance(preds[0], list):
        preds = preds[0]  # type: ignore[assignment]
    return float(max(p["score"] for p in preds))


def get_attack_class(atk_prob: float) -> int:
    return 1 if atk_prob < 0.5 else 0

