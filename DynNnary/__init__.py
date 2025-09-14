"""
DynNnary: Modularized N-ary and Greedy WNR attacks extracted from
`scripts/dyn_N_GPTneo.py`.

Includes optional automatic N selection via `get_n_gpt_neo` or manual N.
"""

from __future__ import annotations

try:  # pragma: no cover
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
except Exception:
    pass

from .attack.nnary import NS_WNR
from .attack.greedy import GS_WNR
from .experiments.runner import run_attack_experiments
from .utils.prediction import pred_class, get_prob, get_attack_class
from .utils.n_selector import get_n_gpt_neo
from .metrics.perturbation import perturbation_rate, get_changed_words
from .metrics.similarity import SemanticSimilarity

__all__ = [
    "NS_WNR",
    "GS_WNR",
    "run_attack_experiments",
    "pred_class",
    "get_prob",
    "get_attack_class",
    "get_n_gpt_neo",
    "perturbation_rate",
    "get_changed_words",
    "SemanticSimilarity",
]

