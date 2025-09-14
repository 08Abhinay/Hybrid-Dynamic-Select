"""
Hybrid-Sentence-Attack-LLM: Modularized components extracted from
`scripts/hybrid_sentence_attack_LLM.py`.

This package mirrors the original script’s functionality but organizes it into
reusable modules following a typical modern repository structure.

Note: The original script remains unchanged.
"""

from __future__ import annotations

# Ensure required NLTK resources are available when the package is imported.
# This mirrors the behavior in the original script which downloaded these
# resources at import time.
try:  # pragma: no cover
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
except Exception:
    # If running in an environment without network, caller should ensure
    # NLTK data availability; we avoid hard failing at import.
    pass

# Re-export commonly used entrypoints
from .attack.hybrid import Hybrid_WNR, get_most_influential_sentence
from .experiments.runner import run_attack_experiments
from .metrics.perturbation import perturbation_rate
from .metrics.similarity import SemanticSimilarity
from .utils.prediction import pred_class, get_prob, get_attack_class

__all__ = [
    "Hybrid_WNR",
    "get_most_influential_sentence",
    "run_attack_experiments",
    "perturbation_rate",
    "SemanticSimilarity",
    "pred_class",
    "get_prob",
    "get_attack_class",
]

