from __future__ import annotations

from sentence_transformers import SentenceTransformer
import torch


class SemanticSimilarity:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, original_text: str, attacked_text: str) -> float:
        a = self.model.encode(original_text, convert_to_tensor=True)
        b = self.model.encode(attacked_text, convert_to_tensor=True)
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())

