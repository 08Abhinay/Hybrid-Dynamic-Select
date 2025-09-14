from __future__ import annotations

from sentence_transformers import SentenceTransformer
import torch


class SemanticSimilarity:
    """Compute cosine similarity between sentence embeddings using SBERT."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, original_text: str, attacked_text: str) -> float:
        orig_emb = self.model.encode(original_text, convert_to_tensor=True)
        atk_emb = self.model.encode(attacked_text, convert_to_tensor=True)
        cos_sim = torch.nn.functional.cosine_similarity(orig_emb, atk_emb, dim=0)
        return float(cos_sim.item())

