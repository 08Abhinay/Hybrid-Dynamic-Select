from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class SentimentPredictor:
    """Simple wrapper that mimics HF pipeline outputs for a given model/tokenizer."""

    def __init__(self, model_name_or_model, tokenizer=None, device: int | str | torch.device = 0):
        if isinstance(model_name_or_model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_model) if tokenizer is None else tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_model).eval()
        else:
            self.model = model_name_or_model.eval()
            self.tokenizer = tokenizer

        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model.to(self.device)

    def predict(self, text: str):
        return self.predict_batch([text])

    def predict_batch(self, texts: list[list[str]] | list[str]):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        id2label = self.model.config.id2label
        results = []
        for prob in probs:
            result = [{"label": id2label[i], "score": round(prob[i].item(), 4)} for i in range(len(prob))]
            results.append(result)
        return results

