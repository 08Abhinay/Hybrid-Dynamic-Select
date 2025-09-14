from __future__ import annotations

from typing import List, Tuple

from nltk.tokenize import word_tokenize


def perturbation_rate(original_text: str, attacked_text: str) -> float:
    orig_tokens = word_tokenize(original_text)
    attacked_tokens = word_tokenize(attacked_text)
    min_len = min(len(orig_tokens), len(attacked_tokens))
    changed = sum(1 for i in range(min_len) if orig_tokens[i] != attacked_tokens[i])
    changed += abs(len(orig_tokens) - len(attacked_tokens))
    if len(orig_tokens) == 0:
        return 0.0
    return changed / len(orig_tokens)


def get_changed_words(original_text: str, attacked_text: str) -> List[Tuple[int, str | None, str | None]]:
    orig_tokens = word_tokenize(original_text)
    attacked_tokens = word_tokenize(attacked_text)
    min_len = min(len(orig_tokens), len(attacked_tokens))
    changed: List[Tuple[int, str | None, str | None]] = []
    for i in range(min_len):
        if orig_tokens[i] != attacked_tokens[i]:
            changed.append((i, orig_tokens[i], attacked_tokens[i]))
    if len(attacked_tokens) > len(orig_tokens):
        for i in range(min_len, len(attacked_tokens)):
            changed.append((i, None, attacked_tokens[i]))
    elif len(orig_tokens) > len(attacked_tokens):
        for i in range(min_len, len(orig_tokens)):
            changed.append((i, orig_tokens[i], None))
    return changed

