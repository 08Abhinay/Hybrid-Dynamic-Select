from __future__ import annotations

from typing import List, Tuple


def find_word_indices(query_tokens: List[str], subquery_tokens: List[str]) -> Tuple[int, int]:
    """Find start/end indices of subquery_tokens within query_tokens.

    Returns (-1, -1) if not found.
    """
    for i in range(len(query_tokens) - len(subquery_tokens) + 1):
        if query_tokens[i : i + len(subquery_tokens)] == subquery_tokens:
            return i, i + len(subquery_tokens) - 1
    return -1, -1

