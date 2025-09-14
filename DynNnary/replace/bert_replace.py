from __future__ import annotations

from typing import Tuple, Optional


def bert_replace(
    classifier,
    query: str,
    initial_prob: float,
    replace_pos: int,
    attack_class: int,
    *,
    fill_mask=None,
    tokenizer=None,
) -> Tuple[bool, str, float, int]:
    """BERT fill-mask based replacement.

    Requires `fill_mask` pipeline and `tokenizer` to be provided by caller.
    If either is missing, returns no-op.
    """
    if fill_mask is None or tokenizer is None:
        return False, query, initial_prob, 0

    split_query = query.split()
    if replace_pos >= len(split_query):
        replace_pos = len(split_query) - 1

    replace_word = split_query[replace_pos]
    mask_token = tokenizer.mask_token

    # Place mask
    split_query[replace_pos] = mask_token
    masked_query = " ".join(split_query)

    # Encode/decode to respect max length
    tokenized = tokenizer.encode(masked_query, add_special_tokens=False)
    if len(tokenized) > 512:
        tokenized = tokenized[:512]
    masked_query = tokenizer.decode(tokenized, skip_special_tokens=True)

    if mask_token not in masked_query:
        return False, query, initial_prob, 0

    mask_predictions = fill_mask(masked_query, top_k=10)
    mask_candidates = [pred["token_str"] for pred in mask_predictions if pred["token_str"] != replace_word]
    query_count = 0
    if not mask_candidates:
        return False, query, initial_prob, query_count

    prob_list = []
    for candidate in mask_candidates:
        cur_query = split_query[:replace_pos] + [candidate] + split_query[replace_pos + 1 :]
        cur_query_str = " ".join(cur_query)
        cur_preds = classifier(cur_query_str)[0]
        query_count += 1
        cur_prob = cur_preds[attack_class]["score"]
        pred_idx = 0
        best = cur_preds[0]["score"]
        for i in range(1, len(cur_preds)):
            if cur_preds[i]["score"] > best:
                best = cur_preds[i]["score"]
                pred_idx = i
        if pred_idx != attack_class:
            return True, cur_query_str, cur_prob, query_count
        else:
            prob_list.append((cur_prob, cur_query_str))

    great_drop = 0.0
    choice = 0
    for i, (prob, q) in enumerate(prob_list):
        drop = initial_prob - prob
        if drop > great_drop:
            choice = i
            great_drop = drop

    return False, prob_list[choice][1], prob_list[choice][0], query_count

