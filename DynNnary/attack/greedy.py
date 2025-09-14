from __future__ import annotations

from typing import Dict, List, Tuple

from ..replace.wordnet_replace import wordnet_replace
from ..replace.bert_replace import bert_replace


def GreedySelect(classifier, query: str, attack_class: int = 0, initial_prob: float | None = None):
    if initial_prob is None:
        initial_prob = classifier(query)[0][attack_class]["score"]
    prob_drops: Dict[int, float] = {}
    split_query = query.split()
    for i in range(len(split_query)):
        cur_query = " ".join(split_query[:i] + split_query[i + 1 :])
        cur_prob = classifier(cur_query)[0][attack_class]["score"]
        prob_drops[i] = float(initial_prob) - float(cur_prob)
    return prob_drops


def GS_WNR(
    classifier,
    query: str,
    attack_class: int,
    k: int = -1,
    replace: str = "wordnet",
):
    done = False
    query_count = 0
    initial_prob = classifier(query)[0][attack_class]["score"]
    query_count += 1

    cur_query = query
    prob_drops = GreedySelect(classifier, query, attack_class, initial_prob)
    query_count += len(prob_drops)
    prob_sorted = dict(sorted(prob_drops.items(), key=lambda item: item[1], reverse=True))
    chars_changed = 0

    while not done:
        replace_pos = list(prob_sorted.keys())[0]
        prob_sorted.pop(replace_pos)
        if replace == "bert":
            success, cur_query, cur_prob, queries = bert_replace(
                classifier, cur_query, initial_prob, replace_pos, attack_class
            )
        else:
            success, cur_query, cur_prob, queries = wordnet_replace(
                classifier, cur_query, initial_prob, replace_pos, attack_class
            )
        chars_changed += 1
        query_count += queries

        if success:
            return True, cur_query, query_count, cur_prob
        else:
            if len(prob_sorted) == 0:
                done = True
        if k != -1 and chars_changed >= k:
            done = True

    return False, cur_query, query_count, cur_prob

