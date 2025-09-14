from __future__ import annotations

from typing import Tuple

from .n_ary import n_nary_select_iterative_NSNode
from ..replace.wordnet_replace import wordnet_replace
from ..replace.bert_replace import bert_replace


def NS_WNR(
    classifier,
    query: str,
    attack_class: int,
    n: int = 2,
    k: int = -1,
    replace: str = "wordnet",
) -> Tuple[bool, str, int, float]:
    """N-ary selection + replacement (WordNet or BERT)."""
    done = False
    query_count = 0
    cur_struct = None
    initial_prob = classifier(query)[0][attack_class]["score"]
    query_count += 1

    cur_query = query
    chars_changed = 0

    while not done:
        if cur_struct:
            replace_pos, final_prob, queries, cur_struct = n_nary_select_iterative_NSNode(
                classifier, query, attack_class, NSStruct=cur_struct, initial_prob=initial_prob, n=n
            )
        else:
            replace_pos, final_prob, queries, cur_struct = n_nary_select_iterative_NSNode(
                classifier, query, attack_class, initial_prob=initial_prob, n=n
            )
        query_count += queries

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
            if not cur_struct.lowestProb():
                done = True
        if k != -1 and chars_changed >= k:
            done = True

    return False, cur_query, query_count, cur_prob

