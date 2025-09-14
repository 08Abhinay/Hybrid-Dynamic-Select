from __future__ import annotations

from typing import Dict, List, Tuple

from .ns_node import NSNode
from ..replace.wordnet_replace import wordnet_replace


def GreedySelect(
    segment_to_explore: NSNode,
    classifier,
    query: str,
    initial_prob: float,
    attack_class: int = 0,
    excluded_indices: List[int] | None = None,
    n: int = 2,
) -> Tuple[Dict[int, float], NSNode]:
    split_query = query.split()
    queries: List[str] = []
    prob_drops: Dict[int, float] = {}

    while len(segment_to_explore.children) < len(segment_to_explore.exclusion):
        segment_to_explore.children.append(None)

    for exclusion in segment_to_explore.exclusion:
        if exclusion < len(split_query) - 1:
            cur_query = " ".join(split_query[:exclusion] + split_query[exclusion + 1 :])
        else:
            cur_query = " ".join(split_query[:exclusion])
        queries.append(cur_query)

    batch_results = classifier(queries)
    for i, exclusion in enumerate(segment_to_explore.exclusion):
        cur_prob = batch_results[i][attack_class]["score"]
        prob_drops[i] = initial_prob - cur_prob
        ns_node = NSNode(data=split_query[exclusion], prob=float(cur_prob), N=n)
        segment_to_explore.children[i] = ns_node

    return prob_drops, segment_to_explore


def GS_WNR(
    segment_to_explore: NSNode,
    classifier,
    initial_prob: float,
    query: str,
    attack_class: int,
    k: int = -1,
    n: int = 2,
):
    query_count = 0
    cur_query = query
    cur_prob = initial_prob

    if len(segment_to_explore.prob_sorted) < len(segment_to_explore.exclusion):
        prob_drops, segment_to_explore = GreedySelect(
            segment_to_explore,
            classifier,
            query,
            initial_prob,
            attack_class=attack_class,
            excluded_indices=list(range(len(segment_to_explore.exclusion))),
            n=n,
        )
        segment_to_explore.prob_sorted.update(prob_drops)

        valid_prob_drops = [p for p in segment_to_explore.prob_sorted.values() if p is not None]
        query_count += len(valid_prob_drops)

    prob_sorted = dict(sorted(segment_to_explore.prob_sorted.items(), key=lambda item: item[1], reverse=True))

    for index in prob_sorted.keys():
        if segment_to_explore.children[index].explored:
            continue
        replace_pos = segment_to_explore.exclusion[index]
        success, cur_query, cur_prob, queries = False, cur_query, initial_prob, 0
        success, cur_query, cur_prob, queries = wordnet_replace(
            classifier, cur_query, initial_prob, replace_pos, attack_class
        )
        query_count += queries

        if success:
            segment_to_explore.children[index].explored = True
            return True, cur_query, query_count, cur_prob, segment_to_explore
        else:
            segment_to_explore.children[index].explored = True
            break

    return False, cur_query, query_count, cur_prob, segment_to_explore
