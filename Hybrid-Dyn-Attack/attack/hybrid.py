from __future__ import annotations

from typing import Optional, Tuple, List

from attack.ns_node import NSNode
from attack.n_ary import n_nary_select_iterative_NSNode
from attack.greedy import GS_WNR


def Hybrid_WNR(
    classifier,
    query: str,
    attack_class: int,
    n: int = 2,
    k: int = 7,
    split_threshold: float | int | None = None,
):
    """Non sentence-level hybrid attack combining N-ary selection and greedy WNR."""
    done = False
    query_count = 0
    root_node: Optional[NSNode] = None
    initial_prob = classifier(query)[0][attack_class]["score"]
    query_count += 1

    cur_query = query
    words_changed = 0
    all_segments: List[NSNode] = []
    segment_to_explore: Optional[NSNode] = None

    while not done:
        if root_node:
            queries, root_node, segment_to_explore, all_segments = n_nary_select_iterative_NSNode(
                classifier,
                query,
                all_segments,
                split_threshold,
                attack_class,
                NSStruct=root_node,
                initial_prob=initial_prob,
                n=n,
                previous_segment=segment_to_explore,
            )
        else:
            queries, root_node, segment_to_explore, all_segments = n_nary_select_iterative_NSNode(
                classifier,
                query,
                all_segments,
                split_threshold,
                attack_class,
                NSStruct=root_node,
                initial_prob=initial_prob,
                n=n,
            )
        query_count += queries

        success, cur_query, queries, cur_prob, segment_to_explore = GS_WNR(
            segment_to_explore,
            classifier,
            initial_prob,
            query=cur_query,
            attack_class=attack_class,
            k=k,
            n=n,
        )
        query_count += queries
        words_changed += 1

        if success:
            return True, cur_query, query_count, cur_prob
        else:
            if not root_node.lowestProb():
                done = True
        if k != -1 and words_changed >= k:
            done = True

    return False, cur_query, query_count, initial_prob
