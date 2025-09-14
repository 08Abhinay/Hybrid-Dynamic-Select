from __future__ import annotations

from typing import List, Tuple, Optional

from attack.ns_node import NSNode


def n_nary_select_iterative_NSNode(
    classifier,
    query: str,
    all_segments: List[NSNode],
    split_threshold: float | int | None = 0.1,
    attack_class: int = 0,
    NSStruct: Optional[NSNode] = None,
    initial_prob: Optional[float] = None,
    n: int = 2,
    previous_segment: Optional[NSNode] = None,
) -> Tuple[int, NSNode, NSNode, List[NSNode]]:
    """Iteratively explore an N-ary segmentation tree to identify a segment.

    This version mirrors scripts/Hybrid_dyn_N_LLM.py behavior (not sentence-level).
    """
    query_count = 0
    split_query = query.split()
    if split_threshold is None:
        split_threshold = max(1, int(0.1 * len(split_query)))
    elif isinstance(split_threshold, float):
        split_threshold = max(1, int(split_threshold * len(split_query)))

    if not NSStruct:
        if not initial_prob:
            initial_prob = classifier(query)[0][attack_class]["score"]
        query_count += 1
        NSStruct = NSNode(query, float(initial_prob), n)

    initial_prob = NSStruct.prob

    cur_struct = NSStruct.lowestProb()
    if previous_segment and cur_struct in previous_segment.children:
        return 0, NSStruct, previous_segment, all_segments

    while True:
        if len(cur_struct.exclusion) > 0 and len(cur_struct.exclusion) <= split_threshold:
            if cur_struct not in all_segments:
                all_segments.append(cur_struct)
            return query_count, NSStruct, cur_struct, all_segments

        for segment in all_segments:
            if cur_struct in segment.children:
                return query_count, NSStruct, segment, all_segments

        if len(cur_struct.exclusion) == 1:
            cur_struct.explored = True
            return 0, NSStruct, cur_struct, all_segments
        else:
            if len(cur_struct.exclusion) == 0:
                start = 0
                end = len(split_query) - 1
            else:
                start = cur_struct.exclusion[0]
                end = start + len(cur_struct.exclusion) - 1

        positions = [start]
        cur_pos = start
        step = (end - start) // n
        for _ in range(n - 1):
            next_pos = cur_pos + step + 1
            if next_pos > len(split_query):
                continue
            positions.append(next_pos)
            cur_pos = next_pos
        positions.append(end)

        exclusions: List[List[int]] = []
        for i in range(len(positions) - 1):
            if i + 1 == len(positions) - 1:
                cur_exclusions = list(range(positions[i], positions[i + 1] + 1))
            else:
                cur_exclusions = list(range(positions[i], positions[i + 1]))
            exclusions.append(cur_exclusions)

        queries: List[str] = []
        for excl in exclusions:
            if len(excl) == 0:
                continue
            cur_n_query = " ".join([split_query[j] for j in range(len(split_query)) if j not in excl])
            queries.append(cur_n_query)

        prob_drops: List[float] = []
        for i, cur_query in enumerate(queries):
            cur_prob = classifier(cur_query)[0][attack_class]["score"]
            query_count += 1
            prob_drops.append(initial_prob - cur_prob)
            if cur_struct.children[i] is None:
                node = NSNode(cur_query, float(cur_prob), n)
                node.exclusion = exclusions[i]
                cur_struct.children[i] = node

        g_drop = prob_drops.index(max(prob_drops))
        if len(cur_struct.children[g_drop].exclusion) == 1:
            cur_struct.children[g_drop].explored = True

        cur_struct = cur_struct.children[g_drop]
