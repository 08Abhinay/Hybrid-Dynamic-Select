from __future__ import annotations

from typing import List, Tuple, Optional

from .ns_node import NSNode


def n_nary_select_iterative_NSNode(
    classifier,
    query: str,
    attack_class: int = 0,
    NSStruct: Optional[NSNode] = None,
    initial_prob: Optional[float] = None,
    n: int = 2,
) -> Tuple[int, float, int, NSNode]:
    """Iteratively identify most-influential position using N-ary segmentation."""
    query_count = 0

    if not NSStruct:
        if not initial_prob:
            initial_prob = classifier(query)[0][attack_class]["score"]
        query_count += 1
        NSStruct = NSNode(query, float(initial_prob), n)

    initial_prob = NSStruct.prob
    cur_struct = NSStruct.lowestProb()
    final_prob = initial_prob
    split_query = query.split()

    if len(cur_struct.exclusion) == 1:
        cur_struct.explored = True
        return cur_struct.exclusion[0], cur_struct.prob, 0, NSStruct
    else:
        if len(cur_struct.exclusion) == 0:
            start = 0
            end = len(split_query) - 1
        else:
            start = cur_struct.exclusion[0]
            end = start + len(cur_struct.exclusion) - 1

    while start < end:
        positions = [start]
        cur_pos = start
        step = (end - start) // n
        for _ in range(n - 1):
            next_pos = cur_pos + step + 1
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
            cur_struct.children[i] = NSNode(cur_query, float(cur_prob), n)
            cur_struct.children[i].exclusion = exclusions[i]

        g_drop = prob_drops.index(max(prob_drops))
        if len(cur_struct.children[g_drop].exclusion) == 1:
            cur_struct.children[g_drop].explored = True

        final_prob = prob_drops[g_drop]
        cur_struct = cur_struct.children[g_drop]
        start = cur_struct.exclusion[0]
        end = cur_struct.exclusion[-1]

        most_influential_pos = start

    return most_influential_pos, final_prob, query_count, NSStruct
