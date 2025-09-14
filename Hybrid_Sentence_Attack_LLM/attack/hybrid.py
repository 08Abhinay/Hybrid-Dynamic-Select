from __future__ import annotations

from typing import Optional, Tuple, List

from nltk.tokenize import sent_tokenize

from .ns_node import NSNode
from .n_ary import n_nary_select_iterative_NSNode
from .greedy import GS_WNR


def get_most_influential_sentence(
    classifier,
    query: str,
    attack_class: int,
    root: Optional[NSNode],
    n: int = 2,
):
    query_count = 0
    original_prob = classifier(query)[0][attack_class]["score"]
    query_count += 1
    sentences = sent_tokenize(query)
    num_sentences = len(sentences)

    if not root:
        root = NSNode(query, float(original_prob), N=num_sentences)
        root.children = [None] * num_sentences

    for i, sentence in enumerate(sentences):
        if root.children[i] is None:
            root.children[i] = NSNode(sentence, float(original_prob), N=n)

    if len(root.prob_sorted) < num_sentences:
        modified_texts = [" ".join(sentences[:i] + sentences[i + 1 :]) for i in range(num_sentences)]
        modified_preds = classifier(modified_texts)
        query_count += len(modified_preds)
        for i, pred in enumerate(modified_preds):
            prob = pred[attack_class]["score"]
            root.children[i].prob = float(prob)
            root.prob_sorted[i] = float(prob)

    # The original script computes max change but only returns root, queries, original_prob
    return root, query_count, float(original_prob)


def Hybrid_WNR(
    classifier,
    query: str,
    attack_class: int,
    n: int = 2,
    k: int = 7,
    split_threshold_percentage: float | None = None,
):
    """Hybrid approach combining N-ary selection and greedy WordNet replace."""
    done = False
    query_count = 0
    root_node: Optional[NSNode] = None

    cur_query = query
    words_changed = 0

    all_segments: List[NSNode] = []
    segment_to_explore: Optional[NSNode] = None

    root_node, queries, initial_prob = get_most_influential_sentence(
        classifier, query, attack_class, root=None, n=n
    )
    query_count += queries

    while not done:
        if root_node:
            queries, root_node, segment_to_explore, all_segments = n_nary_select_iterative_NSNode(
                classifier,
                query,
                all_segments,
                split_threshold_percentage,
                attack_class,
                NSStruct=root_node,
                initial_prob=initial_prob,
                n=n,
                previous_segment=segment_to_explore,
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

