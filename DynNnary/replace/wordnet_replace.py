from __future__ import annotations

from typing import List, Tuple

from nltk.corpus import wordnet


def get_synonyms(word: str) -> List[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms - {word})


def wordnet_replace(
    classifier,
    query: str,
    initial_prob: float,
    replace_pos: int,
    attack_class: int,
) -> Tuple[bool, str, float, int]:
    split_query = query.split()
    if replace_pos >= len(split_query):
        replace_pos = len(split_query) - 1
    replace_word = split_query[replace_pos]
    syns = get_synonyms(replace_word)
    query_count = 0

    if len(syns) == 0:
        return False, query, initial_prob, query_count

    prob_list: List[tuple[float, str]] = []
    for cur_syn in syns:
        cur_query = split_query[:replace_pos] + [cur_syn] + split_query[replace_pos + 1 :]
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

