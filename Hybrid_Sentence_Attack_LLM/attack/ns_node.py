from __future__ import annotations

from typing import List, Optional


class NSNode:
    """N-ary search tree node for text segments.

    Attributes
    -----------
    data: str
        The text (or segment) represented by this node.
    prob: float
        The classifier probability for the attack class on this data.
    exclusion: List[int]
        Indices excluded to produce this node's data from the original text.
    explored: bool
        Whether this node has been fully explored.
    children: List[Optional[NSNode]]
        N-ary children representing further exclusions.
    N: int
        Branching factor.
    prob_sorted: dict
        Auxiliary map used by greedy selection.
    """

    def __init__(self, data: str, prob: float, N: int = 2):
        self.data = data
        self.prob = prob
        self.exclusion: List[int] = []
        self.explored = False
        self.children: List[Optional[NSNode]] = [None for _ in range(N)]
        self.N = N
        self.prob_sorted: dict = {}

    def isLeaf(self) -> bool:
        return not any(self.children)

    def printNode(self) -> None:
        if self.isLeaf():
            print(self.data, self.prob, self.explored)
        else:
            for i in range(self.N // 2):
                if self.children[i]:
                    self.children[i].printNode()
            print(self.data, self.prob, self.explored)
            for i in range(self.N // 2, self.N):
                if self.children[i]:
                    self.children[i].printNode()

    def lowestProb(self) -> Optional["NSNode"]:
        """Return the lowest-prob non-explored descendant, or None if exhausted."""
        if self.isLeaf():
            return None if self.explored else self

        lows: List[Optional[NSNode]] = [None for _ in self.children]
        for i, child in enumerate(self.children):
            if child and not child.explored:
                lows[i] = child.lowestProb()

        # If all children are explored (or None), mark this node explored
        if all(c is None or c.explored for c in self.children):
            self.explored = True
            return None

        lowest_pos = -1
        lowest_prob = float("inf")
        for i, cand in enumerate(lows):
            if cand and cand.prob < lowest_prob:
                lowest_pos = i
                lowest_prob = cand.prob

        if lowest_pos == -1:
            self.explored = True
            return None

        return lows[lowest_pos]

