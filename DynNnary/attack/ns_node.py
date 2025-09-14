from __future__ import annotations

from typing import List, Optional


class NSNode:
    def __init__(self, data: str, prob: float, N: int = 2):
        self.data = data
        self.prob = float(prob)
        self.exclusion: List[int] = []
        self.explored: bool = False
        self.children: List[Optional[NSNode]] = [None for _ in range(N)]
        self.N = N

    def isLeaf(self) -> bool:
        return not any(self.children)

    def lowestProb(self) -> Optional["NSNode"]:
        if self.isLeaf():
            return None if self.explored else self

        lows: List[Optional[NSNode]] = [None for _ in self.children]
        for i, child in enumerate(self.children):
            if child and not child.explored:
                lows[i] = child.lowestProb()

        if all(c is None or c.explored for c in self.children):
            self.explored = True
            return None

        lowest = None
        for cand in lows:
            if cand and (lowest is None or cand.prob < lowest.prob):
                lowest = cand
        if lowest is None:
            self.explored = True
        return lowest

