from typing import Dict, List
from src.segmentation import Segment
from src.constraints import Constraint


class Algorithm:
    def __init__(self, name, description, verbose=False):
        self.name = name
        self.description = description
        self.verbose = verbose

    def solve(self, items: Dict[str, float], segments: Dict[str, Segment], constraints: List[Constraint], N: int):
        raise NotImplementedError("Solve method not implemented.")

    def __str__(self):
        return self.name + ': ' + self.description
