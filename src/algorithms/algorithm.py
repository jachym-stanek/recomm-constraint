from typing import Dict, List
from src.segmentation import Segment
from src.constraints.constraint import Constraint


class Algorithm:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def solve(self, items: Dict[str, float],
              segmentations: List[Segment],
              constraints: List[Constraint],
              N: int):
        raise NotImplementedError("Solve method not implemented.")

    def __str__(self):
        return self.name + ': ' + self.description
