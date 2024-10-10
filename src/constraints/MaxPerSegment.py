from src.constraints.constraint import Constraint


class MaxPerSegmentConstraint(Constraint):
    def __init__(self, name, weight):
        super().__init__(name, weight)
