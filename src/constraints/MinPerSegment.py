from src.constraints.constraint import Constraint


class MinPerSegmentConstraint(Constraint):
    def __init__(self, name, weight):
        super().__init__(name, weight)
