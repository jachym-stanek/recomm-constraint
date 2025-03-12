from collections import deque

from src.algorithms.algorithm import Algorithm
from src.constraints import *


class IdfsSolver(Algorithm):
    def __init__(self, name="DFS", description="Informed Depth First Search Solver", verbose=True):
        super().__init__(name, description, verbose)

    def solve(self, items: Dict[str, float], segments: Dict[str, Segment], constraints: List[Constraint], N: int):
        partial_solution = list()   # init empty partial solution
        stack = deque()
        stack.append(partial_solution)

        # items as list sorted by value (sort ascending so that last to stack are the most scoring items)
        sorted_items = sorted(items, key=items.get)

        max_W = max(c.window_size for c in constraints if isinstance(c, MinItemsPerSegmentConstraint) or isinstance(c, MaxItemsPerSegmentConstraint)
                    or isinstance(c, GlobalMinItemsPerSegmentConstraint) or isinstance(c, GlobalMaxItemsPerSegmentConstraint)
                    or isinstance(c, MinSegmentsConstraint) or isinstance(c, MaxSegmentsConstraint))

        while stack:
            partial_solution = stack.pop()
            if len(partial_solution) == N:
                if self.check_partial_solution(partial_solution, items, segments, constraints):
                    return partial_solution
            else:
                for item in sorted_items:
                    if item not in partial_solution:
                        new_partial_solution = partial_solution.copy()
                        new_partial_solution.append(item)
                        if self.check_partial_solution(partial_solution[-max_W:], items, segments, constraints): # optimization: check only the last max_W items
                            stack.append(new_partial_solution)


    def check_partial_solution(self, partial_solution, items, segments, constraints: List[Constraint]):
        for constraint in constraints:
            if not constraint.check_constraint(partial_solution, items, segments):
                return False
        return True
