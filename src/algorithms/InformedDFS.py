from collections import deque

from src.algorithms.algorithm import Algorithm
from src.segmentation import Segment
from src.constraints.constraint import *


class IdfsSolver(Algorithm):
    def __init__(self, name="DFS", description="Informed Depth First Search Solver", verbose=True):
        super().__init__(name, description, verbose)

    def solve(self, items: Dict[str, float], segments: Dict[str, Segment], constraints: List[Constraint], N: int):
        partial_solution = list()   # init empty partial solution
        stack = deque()
        stack.append(partial_solution)

        while stack:
            partial_solution = stack.pop()
            if len(partial_solution) == N:
                if self.check_partial_solution(partial_solution, items, segments, constraints):
                    return partial_solution
            else:
                for item in items:
                    if item not in partial_solution.values():
                        new_partial_solution = partial_solution.copy()
                        new_partial_solution[len(partial_solution)+1] = item
                        if self.check_partial_solution(new_partial_solution, items, segments, constraints):
                            stack.append(new_partial_solution)


    def check_partial_solution(self, partial_solution, items, segments, constraints: List[Constraint]):
        for constraint in constraints:
            if not constraint.check_constraint(partial_solution, items, segments):
                return False
        return True
