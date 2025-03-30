############################################
# DISCLAIMER
# This code in its current  state might not be runnable as it was not updated thoroughly after every refactoring.
# It is meant to be used as a reference for the experiments conducted in the thesis.
############################################

import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

from src.algorithms.ILP import IlpSolver
from src.algorithms.CP import CpSolver, PermutationCpSolver
from src.algorithms.Preprocessor import ItemPreprocessor
from src.algorithms.StateSpaceSearch import StateSpaceSolver
from src.constraints import *
from ilp_experiments import run_test_preprocessing as run_ilp_test_preprocessing
from ilp_experiments import run_test_all_approaches as run_ilp_test_all_approaches
from dfs_experiments import run_test_idfs
from src.constraint_generator import ConstraintGenerator


def basic_state_space_search_test():
    # Test 1 N=10, M=100, S=10
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = {f'segment-{i}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(10)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
        MinSegmentsConstraint(segmentation_property, 2, 5)
    ]
    N = 10
    solver = StateSpaceSolver()
    solution = solver.solve(items, segments, constraints, N)
    print(f"Solution: {solution}")
    score = sum(items[item] for item in solution.values())
    constraints_satisfied = all(constraint.check_constraint(solution, items, segments) for constraint in constraints)
    print(f"Score: {score}")
    print(f"Constraints satisfied: {constraints_satisfied}")



if __name__ == "__main__":
    basic_state_space_search_test()
