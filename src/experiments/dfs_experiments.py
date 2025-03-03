import random
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
import pickle

from src.algorithms.ILP import IlpSolver
from src.algorithms.CP import CpSolver
from src.algorithms.InformedDFS import IdfsSolver
from src.segmentation import Segment
from src.constraints.constraint import *
from ilp_experiments import run_test as run_ilp_test
from ilp_experiments import run_test_preprocessing as run_ilp_test_preprocessing
from ilp_experiments import run_test_all_approaches as run_ilp_test_all_approaches


def basic_test_dfs():
    # Test 1 N=10, M=100, S=10
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = {f'segment-{i}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(10)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
        MinSegmentsConstraint(segmentation_property, 2, 5)
    ]
    dfs_solver = IdfsSolver()
    N = 10
    start_time = time.time()
    solution = dfs_solver.solve(items, segments, constraints, N)
    constraints_satisfied = all([constraint.check_constraint(solution, items, segments) for constraint in constraints])
    print("=== Test Case 1 ===")
    print(f"Solution: {solution}")
    print(f"Time: {(time.time() - start_time)*1000} ms")
    print(f"Constraints satisfied: {constraints_satisfied}")

    # Test 2 N=20, M=200, S=20
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201)}
    segments = {f'segment-{i}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(20)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 3, 5),
        MinSegmentsConstraint(segmentation_property, 3, 6)
    ]
    dfs_solver = IdfsSolver()
    N = 20
    start_time = time.time()
    solution = dfs_solver.solve(items, segments, constraints, N)
    constraints_satisfied = all([constraint.check_constraint(solution, items, segments) for constraint in constraints])
    score = sum([items[item] for item in solution.values()])
    print("\n=== Test Case 2 ===")
    print(f"Solution: {solution}")
    print(f"Time: {(time.time() - start_time)*1000} ms")
    print(f"Score: {score}")
    print(f"Constraints satisfied: {constraints_satisfied}")

    # compare to ILP
    solver = IlpSolver(verbose=False)
    segments_dict = {seg.id: seg for seg in segments.values()}
    results_ilp = run_ilp_test_all_approaches(f"Test Case N:{N}, M: 100", solver, items, segments_dict, constraints, N, 100, [10],
                                verbose=False)
    print(f"ILP: {results_ilp}")


if __name__ == "__main__":
    basic_test_dfs()


