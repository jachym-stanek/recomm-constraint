############################################
# DISCLAIMER
# This code in its current  state might not be runnable as it was not updated thoroughly after every refactoring.
# It is meant to be used as a reference for the experiments conducted in the thesis.
############################################

import random
import time
import matplotlib.pyplot as plt

from src.algorithms.ILP import IlpSolver
from src.algorithms.InformedDFS import IdfsSolver
from src.algorithms.Preprocessor import ItemPreprocessor
from src.constraints import *
from ilp_experiments import run_test_all_approaches as run_ilp_test_all_approaches


def run_test_idfs(test_name, items, segments, constraints, N):
    results = {"preprocessing": dict(), "normal": dict()}

    # Normal branch with a fresh solver
    start_time = time.perf_counter()
    solver_no_pre = IdfsSolver()
    solution = solver_no_pre.solve(items, segments, constraints, N)
    solution_time = (time.perf_counter() - start_time) * 1000  # in ms
    score = sum(items[item] for item in solution)
    constraints_satisfied = all(constraint.check_constraint(solution, items, segments) for constraint in constraints)

    results["normal"]["score"] = score
    results["normal"]["time"] = solution_time
    results["normal"]["constraints_satisfied"] = constraints_satisfied

    # Preprocessing branch with a fresh solver
    item_preprocessor = ItemPreprocessor(verbose=False)
    start_time = time.perf_counter()
    item_segment_map = {}
    for seg_id, segment in segments.items():
        for item_id in segment:
            item_segment_map.setdefault(item_id, []).append(seg_id)

    filtered_items = item_preprocessor.preprocess_items(items, segments, constraints, N)
    solver_pre = IdfsSolver()
    solution = solver_pre.solve(filtered_items, segments, constraints, N)
    solution_time = (time.perf_counter() - start_time) * 1000  # convert to ms
    score = sum(items[item] for item in solution)
    constraints_satisfied = all(constraint.check_constraint(solution, items, segments) for constraint in constraints)

    results["preprocessing"]["score"] = score
    results["preprocessing"]["time"] = solution_time
    results["preprocessing"]["constraints_satisfied"] = constraints_satisfied

    return results

def basic_test_dfs():
    # Test 0 N=5, M=10, S=2
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 11)}
    segments = {f'segment-{i}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*5:(i+1)*5]) for i in range(2)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 2, 3),
        MinSegmentsConstraint(segmentation_property, 1, 2)
    ]
    dfs_solver = IdfsSolver()
    N = 5
    start_time = time.time()
    solution = dfs_solver.solve(items, segments, constraints, N)
    constraints_satisfied = all([constraint.check_constraint(solution, items, segments) for constraint in constraints])
    print("=== Test Case 0 ===")
    print(f"Solution: {solution}")
    print(f"Time: {(time.time() - start_time)*1000} ms")
    print(f"Constraints satisfied: {constraints_satisfied}")


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
    score = sum([items[item] for item in solution])
    print("\n=== Test Case 2 ===")
    print(f"Solution: {solution}")
    print(f"Time: {(time.time() - start_time)*1000} ms")
    print(f"Score: {score}")
    print(f"Constraints satisfied: {constraints_satisfied}")

    # compare to ILP
    preprocessor = ItemPreprocessor(verbose=False)
    solver = IlpSolver(verbose=False)
    segments_dict = {seg.id: seg for seg in segments.values()}
    results_ilp = run_ilp_test_all_approaches(f"Test Case N:{N}, M: 100", solver, preprocessor, items, segments_dict, constraints, N, 100, [10],
                                verbose=False)
    print(f"ILP: {results_ilp}")

    # try segments with decreasing scores
    segmentation_property = 'test-prop'
    items = {f'item-{i}': i for i in range(1, 41)}
    segments = {f'segment-{i}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(4)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 3, 4),
    ]
    N = 40
    dfs_solver = IdfsSolver()
    start_time = time.time()
    solution = dfs_solver.solve(items, segments, constraints, N)
    constraints_satisfied = all([constraint.check_constraint(solution, items, segments) for constraint in constraints])
    score = sum([items[item] for item in solution])
    print("\n=== Test Case 3 ===")
    print(f"Solution: {solution}")
    print(f"Time: {(time.time() - start_time)*1000} ms")
    print(f"Score: {score}")
    print(f"Constraints satisfied: {constraints_satisfied}")

    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments_list = [Segment(f'segment{i}', segmentation_property, *list(items.keys())[i * 10:(i + 1) * 10])
                     for i in range(10)]
    segments_dict = {seg.id: seg for seg in segments_list}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
        MinSegmentsConstraint(segmentation_property, 2, 5)
    ]
    N = 50
    start_time = time.perf_counter()
    dfs_solver = IdfsSolver()
    solution = dfs_solver.solve(items, segments, constraints, N)
    solution_time = (time.perf_counter() - start_time) * 1000  # convert to ms
    constraints_satisfied = all([constraint.check_constraint(solution, items, segments) for constraint in constraints])
    score = sum([items[item] for item in solution])
    print("\n=== Test Case 4 ===")
    print(f"Solution: {solution}")
    print(f"Time: {solution_time} ms")
    print(f"Score: {score}")
    print(f"Constraints satisfied: {constraints_satisfied}")

    results = run_test_idfs(f"Test Case N:{N}, M: 100", items, segments_dict, constraints, N)
    print(f"IDFS time with run_test: {results['normal']['time']} ms")

    # Test 5 N=20, M=500, S=50
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 501)}
    segments = {f'segment-{i}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(50)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
        MinSegmentsConstraint(segmentation_property, 2, 5)
    ]
    dfs_solver = IdfsSolver()
    N = 20
    start_time = time.time()
    solution = dfs_solver.solve(items, segments, constraints, N)
    constraints_satisfied = all([constraint.check_constraint(solution, items, segments) for constraint in constraints])
    print("\n=== Test Case 5 ===")
    print(f"Solution: {solution}")
    print(f"Time: {(time.time() - start_time)*1000} ms")
    print(f"Constraints satisfied: {constraints_satisfied}")


def idfs_speed_efficiency():
    # graph for increasing M
    results = dict()
    dfs_solver = IdfsSolver()
    N = 20

    for M in [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]:
        print(f"Running test for M={M}")
        segmentation_property = 'test-prop'
        items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
        segments = {f'segment-{i}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(M//20)}
        constraints = [
            GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
            MinSegmentsConstraint(segmentation_property, 14, 15)
        ]

        start_time = time.time()
        solution = dfs_solver.solve(items, segments, constraints, N)
        results[M] = (time.time() - start_time)*1000
        constraints_satisfied = all([constraint.check_constraint(solution, items, segments) for constraint in constraints])
        print(f"Constraints satisfied: {constraints_satisfied}, Time: {results[M]} ms")

    plt.plot(list(results.keys()), list(results.values()))
    plt.xlabel("M")
    plt.ylabel("Time (ms)")
    plt.title("IDFS Time vs M")
    plt.tight_layout()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    basic_test_dfs()
    idfs_speed_efficiency()
    # zkusit udelat poradne zateze (testy s nahodnymi constrainty, testy s nesplnitelnymi constrainty)
    # udelat nejake testy na soft constrainty

