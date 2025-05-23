############################################
# DISCLAIMER
# This code in its current  state might not be runnable as it was not updated thoroughly after every refactoring.
# It is meant to be used as a reference for the experiments conducted in the thesis.
############################################
import os
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
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
from src.util import total_satisfaction
from ilp_experiments import run_test_all_approaches
from src.constraint_generator import ConstraintGenerator


def run_state_space_search_experiment(solver, items, segments, constraints, N, vervose=False):
    start = time.perf_counter()
    solution = solver.solve(items, segments, constraints, N)
    elapsed = (time.perf_counter() - start) * 1000
    score = sum(items[item] for item in solution.values())
    constraints_satisfied = all(constraint.check_constraint(solution, items, segments) for constraint in constraints)
    satisfaction_score = total_satisfaction(solution, items, segments, constraints)
    if vervose:
        print(f"Score: {score}")
        print(f"Constraints satisfied: {constraints_satisfied}")
        print(f"Satisfaction score: {satisfaction_score}")
    return {"time": elapsed, "score": score, "satisfaction_score": satisfaction_score, "constraints_satisfied": constraints_satisfied}

def basic_state_space_search_test():
    # Test 1 N=10, M=100, S=10
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = {f'segment-{i}-{segmentation_property}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(10)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5, weight=0.9),
        MinSegmentsConstraint(segmentation_property, 2, 5, weight=0.9)
    ]
    N = 10
    solver = StateSpaceSolver()
    solution = solver.solve(items, segments, constraints, N)
    print(f"Solution: {solution}")
    score = sum(items[item] for item in solution.values())
    constraints_satisfied = all(constraint.check_constraint(solution, items, segments) for constraint in constraints)
    satisfaction_score = total_satisfaction(solution, items, segments, constraints)
    print(f"Score: {score}")
    print(f"Constraints satisfied: {constraints_satisfied}")
    print(f"Satisfaction score: {satisfaction_score}")

    # compare with ILP
    ilp_solver = IlpSolver()
    start_time = time.time()
    ilp_solution = ilp_solver.solve(items, segments, constraints, N)
    elapsed = time.time() - start_time
    ilp_score = sum(items[item] for item in ilp_solution.values())
    ilp_constraints_satisfied = all(constraint.check_constraint(ilp_solution, items, segments) for constraint in constraints)
    satisfaction_score = total_satisfaction(ilp_solution, items, segments, constraints)
    print(f"ILP Solution: {ilp_solution}")
    print(f"ILP Score: {ilp_score}")
    print(f"ILP Constraints satisfied: {ilp_constraints_satisfied}")
    print(f"ILP Satisfaction score: {satisfaction_score}")
    print(f"ILP elapsed time: {elapsed:.2f} seconds")

    # Test 2 N=20, M=300, S=30
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 301)}
    segments = {f'segment-{i}-{segmentation_property}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(30)}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 3, 8, weight=0.9),
        MinSegmentsConstraint(segmentation_property, 2, 5, weight=0.6),
        MaxSegmentsConstraint(segmentation_property, 4, 7, weight=0.9),
    ]
    N = 20
    solver = StateSpaceSolver(time_limit=10)
    solution = solver.solve(items, segments, constraints, N)
    print(f"Solution: {solution}")
    score = sum(items[item] for item in solution.values())
    constraints_satisfied = all(constraint.check_constraint(solution, items, segments) for constraint in constraints)
    satisfaction_score = total_satisfaction(solution, items, segments, constraints)
    print(f"Score: {score}")
    print(f"Constraints satisfied: {constraints_satisfied}")
    print(f"Satisfaction score: {satisfaction_score}")

    # compare with ILP
    ilp_solver = IlpSolver()
    start_time = time.time()
    ilp_solution = ilp_solver.solve(items, segments, constraints, N)
    elapsed = time.time() - start_time
    ilp_score = sum(items[item] for item in ilp_solution.values())
    ilp_constraints_satisfied = all(constraint.check_constraint(ilp_solution, items, segments) for constraint in constraints)
    satisfaction_score = total_satisfaction(ilp_solution, items, segments, constraints)
    print(f"ILP Solution: {ilp_solution}")
    print(f"ILP time: {elapsed:.2f} seconds")
    print(f"ILP Score: {ilp_score}")
    print(f"ILP Constraints satisfied: {ilp_constraints_satisfied}")
    print(f"ILP Satisfaction score: {satisfaction_score}")

    # ILP slicing solver
    preprocessor = ItemPreprocessor(verbose=False)
    ilp_solver = IlpSolver(verbose=False)
    start_time = time.time()
    ilp_solution = ilp_solver.solve_by_slicing(preprocessor, items, segments, constraints, N, slice_size=5)
    elapsed = time.time() - start_time
    ilp_score = sum(items[item] for item in ilp_solution.values())
    ilp_constraints_satisfied = all(constraint.check_constraint(ilp_solution, items, segments) for constraint in constraints)
    satisfaction_score = total_satisfaction(ilp_solution, items, segments, constraints)
    print(f"ILP Slicing Solution: {ilp_solution}")
    print(f"ILP Slicing time: {elapsed:.2f} seconds")
    print(f"ILP Slicing Score: {ilp_score}")
    print(f"ILP Slicing Constraints satisfied: {ilp_constraints_satisfied}")
    print(f"ILP Slicing Satisfaction score: {satisfaction_score}")


def compare_state_space_and_ilp():
    # Ns = [10]
    Ns = [10, 15, 20, 25, 30, 40]
    # Ms = [20, 40, 60, 100, 200, 300, 400, 500]
    Ms = [300]
    constraint_generator = ConstraintGenerator()
    segmentation_properties = ['test-prop1', 'test-prop2']

    state_space_solver = StateSpaceSolver(verbose=False, time_limit=10)
    ilp_solver = IlpSolver(verbose=False, time_limit=10)
    preprocessor = ItemPreprocessor(verbose=False)

    constraints_lists = [
        constraint_generator.generate_random_constraints(2, 10, None, None, segmentation_properties,
                                                         weight_type="soft",
                                                         exclude_specific=[ItemAtPositionConstraint,
                                                                           ItemFromSegmentAtPositionConstraint]),
        constraint_generator.generate_random_constraints(5, 10, None, None, segmentation_properties,
                                                         weight_type="soft",
                                                         exclude_specific=[ItemAtPositionConstraint,
                                                                           ItemFromSegmentAtPositionConstraint]),
        constraint_generator.generate_random_constraints(10, 10, None, None, segmentation_properties,
                                                         weight_type="soft",
                                                         exclude_specific=[ItemAtPositionConstraint,
                                                                           ItemFromSegmentAtPositionConstraint]),
    ]

    results = {}
    for N in Ns:
        for M in Ms:
            items_per_segment = 5
            S = M // items_per_segment
            items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M + 1)}

            segments1 = {f'segment-{i}-{segmentation_properties[0]}': Segment(f'segment-{i}', segmentation_properties[0], *list(items.keys())[i*items_per_segment:(i+1)*items_per_segment]) for i in range(S)}
            segments2 = {f'segment-1-{segmentation_properties[1]}': Segment('segment-1', segmentation_properties[1], *list(items.keys())[0::4]),
                         f'segment-2-{segmentation_properties[1]}': Segment('segment-2', segmentation_properties[1], *list(items.keys())[1::4]),
                         f'segment-3-{segmentation_properties[1]}': Segment('segment-3', segmentation_properties[1], *list(items.keys())[2::4]),
                         f'segment-4-{segmentation_properties[1]}': Segment('segment-4', segmentation_properties[1], *list(items.keys())[3::4])}
            segments = {**segments1, **segments2}
            for constraints in constraints_lists:
                results_ilp = run_ilp_test_all_approaches("", ilp_solver, preprocessor, items, segments, constraints, N, M, [2, 5, 8],
                        verbose=False, run_normal=True)
                results_ss = run_state_space_search_experiment(state_space_solver, items, segments, constraints, N)
                res_key = f"N={N}, M={M}, S={S}, Constraints={constraints}"
                results[res_key] = {
                    "state_space": results_ss,
                    "ilp": results_ilp
                }
                print(f"Results for {res_key}:")
                print(f"State Space Search: {results_ss}")
                print(f"ILP: {results_ilp}")
                print("-" * 50)

                 # save results to file
                save_results_to_file(M, N, [str(c) for c in constraints], results_ilp, results_ss, "state_space_vs_ilp_results.csv")


def save_results_to_file(M, N, constraints, results_ILP, results_SS, filename):
    with open("state_space_vs_ilp_results.csv", "a") as f:
        # transform results to a pandas dataframe
        rows = []
        for solver, solver_result in results_ILP.items():
            if len(solver_result) == 0:
                continue
            if solver == "slicing":
                for slicing_solver, slicing_result in solver_result.items():
                    rows.append({"useridx": "test-user", "M": M, "N": N, "constraints": constraints, "solver": f"ilp-slicing-s={slicing_solver}",
                                 "time_ms": slicing_result["time"], "score": slicing_result["score"],
                                 "constraint_satisfaction": slicing_result["satisfaction_score"],
                                 "empty": slicing_result["satisfaction_score"] == 0})
            else:
                if solver == "normal":
                    name = "ilp"
                elif solver == "preprocessing":
                    name = "ilp-preprocessing"
                else:
                    name = solver
                rows.append({"useridx": "test-user","M": M, "N": N, "constraints": constraints, "solver": name,
                             "time_ms": solver_result["time"], "score": solver_result["score"],
                             "constraint_satisfaction": solver_result["satisfaction_score"],
                             "empty": solver_result["satisfaction_score"] == 0})
        rows.append({"useridx": "test-user","M": M, "N": N, "constraints": constraints, "solver": "StateSpace",
                     "time_ms": results_SS["time"], "score": results_SS["score"],
                     "constraint_satisfaction": results_SS["satisfaction_score"],
                     "empty": results_SS["satisfaction_score"] == 0})
        df = pd.DataFrame(rows)

        header_needed = (not os.path.exists(filename)  # write header we do the first write
                         or os.path.getsize(filename) == 0)
        df.to_csv(filename,
                      mode="a",
                      index=False,
                      header=header_needed)


if __name__ == "__main__":
    # basic_state_space_search_test()
    compare_state_space_and_ilp()
