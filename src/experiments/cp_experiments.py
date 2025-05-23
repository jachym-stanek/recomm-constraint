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
from src.constraints import *
from ilp_experiments import run_test_preprocessing as run_ilp_test_preprocessing
from ilp_experiments import run_test_all_approaches as run_ilp_test_all_approaches
from dfs_experiments import run_test_idfs
from src.constraint_generator import ConstraintGenerator
from src.util import check_solution


def print_test_results(test_name, results):
    print(f"\n=== {test_name} ===")
    print("--- Preprocessing optimal ---")
    print(f"Time: {results['preprocessing_optimal']['time']:.4f} milliseconds")
    print(f"Score: {results['preprocessing_optimal']['score']:.1f}")
    print(f"Constraints satisfied: {results['preprocessing_optimal']['constraints_satisfied']}")
    print("--- Preprocessing first feasible ---")
    print(f"Time: {results['preprocessing_first_feasible']['time']:.4f} milliseconds")
    print(f"Score: {results['preprocessing_first_feasible']['score']:.1f}")
    print(f"Constraints satisfied: {results['preprocessing_first_feasible']['constraints_satisfied']}")
    print("--- Optimal ---")
    print(f"Time: {results['optimal']['time']:.4f} milliseconds")
    print(f"Score: {results['optimal']['score']:.1f}")
    print(f"Constraints satisfied: {results['optimal']['constraints_satisfied']}")
    print("--- First feasible ---")
    print(f"Time: {results['first_feasible']['time']:.4f} milliseconds")
    print(f"Score: {results['first_feasible']['score']:.1f}")
    print(f"Constraints satisfied: {results['first_feasible']['constraints_satisfied']}")
    print("--- Permutation Optimal ---")
    print(f"Time: {results['permutation_optimal']['time']:.4f} milliseconds")
    print(f"Score: {results['permutation_optimal']['score']:.1f}")
    print(f"Constraints satisfied: {results['permutation_optimal']['constraints_satisfied']}")

def run_test_cp_preprocessing(test_name, cp_solver, preprocessor, items, segments, constraints, N, M, S, verbose=False, preprocessing_only=False):
    results = {"preprocessing_optimal": dict(), "preprocessing_first_feasible": dict(), "optimal": dict(), "first_feasible": dict(),
               "permutation_optimal": dict()}

    filtered_items = preprocessor.preprocess_items(items, segments, constraints, N)
    if verbose:
        print(f"Filtered Items: {filtered_items}")

    start_time = time.time()
    recommended_items_optimal = cp_solver.solve(filtered_items, segments, constraints, N)
    elapsed_time_preprocessing_optimal = (time.time() - start_time) * 1000
    score, constraints_satisfied = check_solution("Preprocessing optimal", constraints, recommended_items_optimal, items, segments, verbose)
    results["preprocessing_optimal"]["time"] = elapsed_time_preprocessing_optimal
    results["preprocessing_optimal"]["score"] = score
    results["preprocessing_optimal"]["constraints_satisfied"] = constraints_satisfied

    start_time = time.time()
    recommended_items_first_feasible = cp_solver.solve(filtered_items, segments, constraints, N, return_first_feasible=True)
    elapsed_time_preprocessing_first_feasible = (time.time() - start_time) * 1000
    score, constraints_satisfied = check_solution("Preprocessing first feasible", constraints, recommended_items_first_feasible, items, segments, verbose)
    results["preprocessing_first_feasible"]["time"] = elapsed_time_preprocessing_first_feasible
    results["preprocessing_first_feasible"]["score"] = score
    results["preprocessing_first_feasible"]["constraints_satisfied"] = constraints_satisfied

    if preprocessing_only:
        return results

    # Run the test with the original items and segments
    start_time = time.time()
    recommended_items_optimal = cp_solver.solve(items, segments, constraints, N)
    elapsed_time_optimal = (time.time() - start_time) * 1000
    score, constraints_satisfied = check_solution("Optimal", constraints, recommended_items_optimal, items, segments, verbose)
    results["optimal"]["time"] = elapsed_time_optimal
    results["optimal"]["score"] = score
    results["optimal"]["constraints_satisfied"] = constraints_satisfied

    start_time = time.time()
    recommended_items_first_feasible = cp_solver.solve(items, segments, constraints, N, return_first_feasible=True)
    elapsed_time_first_feasible = (time.time() - start_time) * 1000
    score, constraints_satisfied = check_solution("First feasible", constraints, recommended_items_first_feasible, items, segments, verbose)
    results["first_feasible"]["time"] = elapsed_time_first_feasible
    results["first_feasible"]["score"] = score
    results["first_feasible"]["constraints_satisfied"] = constraints_satisfied

    # run permutation cp solver
    cp_permutation_solver = PermutationCpSolver(items, segments, constraints, N)
    start_time = time.time()
    recommended_items_optimal = cp_permutation_solver.solve_optimal()
    elapsed_time_optimal = (time.time() - start_time) * 1000
    score, constraints_satisfied = check_solution("Optimal", constraints, recommended_items_optimal[0], items, segments, verbose)
    results["permutation_optimal"]["time"] = elapsed_time_optimal
    results["permutation_optimal"]["score"] = score
    results["permutation_optimal"]["constraints_satisfied"] = constraints_satisfied

    return results


def basic_test_cp():
    ilp_solver = IlpSolver(verbose=False)
    cp_solver = CpSolver(verbose=False)
    preprocessor = ItemPreprocessor(verbose=False)

    # Test 1 N=10, M=100, S=10
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = { f"segment{i}-{segmentation_property}": Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(10) }
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
        MinSegmentsConstraint(segmentation_property, 2, 5)
    ]
    results_cp = run_test_cp_preprocessing("Test Case 1 CP", cp_solver, preprocessor, items, segments, constraints, 10, 100, 10, verbose=True)
    print_test_results("Test Case 1", results_cp)
    # run_ilp_test_preprocessing("Test Case 1", ilp_solver, preprocessor, items, segments, constraints, 10, False, verbose=False)

    # Test 2 N=20, M=200, S=20, 3 constraints
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201)}
    segments = { f"segment{i}-{segmentation_property}": Segment(f'segment{i}', segmentation_property, *list(items.keys())[i * 10:(i + 1) * 10]) for i in range(20) }
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
        MinSegmentsConstraint(segmentation_property, 2, 5),
        MaxSegmentsConstraint(segmentation_property, 9, 10)
    ]
    results_cp = run_test_cp_preprocessing("Test Case 2 CP", cp_solver, preprocessor, items, segments, constraints, 20, 200, 20, verbose=True)
    print_test_results("Test Case 2 CP", results_cp)
    # run_ilp_test_preprocessing("Test Case 2", ilp_solver, preprocessor, items, segments, constraints, 20, False, verbose=False)

    # Test 2 N=20, M=200, S=20, 4 random constraints
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201)}
    segments = { f"segment{i}-{segmentation_property}": Segment(f'segment{i}', segmentation_property, *list(items.keys())[i * 10:(i + 1) * 10]) for i in range(20) }
    generator = ConstraintGenerator()
    constraints = generator.generate_random_constraints(4, 200, list(items.keys()),
                                                        segments, [segmentation_property],
                                                        exclude_specific=[ItemAtPositionConstraint,
                                                                          ItemFromSegmentAtPositionConstraint])
    results_cp = run_test_cp_preprocessing("Test Case 3 CP", cp_solver, preprocessor, items, segments, constraints, 20, 200, 20, verbose=False)
    print_test_results("Test Case 3 CP", results_cp)
    # run_ilp_test_preprocessing("Test Case 3", ilp_solver, preprocessor, items, segments, constraints, 20, False, verbose=False)



def compare_ilp_and_cp():
    # effect of increasing N
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201)}
    segments_list = [Segment(f'segment{i}', segmentation_property, *list(items.keys())[i * 10:(i + 1) * 10])
                for i in range(20)]
    segments_dict = {seg.label: seg for seg in segments_list}
    solver = IlpSolver(verbose=False)
    cp_solver = CpSolver(verbose=False)
    preprocessor = ItemPreprocessor(verbose=False)
    results_increasing_N = dict()
    # for N in [5, 10, 15, 20, 30, 40, 50]:
    #     print(f"Running test for N={N}")
    #     constraints = [
    #         GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
    #         MinSegmentsConstraint(segmentation_property, 2, 5)
    #     ]
    #     results_increasing_N[N] = dict()
    #     # results_increasing_N[N]['idfs'] = run_test_idfs(f"Test Case N:{N}, M: 100", items,
    #     #                                                 segments_dict, constraints, N)
    #     results_increasing_N[N]['cp'] = run_test_cp_preprocessing(f"Test Case N:{N}, M: 100", cp_solver, preprocessor, items, segments_dict, constraints, N, 200, 20, verbose=False)
    #     results_increasing_N[N]['ilp'] = run_ilp_test_all_approaches(f"Test Case N:{N}, M: 100", solver, preprocessor, items.copy(), segments_dict, constraints, N, 200, [], verbose=False)
    #
    #
    # # plot results
    # plt.figure(figsize=(8, 6))
    # # plot preprocessing approach in green
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['ilp']['normal']['time'] for N in results_increasing_N], marker='o', label='ILP Optimal', color='green')
    # # plot preprocessing with first feasible
    # # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['ilp']['preprocessing_first_feasible']['time'] for N in results_increasing_N], marker='o', label='ILP Preprocessing First Feasible', color='blue')
    # # plot slicing approach in red
    # # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['ilp']['slicing']['10']['time'] for N in results_increasing_N], marker='o', label='ILP Slicing (s=10)', color='red')
    # # plot look ahead approach in cyan
    # # plt.plot(list(results_increasing_N.keys()),
    # #          [results_increasing_N[N]['ilp']['slicing_look_ahead']['10']['time'] for N in results_increasing_N],
    # #          marker='o', label='ILP Look Ahead', color='cyan')
    # # plot cp preprocessing optimal approach in orange
    # # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['cp']['preprocessing_optimal']['time'] for N in results_increasing_N], marker='o', label='CP Preprocessing Optimal', color='orange')
    # # plot cp preprocessing first feasible approach in purple
    # # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['cp']['preprocessing_first_feasible']['time'] for N in results_increasing_N], marker='o', label='CP Preprocessing First Feasible', color='purple')
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['cp']['first_feasible']['time'] for N in results_increasing_N], marker='o', label='CP First Feasible', color='purple')
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['cp']['optimal']['time'] for N in results_increasing_N], marker='o', label='CP Optimal', color='orange')
    # # plot idfs approach in black
    # # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['idfs']["normal"]['time'] for N in results_increasing_N], marker='o', label='IDFS', color='black')
    #
    # #plt.title("Time Efficiency of ILP and CP Solvers for Increasing Number of Recommendations.\n Using M=100, |S|=10, C={GlobalMaxItems, MinSegments}")
    # plt.xlabel("Number of Recommendations (N)")
    # plt.ylabel("Time (milliseconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()
    # # plt.yticks(range(0, int(results_increasing_N[50]['cp']['optimal']["time"]+50), 50))
    # plt.show()
    #
    # # separate plot for cp preprocessing optimal, cp preprocessing first feasible and ilp preprocessing optimal
    # plt.figure(figsize=(8, 6))
    # # plot preprocessing approach in green
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['ilp']['preprocessing']['time'] for N in results_increasing_N], marker='o', label='ILP Preprocessing Optimal', color='green')
    # # plot preprocessing with first feasible
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['ilp']['preprocessing_first_feasible']['time'] for N in results_increasing_N], marker='o', label='ILP Preprocessing First Feasible', color='blue')
    # # plot cp preprocessing optimal approach in orange
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['cp']['preprocessing_optimal']['time'] for N in results_increasing_N], marker='o', label='CP Preprocessing Optimal', color='orange')
    # # plot cp preprocessing first feasible approach in purple
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['cp']['preprocessing_first_feasible']['time'] for N in results_increasing_N], marker='o', label='CP Preprocessing First Feasible', color='purple')
    #
    # #plt.title(
    # #    "Time Efficiency of ILP and CP Solvers for Increasing Number of Recommendations.\n Using M=100, |S|=10, C={GlobalMaxItems, MinSegments}")
    # plt.xlabel("Number of Recommendations (N)")
    # plt.ylabel("Time (milliseconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()
    # # plt.yticks(range(0, int(results_increasing_N[50]['cp']['preprocessing_optimal']["time"] + 50), 50))
    # plt.show()

    # --- effect of increasing M ---
    print("--- Effect of Increasing M ---")
    results_increasing_M = dict()
    for M in [50, 100, 150, 200, 250, 300]:
        print(f"Running test for M={M}")
        items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
        segments = {f'segment{i}-{segmentation_property}': Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*(M//10):(i+1)*(M//10)]) for i in range(10)}
        constraints = [
            GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
            MinSegmentsConstraint(segmentation_property, 2, 5)
        ]
        results_increasing_M[M] = dict()
        # results_increasing_M[M]['idfs'] = run_test_idfs(f"Test Case N:20, M: {M}", items.copy(),
        #                                                 segments, constraints, N)
        results_increasing_M[M]['cp'] = run_test_cp_preprocessing(f"Test Case N:20, M: {M}", cp_solver, preprocessor, items, segments, constraints, 20, M, 20, verbose=True, preprocessing_only=False)
        results_increasing_M[M]['ilp'] = run_ilp_test_all_approaches(f"Test Case N:20, M: {M}", solver, preprocessor, items, segments, constraints, 20, M, [], verbose=True, run_normal=True)

    # plot results
    plt.figure(figsize=(8, 6))
    # plot preprocessing approach in green
    plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['ilp']['preprocessing']['time'] for M in results_increasing_M], marker='o', label='ILP Preprocessing  Optimal', color='green')
    # plot preprocessing with first feasible
    plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['ilp']['preprocessing_first_feasible']['time'] for M in results_increasing_M], marker='o', label='ILP Preprocessing First Feasible', color='blue')
    # plot slicing approach in red
    # plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['ilp']['slicing']['10']['time'] for M in results_increasing_M], marker='o', label='Slicing (s=10)', color='red')
    # plot look ahead approach in cyan
    # plt.plot(list(results_increasing_M.keys()),
    #          [results_increasing_M[M]['ilp']['slicing_look_ahead']['10']['time'] for M in results_increasing_M],
    #          marker='o', label='ILP Look Ahead', color='cyan')

    # plot cp preprocessing first feasible approach in purple
    plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['cp']['preprocessing_optimal']['time'] for M in results_increasing_M], marker='o', label='CP Preprocessing Optimal', color='orange')
    plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['cp']['preprocessing_first_feasible']['time'] for M in results_increasing_M], marker='o', label='CP Preprocessing First Feasible', color='purple')
    # plot idfs approach in black
    # plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['idfs']["normal"]['time'] for M in results_increasing_M], marker='o', label='IDFS', color='black')

    #plt.title("Time Efficiency of ILP and CP Solvers for Increasing Number of Candidates.\n Using N=20, |S|=10, C={GlobalMaxItems, MinSegments}")
    plt.xlabel("Number of Candidates (M)")
    plt.ylabel("Time (milliseconds)")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    # plt.yticks(range(0, int(results_increasing_M[300]['cp']['preprocessing_optimal']["time"]+50), 50))
    plt.show()

    # plot optimal approaches without preprocessing
    plt.figure(figsize=(8, 6))
    plt.plot(list(results_increasing_M.keys()),
             [results_increasing_M[M]['ilp']['normal']['time'] for M in results_increasing_M], marker='o',
             label='ILP Optimal', color='green')
    plt.plot(list(results_increasing_M.keys()),
             [results_increasing_M[M]['cp']['optimal']['time'] for M in results_increasing_M], marker='o',
             label='CP Optimal', color='orange')
    plt.plot(list(results_increasing_M.keys()),
             [results_increasing_M[M]['cp']['first_feasible']['time'] for M in results_increasing_M], marker='o',
             label='CP First Feasible', color='purple')

    plt.xlabel("Number of Candidates (M)")
    plt.ylabel("Time (milliseconds)")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    # plt.yticks(range(0, int(results_increasing_M[300]['cp']['preprocessing_optimal']["time"] + 50), 50))
    plt.show()

    # plot scores
    plt.figure(figsize=(8, 6))
    plt.plot(list(results_increasing_M.keys()),
             [results_increasing_M[M]['cp']['preprocessing_optimal']['score'] for M in results_increasing_M], marker='o',
             label='CP Preprocessing Optimal', color='orange')
    plt.plot(list(results_increasing_M.keys()),
             [results_increasing_M[M]['cp']['preprocessing_first_feasible']['score'] for M in results_increasing_M],
             marker='o', label='CP Preprocessing First Feasible', color='purple')
    plt.plot(list(results_increasing_M.keys()),
             [results_increasing_M[M]['ilp']['preprocessing']['score'] for M in results_increasing_M], marker='o',
             label='ILP Preprocessing  Optimal', color='green', linestyle='--')
    plt.plot(list(results_increasing_M.keys()),
             [results_increasing_M[M]['ilp']['preprocessing_first_feasible']['score'] for M in results_increasing_M],
             marker='o', label='ILP Preprocessing First Feasible', color='blue')

    plt.xlabel("Number of Candidates (M)")
    plt.ylabel("Solution Score")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    # plt.yticks(range(0, int(results_increasing_M[300]['cp']['preprocessing_optimal']["time"] + 50), 50))
    plt.show()


    # ---  effect of increasing number of segments in candidate items ---
    results_increasing_S = dict()
    M = 200
    N = 20
    solver = IlpSolver(verbose=False)
    for S in [5, 10, 15, 20, 25, 30, 40, 50]:
        print(f"Running test for S={S}")
        items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
        segments = {f'segment-{i}-{segmentation_property}': Segment(f'segment-{i}', segmentation_property, *list(items.keys())[i*(M//S):(i+1)*(M//S)]) for i in range(S)}
        constraints = [
            GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
            MinSegmentsConstraint(segmentation_property, 2, 5)
        ]
        results_increasing_S[S] = dict()
        # results_increasing_S[S]['idfs'] = run_test_idfs(f"Test Case N:{N}, M: {M}, S: {S}", items.copy(),
        #                                                 segments.copy(), constraints, N)
        results_increasing_S[S]['cp'] = run_test_cp_preprocessing(f"Test Case N:{N}, M: {M}, S: {S}", cp_solver, preprocessor, items, segments, constraints, N, M, S, verbose=False, preprocessing_only=True)
        results_increasing_S[S]['ilp'] = run_ilp_test_all_approaches(f"Test Case N:{N}, M: {M}, S: {S}", solver, preprocessor, items, segments, constraints, N, M, [], verbose=False, run_normal=False)

    # plot results
    plt.figure(figsize=(8, 6))
    # plot preprocessing approach in green
    plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['ilp']['preprocessing']['time'] for S in results_increasing_S], marker='o', label='ILP Preprocessing Optimal', color='green')
    # plot preprocessing with first feasible
    plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['ilp']['preprocessing_first_feasible']['time'] for S in results_increasing_S], marker='o', label='Preprocessing First Feasible', color='blue')
    # plot slicing approach in red
    # plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['ilp']['slicing']['5']['time'] for S in results_increasing_S], marker='o', label='Slicing (s=5)', color='red')
    # plot look ahead approach in cyan
    # plt.plot(list(results_increasing_S.keys()),
    #          [results_increasing_S[S]['ilp']['slicing_look_ahead']['5']['time'] for S in results_increasing_S],
    #          marker='o', label='ILP Look Ahead', color='cyan')
    # plot cp preprocessing first feasible approach in purple
    plt.plot(list(results_increasing_S.keys()),
             [results_increasing_S[S]['cp']['preprocessing_optimal']['time'] for S in results_increasing_S],
             marker='o', label='CP Preprocessing Optimal', color='orange')
    plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['cp']['preprocessing_first_feasible']['time'] for S in results_increasing_S], marker='o', label='CP Preprocessing First Feasible', color='purple')
    # plot idfs approach in black
    # plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['idfs']["normal"]['time'] for S in results_increasing_S], marker='o', label='IDFS', color='black')

    #plt.title("Time Efficiency of ILP and CP Solvers for Increasing Number of Segments\n Using N=20, M=200, C={GlobalMaxItems, MinSegments}")
    plt.xlabel("Number of Segments in Candidate Items (|S|)")
    plt.ylabel("Time (milliseconds)")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.yticks(range(0, int(results_increasing_S[50]['cp']['preprocessing_optimal']["time"]+50), 50))
    plt.show()


if __name__ == '__main__':
    compare_ilp_and_cp()
    # basic_test_cp()
