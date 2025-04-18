############################################
# DISCLAIMER
# This code in its current  state might not be runnable as it was not updated thoroughly after every refactoring.
# It is meant to be used as a reference for the experiments conducted in the thesis.
############################################

import random
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle

from src.algorithms.ILP import IlpSolver
from src.algorithms.Preprocessor import ItemPreprocessor
from src.segmentation import Segment
from src.constraints import *
from src.util import *


def run_test(test_name, solver, items, segments, constraints, N, using_soft_constraints=False, already_recommended_items=None,
             partition_size=None, verbose=True, return_first_feasible=False):
    print(f"\n=== {test_name} ===")
    start_time = time.time()
    item_segment_map = {item_id: seg_id for seg_id, segment in segments.items() for item_id in segment}
    if partition_size is not None:
        recommended_items = solver.solve_by_partitioning(items, segments, constraints, N, partition_size=partition_size,
                                                         item_segment_map=item_segment_map, look_ahead=False)
    else:
        recommended_items = solver.solve(items, segments, constraints, N, already_recommended_items, return_first_feasible)
    print(f"Recommended Items: {recommended_items}")

    # Check constraints
    if recommended_items:
        all_constraints_satisfied = True
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments, already_recommended_items):
                print(f"Constraint {constraint} is not satisfied.")
                all_constraints_satisfied = False
        if all_constraints_satisfied or using_soft_constraints:
            print(f"All constraints are satisfied for {test_name}.")
            if verbose:
                print("Recommended Items:")
                if already_recommended_items:
                    for position, item_id in enumerate(already_recommended_items):
                        item_segments = [seg_id for seg_id, segment in segments.items() if item_id in segment]
                        print(f"Position {-len(already_recommended_items) + position}: {item_id} (Item segments: {item_segments})")
            total_score = 0
            for position, item_id in recommended_items.items():
                score = items[item_id]
                total_score += score
                item_segments = [seg_id for seg_id, segment in segments.items() if item_id in segment]
                if verbose:
                    print(f"Position {position}: {item_id} (Item segments: {item_segments} Score: {score:.1f})")
            print(f"Total Score: {total_score:.1f}")
    else:
        print(f"No solution found for {test_name}.")

    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"Elapsed time: {elapsed_time:.4f} milliseconds")

    return elapsed_time


def ILP_basic_test():
    solver = IlpSolver()

    # Define items with fixed scores
    items = {
        'item1': 9.0,
        'item2': 8.5,
        'item3': 8.0,
        'item4': 7.5,
        'item5': 7.0,
        'item6': 6.5,
        'item7': 6.0,
        'item8': 5.5,
        'item9': 5.0,
        'item10': 4.5,
        'item11': 4.0,
        'item12': 3.5,
        'item13': 3.0,
        'item14': 2.5,
        'item15': 2.0,
        'item16': 1.5,
        'item17': 1.0,
        'item18': 0.5,
        'item19': 0.0,
        'item20': 0.0
    }

    # Assign items to segments
    seg1 = Segment('genre1', 'genre', 'item1', 'item2', 'item3', 'item4', 'item5')
    seg2 = Segment('genre2', 'genre', 'item6', 'item7', 'item8', 'item9', 'item10')
    seg3 = Segment('genre3', 'genre', 'item11', 'item12', 'item13', 'item14', 'item15')
    seg4 = Segment('genre4', 'genre', 'item16', 'item17', 'item18', 'item19', 'item20')
    segments = {seg.id: seg for seg in [seg1, seg2, seg3, seg4]}

    # Test Case 1: Single Constraint
    N = 5
    constraints = [
        MinItemsPerSegmentConstraint(segment_id='genre2', min_items=2, window_size=N)
    ]
    run_test("Test Case 1", solver, items, segments, constraints, N)

    # Test Case 2: Multiple Constraints
    N = 7
    constraints = [
        ItemFromSegmentAtPositionConstraint(segment_id='genre1', position=1),
        MaxItemsPerSegmentConstraint(segment_id='genre3', max_items=2, window_size=N),
        ItemAtPositionConstraint(item_id='item6', position=3)
    ]
    run_test("Test Case 2", solver, items, segments, constraints, N)

    # Test Case 3: Multiple Constraints
    N = 5
    constraints = [
        ItemFromSegmentAtPositionConstraint(segment_id='genre1', position=1),
        MinItemsPerSegmentConstraint(segment_id='genre3', min_items=2, window_size=N),
        MaxItemsPerSegmentConstraint(segment_id='genre3', max_items=4, window_size=N),
        ItemAtPositionConstraint(item_id='item6', position=3),
        MaxItemsPerSegmentConstraint(segment_id='genre1', max_items=2, window_size=N)
    ]
    run_test("Test Case 3", solver, items, segments, constraints, N)

    # Test Case 4: Hard Constraints, use 100 candidate items for N=10
    items = {f'item-{i}': random.uniform(0, 10) for i in range(1, 101)}
    segment1 = Segment('segment1', 'genre', *list(items.keys())[:25])
    segment2 = Segment('segment2', 'genre', *list(items.keys())[75:])
    segment3 = Segment('segment3', 'genre', *list(items.keys())[25:75])
    segment4 = Segment('segment4', 'genre', *list(items.keys())[1::2])
    segment5 = Segment('segment5', 'genre', *list(items.keys())[:10])
    segments = {seg.id: seg for seg in [segment1, segment2, segment3, segment4, segment5]}

    N = 10
    constraints = [
        ItemFromSegmentAtPositionConstraint(segment_id='segment3', position=3),
        ItemAtPositionConstraint(item_id='item-50', position=5),

        MinItemsPerSegmentConstraint(segment_id='segment2', min_items=2, window_size=N),
        MinItemsPerSegmentConstraint(segment_id='segment5', min_items=2, window_size=N),

        MaxItemsPerSegmentConstraint(segment_id='segment2', max_items=2, window_size=N),
        MaxItemsPerSegmentConstraint(segment_id='segment3', max_items=3, window_size=N),
        MaxItemsPerSegmentConstraint(segment_id='segment4', max_items=3, window_size=N)
    ]
    run_test("Test Case 4", solver, items, segments, constraints, N)

    # Test Case 5: Hard Constraints, use 100 candidate items for N=10, sliding window constraints
    N = 10
    # reassign items to non-overlapping segments
    segment1 = Segment('segment1', 'genre', *list(items.keys())[:25])
    segment2 = Segment('segment2', 'genre', *list(items.keys())[25:50])
    segment3 = Segment('segment3', 'genre', *list(items.keys())[50:75])
    segment4 = Segment('segment4', 'genre', *list(items.keys())[75:])
    segments = {seg.id: seg for seg in [segment1, segment2, segment3, segment4]}
    constraints = [
        MinItemsPerSegmentConstraint(segment_id='segment1', min_items=2, window_size=4),
        MinItemsPerSegmentConstraint(segment_id='segment2', min_items=1, window_size=4),
        MinItemsPerSegmentConstraint(segment_id='segment3', min_items=2, window_size=N),
        MinItemsPerSegmentConstraint(segment_id='segment4', min_items=1, window_size=7),

        MaxItemsPerSegmentConstraint(segment_id='segment1', max_items=3, window_size=5),
        MaxItemsPerSegmentConstraint(segment_id='segment2', max_items=3, window_size=N),
        MaxItemsPerSegmentConstraint(segment_id='segment3', max_items=5, window_size=N),

    ]
    run_test("Test Case 5", solver, items, segments, constraints, N)

    # 1000 candidate items for N=50
    N = 50
    items = {f'item-{i}': random.uniform(0, 10) for i in range(1, 1001)}
    # 10 non overlapping segments
    segment_list = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*100:(i+1)*100]) for i in range(10)]
    segments = {seg.id: seg for seg in segment_list}
    min_window_constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=2, window_size=10) for i in range(10)
    ]
    max_window_constraints = [
        MaxItemsPerSegmentConstraint(segment_id=f'segment{i}', max_items=3, window_size=10) for i in range(10)
    ]
    random_constraints = random.choices(min_window_constraints + max_window_constraints, k=10)
    run_test("Test Case 6", solver, items, segments, random_constraints, N)

    # Test Case 7: 100 candidate items for N=10, soft constraints
    N = 10
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segment_list = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*10:(i+1)*10]) for i in range(10)]
    segments = {seg.id: seg for seg in segment_list}
    constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment0', min_items=1, window_size=5, weight=1.0),
        MinItemsPerSegmentConstraint(segment_id=f'segment1', min_items=1, window_size=5, weight=0.9),
        MinItemsPerSegmentConstraint(segment_id=f'segment2', min_items=1, window_size=5, weight=0.8),
        MinItemsPerSegmentConstraint(segment_id=f'segment3', min_items=1, window_size=5, weight=0.7),
        MinItemsPerSegmentConstraint(segment_id=f'segment4', min_items=1, window_size=5, weight=0.6),
        MinItemsPerSegmentConstraint(segment_id=f'segment5', min_items=1, window_size=5, weight=0.5),
    ]
    run_test("Test Case 7", solver, items, segments, constraints, N, using_soft_constraints=True)

    # Test Case 8: 100 candidate items for N=10, hard constraints - GlobalMinItemsPerSegmentConstraint
    N = 10
    items = {f'item-{i}': max(random.uniform(0, 1) - i*0.01, 0) for i in range(1, 101)} # Decreasing scores to check diversity
    segment_list = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*25:(i+1)*25]) for i in range(4)]
    segments = {seg.id: seg for seg in segment_list}
    constraints = [
        GlobalMinItemsPerSegmentConstraint(segmentation_property='test-prop', min_items=2, weight=1.0, window_size=N)
    ]
    run_test("Test Case 8", solver, items, segments, constraints, N) # should include 2 items from each segment and maximize score by including items from earlier segments

    # Test Case 9: 100 candidate items for N=10, hard constraints - GlobalMaxItemsPerSegmentConstraint
    N = 10
    items = {f'item-{i}': max(random.uniform(0, 1) - i*0.01, 0) for i in range(1, 101)} # Decreasing scores to check diversity
    segment_list = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*20:(i+1)*20]) for i in range(5)]
    segments = {seg.id: seg for seg in segment_list}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property='test-prop', max_items=2, weight=1.0, window_size=N)
    ]
    run_test("Test Case 9", solver, items, segments, constraints, N) # should include 2 items from each segment even if it reduces the total score

    # Test Case 10: Test MinSegmentDiversity simple
    N = 10
    items = {f'item-{i}': i*0.01 for i in range(1, 101)}
    segment_list = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*10:(i+1)*10]) for i in range(10)]
    segments = {seg.id: seg for seg in segment_list}
    constraints = [
        MinSegmentsConstraint(segmentation_property='test-prop', min_segments=2, window_size=3, weight=1.0)
    ]
    run_test("Test Case 10", solver, items, segments, constraints, N) # should

    # Test Case 11: Test MaxSegmentDiversity simple
    constraints = [
        MaxSegmentsConstraint(segmentation_property='test-prop', max_segments=3, window_size=5, weight=1.0)
    ]
    run_test("Test Case 11", solver, items, segments, constraints, N) # should fill everything with the most scoring segment



def ILP_time_efficiency(constraint_weight=1.0, use_preprocessing=False):
    solver = IlpSolver()
    num_recomms = [5, 10, 20, 50, 75, 100, 200, 300, 500] # N
    num_candidates = [10, 50, 100, 200, 500, 1000] # M
    num_constraints = [1, 2, 3, 4, 5, 8, 10, 15] # C
    results = dict() # (N, M, C) -> elapsed_time/elapsed_times

    for N in num_recomms:
        for M in num_candidates:
            if M < N:
                continue
            items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
            num_segments = 10
            segment_size = M // num_segments
            segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*segment_size:(i+1)*segment_size]) for i in range(num_segments)]
            available_constraints = [
                                        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=N//5,
                                                                     window_size=N, weight=constraint_weight) for i in range(num_segments)
                                    ] + [
                                        MaxItemsPerSegmentConstraint(segment_id=f'segment{i}', max_items=N//5+1,
                                                                     window_size=N, weight=constraint_weight) for i in range(num_segments)
                                    ] + [
                                        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=2,
                                                                     window_size=5, weight=constraint_weight) for i in range(num_segments)
                                    ] + [
                                        MaxItemsPerSegmentConstraint(segment_id=f'segment{i}', max_items=3,
                                                                     window_size=5, weight=constraint_weight) for i in range(num_segments)
                                    ]
            if use_preprocessing:
                C = 5
                constraints = random.choices(available_constraints, k=C)
                result = run_test_preprocessing(f"Test Case ({N}, {M}, {C})",
                                                                                                            solver, items, segments,constraints, N)
                results[(N, M, C)] = result
            else:
                for C in num_constraints:
                    constraints = random.choices(available_constraints, k=C)
                    elapsed_time = run_test(f"Test Case ({N}, {M}, {C})", solver, items, segments, constraints, N, using_soft_constraints=True)
                    results[(N, M, C)] = elapsed_time

    if not use_preprocessing:
        plot_results(results)
        plot_results_one_graph(results)
    else:
        plot_results_preprocessing(results)

def plot_results(results: dict):
    num_recomms = [5, 10, 20, 50, 100, 200, 500]  # N

    # Convert the results dictionary into a pandas DataFrame for easier manipulation
    data = []
    for (N, M, C), elapsed_time in results.items():
        data.append({'N': N, 'M': M, 'C': C, 'Time': elapsed_time})
    df = pd.DataFrame(data)

    # Iterate over each value of N and create a heatmap of Time vs M and C
    for N in num_recomms:
        df_N = df[df['N'] == N]
        if df_N.empty:
            continue  # Skip if there is no data for this N

        # Pivot the DataFrame to get M as rows, C as columns, and Time as values
        pivot_table = df_N.pivot(index='M', columns='C', values='Time')

        # Plot the heatmap with color bar label
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".4f",
            cmap='viridis_r',
            cbar_kws={'label': 'Elapsed Time (milliseconds)'}
        )

        # Add labels and title with time units
        plt.title(f'ILP Time Efficiency for N={N}')
        plt.xlabel('Number of Constraints (C)')
        plt.ylabel('Number of Candidates (M)')

        plt.tight_layout()
        plt.show()


# plot results NxM on x-axis, C=5, time on y-axis
def plot_results_one_graph(results: dict):
    # process results
    NM = []
    time = []
    for (N, M, C), elapsed_time in results.items():
        if C == 5:
            NM.append(N*M)
            time.append(elapsed_time)

    # plot
    plt.figure(figsize=(10, 8))
    plt.plot(NM, time, marker='o')
    plt.xlabel('N*M')
    plt.ylabel('Elapsed Time (milliseconds)')
    plt.title('ILP Time Efficiency')
    plt.tight_layout()
    plt.show()


# plot results NxM on x-axis, time on y-axis for different C (all in one graph)
# use blue for normal results, red for preprocessed results
def plot_results_preprocessing(results: dict):
    # process results
    NM = []
    time_differences = []
    average_times_N_preprocessed = {}
    num_filtered_data = []
    for (N, M, C), result in results.items():
        if N==M:    # skip N=M because filtering does not make sense in this case
            continue
        elapsed_time_preprocessing, elapsed_time_normal, num_filtered_items = result
        NM.append(N*M)
        time_differences.append(elapsed_time_preprocessing / elapsed_time_normal * 100)
        if N not in average_times_N_preprocessed:
            average_times_N_preprocessed[N] = []
        average_times_N_preprocessed[N].append(elapsed_time_preprocessing)
        num_filtered_data.append({'N': N, 'M': M, 'Filtered Items': num_filtered_items})

    # average times for each N
    average_times_N_preprocessed = {N: sum(times)/len(times) for N, times in average_times_N_preprocessed.items()}

    # plot percentage of time difference
    ticks = np.arange(0, max(time_differences) + 10, 10)
    plt.figure(figsize=(10, 8))
    plt.scatter(NM, time_differences, marker='o')
    plt.xlabel('N*M')
    plt.ylabel('Fraction Preprocessed/Original [%]')
    plt.title('ILP Preprocessing Time Efficiency')
    plt.yticks(ticks)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot average times for each N
    minor_ticks = np.arange(0, max(average_times_N_preprocessed.values()) + 200, 100)
    major_ticks = np.arange(0, max(average_times_N_preprocessed.values()) + 200, 500)
    plt.figure(figsize=(10, 8))
    plt.plot(list(average_times_N_preprocessed.keys()), list(average_times_N_preprocessed.values()), marker='o')
    plt.xlabel('N')
    plt.ylabel('Average Elapsed Time after Preprocessing (milliseconds)')
    plt.title('ILP Preprocessing Time Efficiency')
    plt.yticks(minor_ticks, minor=True)
    plt.yticks(major_ticks, minor=False)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # heatmap of number of filtered items for different combinations of N and M
    df = pd.DataFrame(num_filtered_data)
    pivot_table = df.pivot(index='M', columns='N', values='Filtered Items')
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".0f",
        cmap='viridis_r',
        cbar_kws={'label': 'Number of Filtered Items'}
    )
    plt.title('Number of Filtered Items by Preprocessing')
    plt.xlabel('N')
    plt.ylabel('M')
    plt.tight_layout()
    plt.show()


def items_preprocessing_basic_test():
    N = 10
    solver = IlpSolver()
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * 10:(i + 1) * 10]) for i in range(10)]
    segments_dict = {seg.id: seg for seg in segments}
    # item to segment id dict
    item_segment_map = {item_id: seg_id for seg_id, segment in segments_dict.items() for item_id in segment}
    constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment0', min_items=1, window_size=6, weight=1.0),
        MinItemsPerSegmentConstraint(segment_id=f'segment1', min_items=1, window_size=6, weight=0.9),
        MaxItemsPerSegmentConstraint(segment_id=f'segment2', max_items=2, window_size=5, weight=0.8),
        MaxItemsPerSegmentConstraint(segment_id=f'segment3', max_items=2, window_size=5, weight=0.7),
        MinItemsPerSegmentConstraint(segment_id=f'segment4', min_items=1, window_size=6, weight=0.6),
        MaxItemsPerSegmentConstraint(segment_id=f'segment5', max_items=2, window_size=5, weight=0.5),
    ]
    filtered_items, filtered_segments = solver.preprocess_items(items, segments_dict, segments_dict, constraints, item_segment_map, N)
    print(filtered_items)

    run_test("Test Case Preprocessed", solver, filtered_items, filtered_segments, constraints, N, using_soft_constraints=True)
    run_test("Test Case Normal", solver, items, segments, constraints, N, using_soft_constraints=True)

    N = 50
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 1001)}
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * 100:(i + 1) * 100]) for i in range(10)]
    segments_dict = {seg.id: seg for seg in segments}
    item_segment_map = {item_id: seg_id for seg_id, segment in segments_dict.items() for item_id in segment}
    constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment0', min_items=1, window_size=10, weight=1.0),
        MinItemsPerSegmentConstraint(segment_id=f'segment1', min_items=1, window_size=10, weight=0.9),
        MaxItemsPerSegmentConstraint(segment_id=f'segment2', max_items=2, window_size=5, weight=0.8),
        MaxItemsPerSegmentConstraint(segment_id=f'segment3', max_items=2, window_size=5, weight=0.7),
        MinItemsPerSegmentConstraint(segment_id=f'segment4', min_items=1, window_size=10, weight=0.6),
        MaxItemsPerSegmentConstraint(segment_id=f'segment5', max_items=2, window_size=5, weight=0.5),
        MaxItemsPerSegmentConstraint(segment_id=f'segment6', max_items=2, window_size=5, weight=0.4),
        MaxItemsPerSegmentConstraint(segment_id=f'segment7', max_items=2, window_size=5, weight=0.3),
        MaxItemsPerSegmentConstraint(segment_id=f'segment8', max_items=2, window_size=5, weight=0.2),
        MaxItemsPerSegmentConstraint(segment_id=f'segment9', max_items=2, window_size=5, weight=0.1),
    ]
    filtered_items, filtered_segments = solver.preprocess_items(items, segments_dict, segments_dict, constraints, item_segment_map, N)
    print(filtered_items)

    run_test("Test Case Preprocessed", solver, filtered_items, filtered_segments, constraints, N, using_soft_constraints=True)
    run_test("Test Case Normal", solver, items, segments, constraints, N, using_soft_constraints=True)

def run_test_preprocessing(test_name, solver, preprocessor, items, segments, constraints, N, using_soft_constraints=False, verbose=False,
                           preprocessing_only=False, return_first_feasible=False):
    segments_dict = {seg.id: seg for seg in segments}
    item_segment_map = create_item_segment_map_from_segments(segments_dict)

    print(f"\n=== {test_name} ===")
    start_time = time.time()
    filtered_items = preprocessor.preprocess_items(items, segments_dict, constraints, N)
    if verbose:
        print(f"Filtered Items: {filtered_items}")
    recommended_items = solver.solve(filtered_items, segments_dict, constraints, N, return_first_feasible=return_first_feasible)

    all_constraints_satisfied_preprocess = True
    total_score_preprocess = 0

    # Check constraints
    if recommended_items:
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments_dict):
                all_constraints_satisfied_preprocess = False
                print(f"Constraint {constraint} is not satisfied.")
        if all_constraints_satisfied_preprocess or using_soft_constraints:
            print(f"All constraints are satisfied for preprocessing test.")
        for position, item_id in recommended_items.items():
            score = items[item_id]
            total_score_preprocess += score
            item_segments = [seg.id for seg in segments if item_id in seg]
            if verbose:
                print(f"Position {position}: {item_id} (Item segments: {item_segments} Score: {score:.1f})")
        print(f"Total Score: {total_score_preprocess:.1f}")
    else:
        print(f"No solution found for {test_name}.")

    elapsed_time_preprocessing = (time.time() - start_time)*1000

    if preprocessing_only:
        print(f"Elapsed time for preprocessing test: {elapsed_time_preprocessing:.4f} milliseconds")
        return elapsed_time_preprocessing

    # Run the test with the original items and segments
    start_time = time.time()
    recommended_items = solver.solve(items, segments_dict, constraints, N, return_first_feasible=return_first_feasible)

    all_constraints_satisfied = True
    total_score = 0

    # Check constraints
    if recommended_items:
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments_dict):
                all_constraints_satisfied = False
                print(f"Constraint {constraint} is not satisfied.")
        if all_constraints_satisfied or using_soft_constraints:
            print(f"All constraints are satisfied for standard test.")
        for position, item_id in recommended_items.items():
            score = items[item_id]
            total_score += score
            item_segments = [seg.id for seg in segments if item_id in seg]
            if verbose:
                print(f"Position {position}: {item_id} (Item segments: {item_segments} Score: {score:.1f})")
        print(f"Total Score: {total_score:.1f}")
    else:
        print(f"No solution found for {test_name}.")

    elapsed_time = (time.time() - start_time)*1000
    print(f"Elapsed time for preprocessing test: {elapsed_time_preprocessing:.4f} milliseconds")
    print(f"Elapsed time for original test: {elapsed_time:.4f} milliseconds")
    print(f"Elapsed time difference: {elapsed_time - elapsed_time_preprocessing:.4f} milliseconds, score difference: {total_score - total_score_preprocess:.4f}")

    return elapsed_time_preprocessing, elapsed_time, len(filtered_items)


def ILP_solve_with_already_recommeded_items_test():
    solver = IlpSolver()
    N = 10
    items = {f'item-{i}': 1.0 - i*0.01 for i in range(1, 101)} # Decreasing scores to check diversity
    segments_list = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * 20:(i + 1) * 20]) for i in range(5)]
    segments = {seg.id: seg for seg in segments_list}
    constraints = [
        GlobalMaxItemsPerSegmentConstraint(segmentation_property='test-prop', max_items=2, weight=1.0, window_size=5)
    ]
    already_recommended_items = ['item-1', 'item-21', 'item-41', 'item-61', 'item-81', 'item-2', 'item-22', 'item-42', 'item-62', 'item-82']
    run_test("Test Case 1", solver, items, segments, constraints, N, already_recommended_items=already_recommended_items)

    already_recommended_items = ['item-1', 'item-2', 'item-21', 'item-22']
    run_test("Test Case 2", solver, items, segments, constraints, N, already_recommended_items=already_recommended_items)

    constraints = [
        GlobalMinItemsPerSegmentConstraint(segmentation_property='test-prop', min_items=1, weight=1.0, window_size=5)
    ]
    already_recommended_items = ['item-81', 'item-61', 'item-41', 'item-21', 'item-1']
    run_test("Test Case 3", solver, items, segments, constraints, N, already_recommended_items=already_recommended_items)

    constraints = [MaxSegmentsConstraint('test-prop', max_segments=3, window_size=3)]
    already_recommended_items = ['item-1', 'item-21', 'item-41']
    # should fill everything with the most scoring segment
    run_test("Test Case 4", solver, items, segments, constraints, N, already_recommended_items=already_recommended_items)

    constraints = [MinSegmentsConstraint('test-prop', min_segments=3, window_size=3)]
    already_recommended_items = ['item-1', 'item-21', 'item-41']
    # should fill everything with the 3 most scoring segments in order
    run_test("Test Case 5", solver, items, segments, constraints, N, already_recommended_items=already_recommended_items)



def ILP_partitioning_test():
    verbose = True
    solver = IlpSolver(verbose=verbose)
    num_items = 1000
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, num_items+1)}
    num_segments = 10
    segment_size = num_items // num_segments
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * segment_size:(i + 1) * segment_size]) for i in range(num_segments)]
    N = 100
    partition_size = 10
    constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=1, window_size=10) for i in range(num_segments)
    ]
    run_test_preprocessing("Test Case 1 preprocessing + normal", solver, items, segments, constraints, N, verbose=verbose)
    run_test("Test Case 1 partitioning", solver, items, segments, constraints, N, partition_size=partition_size, verbose=verbose)

    # Test Case 2: partition size smaller than window size
    constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=2, window_size=20) for i in range(num_segments)
    ] + [
        MaxItemsPerSegmentConstraint(segment_id=f'segment{i}', max_items=3, window_size=10) for i in range(num_segments)
    ]
    run_test_preprocessing("Test Case 2 preprocessing + normal", solver, items, segments, constraints, N, verbose=verbose)
    run_test("Test Case 2 partitioning", solver, items, segments, constraints, N, partition_size=partition_size, verbose=verbose)

    N = 200
    constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=1, window_size=20) for i in range(num_segments)
    ] + [
        MaxItemsPerSegmentConstraint(segment_id=f'segment{i}', max_items=1, window_size=5) for i in range(num_segments)
    ]
    run_test_preprocessing("Test Case 3 preprocessing + normal", solver, items, segments, constraints, N, verbose=verbose)
    run_test("Test Case 3 partitioning", solver, items, segments, constraints, N, partition_size=partition_size, verbose=verbose)

    # Try when min per window is smaller than amount of segments and segments have decreasing scores
    items = {"item-1": 1, "item-2": 2, "item-3": 3, "item-4": 4, "item-5": 5, "item-6": 6, "item-7": 7, "item-8": 8, "item-9": 9, "item-10": 10}
    segment1 = Segment('segment1', 'test-prop', 'item-1', 'item-2', 'item-3', 'item-4', 'item-5')
    segment2 = Segment('segment2', 'test-prop', 'item-6', 'item-7', 'item-8', 'item-9', 'item-10')
    segments = [segment1, segment2]
    N = 10
    constraints = [GlobalMinItemsPerSegmentConstraint(segmentation_property='test-prop', min_items=1, weight=1.0, window_size=3)]
    run_test_preprocessing("Test Case 4 preprocessing + normal", solver, items, segments, constraints, N, verbose=verbose)
    run_test("Test Case 4 partitioning", solver, items, segments, constraints, N, partition_size=3, verbose=verbose)

    items = {f"item-{i}": i for i in range(1, 31)}
    segment1 = Segment('segment1', 'test-prop', *list(items.keys())[:10])
    segment2 = Segment('segment2', 'test-prop', *list(items.keys())[10:20])
    segment3 = Segment('segment3', 'test-prop', *list(items.keys())[20:])
    segments = [segment1, segment2, segment3]
    N = 28
    constraints = [GlobalMinItemsPerSegmentConstraint(segmentation_property='test-prop', min_items=1, weight=1.0, window_size=5)]
    run_test_preprocessing("Test Case 5 preprocessing + normal", solver, items, segments, constraints, N, verbose=verbose)
    run_test("Test Case 5 partitioning", solver, items, segments, constraints, N, partition_size=3, verbose=verbose)

    # Test case 6 - 1000 items, 10 segments, 100 recomms, 10 partition size, mix Min and Max constraints
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 1001)}
    num_segments = 10
    segment_size = 100
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * segment_size:(i + 1) * segment_size]) for i in range(num_segments)]
    N = 100
    partition_size = 10
    constraints = [
        GlobalMinItemsPerSegmentConstraint(segmentation_property='test-prop', min_items=1, weight=1.0, window_size=15),
        GlobalMaxItemsPerSegmentConstraint(segmentation_property='test-prop', max_items=2, weight=1.0, window_size=15)
    ]
    run_test_preprocessing("Test Case 6 preprocessing + normal", solver, items, segments, constraints, N, verbose=verbose)
    run_test("Test Case 6a partitioning", solver, items, segments, constraints, N, partition_size=partition_size, verbose=verbose)

    # Test case 6b - try larger partition size
    partition_size = 20
    run_test("Test Case 6b partitioning", solver, items, segments, constraints, N, partition_size=partition_size, verbose=verbose)

    # Test case 6c - try smaller partition size
    partition_size = 5
    run_test("Test Case 6c partitioning", solver, items, segments, constraints, N, partition_size=partition_size, verbose=verbose)

    # Test case 6d - try partition size larger than window size
    partition_size = 30
    run_test("Test Case 6d partitioning", solver, items, segments, constraints, N, partition_size=partition_size, verbose=verbose)

    # Test case 7 - erroneous case for p in {10, 25, 30} - possible could not be solved because unless we order items in a specific
    # way (not greedily) there might not be a solution for next partition
    N = 50
    M = 100
    num_segments = 20
    constraints = [
        GlobalMinItemsPerSegmentConstraint('test-prop', 1, 25, weight=0.9),
        GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 30, weight=0.9)
    ]
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M + 1)}
    segment_size = M // num_segments
    segments = [Segment(f'segment{i}', 'test-prop',
                                       *list(items.keys())[i * segment_size:(i + 1) * segment_size]) for i in
                range(num_segments)]
    for p in [ 25, 30]:
        run_test(f"Test Case 7 partitioning p={p}", solver, items, segments, constraints, N, partition_size=p, verbose=verbose)


def ILP_partitioning_time_efficiency():
    num_recomms = [10, 50, 100, 200, 300]
    num_candidates = [200, 300, 500, 1000, 5000, 10000, 20000]
    partition_sizes = [5, 8, 10, 15, 20, 30]
    num_segments = 20
    results = dict()
    results_preprocessing = dict()
    solver = IlpSolver(verbose=False)

    for M in num_candidates:
        items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
        segment_size = M // num_segments
        segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * segment_size:(i + 1) * segment_size]) for i in range(num_segments)]

        for N in num_recomms:
            if M <= N:
                continue
            for p in partition_sizes:
                # test only MinDiversity constraint
                constraints = [
                    GlobalMinItemsPerSegmentConstraint('test-prop', 1, 25)
                ]
                try:
                    elapsed_time = run_test(f"Test Case N:{N}, M: {M}, p:{p}, MinOnly partitioning", solver, items, segments, constraints, N, partition_size=p, verbose=False)
                    results[(M, N, p, "MinOnly")] = elapsed_time
                except Exception as e:
                    print(f"Error in Test Case N:{N}, M: {M}, p:{p}, MinOnly partitioning: {e}")

                # test only MaxDiversity constraint
                constraints = [
                    GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 25)
                ]
                try:
                    elapsed_time = run_test(f"Test Case N:{N}, M: {M}, p:{p}, MaxOnly partitioning", solver, items, segments, constraints, N, partition_size=p, verbose=False)
                    results[(M, N, p, "MaxOnly")] = elapsed_time
                except Exception as e:
                    print(f"Error in Test Case N:{N}, M: {M}, p:{p}, MaxOnly partitioning: {e}")

                # test both MinDiversity and MaxDiversity constraints
                constraints = [
                    GlobalMinItemsPerSegmentConstraint('test-prop', 1, 25),
                    GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 25)
                ]
                try:
                    elapsed_time = run_test(f"Test Case N:{N}, M: {M}, p:{p}, Both partitioning", solver, items, segments, constraints, N, partition_size=p, verbose=False)
                    results[(M, N, p, "Both")] = elapsed_time
                except Exception as e:
                    print(f"Error in Test Case N:{N}, M: {M}, p:{p}, Both partitioning: {e}")

    # save results to a file
    with open('../results_ILP_partitioning_time_efficiency.txt', 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    with open('results_ILP_partitioning_time_efficiency_preprocessing_only.txt', 'w') as f:
        for key, value in results_preprocessing.items():
            f.write(f"{key}: {value}\n")

def plot_results_ILP_partitioning(results_file: str):
    # Load the results from the file
    results = dict()
    with open(results_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            M, N, P, constraint = key[1:-1].split(', ')
            M = int(M)
            N = int(N)
            P = int(P)
            constraint = constraint[1:-1]
            value = float(value)
            results[(M, N, P, constraint)] = value

    # Convert the results dictionary into a pandas DataFrame for easier manipulation
    data = []
    for (M, N, P, constraint), elapsed_time in results.items():
        data.append({'M': M, 'N': N, 'P': P, 'Constraint': constraint, 'Time': elapsed_time})
    df = pd.DataFrame(data)
    print(df)

    # Set a plotting style
    sns.set_style("whitegrid")

    # 1. Effect of Partition Size (P) for fixed M and N (no averaging here)
    fixed_M = 1000
    fixed_N = 100
    df_fixed = df[(df['M'] == fixed_M) & (df['N'] == fixed_N)]
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_fixed, x='P', y='Time', hue='Constraint', marker='o')
    plt.title(f"Effect of Partition Size (P) on Computation Time\n(M={fixed_M}, N={fixed_N})")
    plt.xlabel("Partition Size (P)")
    plt.ylabel("Time (milliseconds)")
    plt.tight_layout()
    plt.show()

    # 2. Effect of Number of Candidates (M) for a fixed N, averaged over all P
    # Fix N and average Time over P for each M and Constraint
    fixed_N = 100
    df_m = df[df['N'] == fixed_N].groupby(['M', 'Constraint'], as_index=False)['Time'].mean()
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_m, x='M', y='Time', hue='Constraint', marker='o')
    plt.title(f"Effect of Number of Candidates (M) on Computation Time (Averaged over P)\n(N={fixed_N})")
    plt.xlabel("Number of Candidates (M)")
    plt.ylabel("Average Time (milliseconds)")
    plt.tight_layout()
    plt.show()

    # 3. Effect of Number of Recommendations (N) for a fixed M, averaged over all P
    fixed_M = 1000
    df_n = df[df['M'] == fixed_M].groupby(['N', 'Constraint'], as_index=False)['Time'].mean()
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_n, x='N', y='Time', hue='Constraint', marker='o')
    plt.title(f"Effect of Number of Recommendations (N) on Computation Time (Averaged over P)\n(M={fixed_M})")
    plt.xlabel("Number of Recommendations (N)")
    plt.ylabel("Average Time (milliseconds)")
    plt.tight_layout()
    plt.show()


def ILP_solve_for_overlapping_segments():
    print("=============== Testing Functionality for Overlapping Segments ===============")

    solver = IlpSolver(verbose=False)
    preprocessor = ItemPreprocessor(verbose=True)

    items = {f'item-{i}': i for i in range(1, 101)}
    segment1 = Segment('segment1', 'test-prop', *list(items.keys())[:50])
    segment2 = Segment('segment2', 'test-prop', *list(items.keys())[25:75])
    segment3 = Segment('segment3', 'test-prop', *list(items.keys())[::2])
    segment4 = Segment('segment4', 'test-prop', *list(items.keys())[1::2])
    segments = [segment1, segment2, segment3, segment4]
    N = 10
    constraints = [
        GlobalMinItemsPerSegmentConstraint(segmentation_property='test-prop', min_items=1, weight=1.0, window_size=5)
    ]
    run_test_preprocessing("Test Case 1", solver, preprocessor, items, segments, constraints, N, verbose=True)

    items = {f'item-{i}': i for i in range(1, 21)}
    segment1 = Segment('segment1', 'test-prop', *list(items.keys())[5:])
    segment2 = Segment('segment2', 'test-prop', 'item-20', 'item-19', 'item-18', 'item-17', 'item-16')
    segment3 = Segment('segment3', 'test-prop', *list(items.keys())[::2])
    segment4 = Segment('segment4', 'test-prop', *list(items.keys())[1::2])
    N = 6
    constraints = [GlobalMaxItemsPerSegmentConstraint(segmentation_property='test-prop', max_items=3, window_size=N)]
    run_test_preprocessing("Test Case 2", solver, preprocessor, items, [segment1, segment2, segment3, segment4], constraints, N, verbose=True)


def ILP_2D_constraints_test():
    # Test 1 - see if 2D constraints work
    solver = IlpSolver(verbose=True)
    print("=============== Test 1 - 2D constraints ===============")
    N = 5
    items1 = {"item1": 20, "item2": 2, "item3": 3, "item4": 2, "item5": 5, "item6": 2, "item7": 7, "item8": 2, "item9": 9, "item10": 2}
    items2 = {"item1": 20, "item8": 2, "item9": 11, "item10": 2, "item11": 13, "item12": 2, "item13": 15, "item14": 2, "item15": 17, "item16": 2, "item17": 19}
    items3 = {"item1": 20, "item3": 3, "item5": 5, "item7": 7, "item9": 9, "item11": 2, "item13": 4, "item15": 6, "item17": 8, "item19": 10}
    items = [items1, items2, items3]
    constraints = [[], [], []]
    constraints2D = [ ItemUniqueness2D(width=3, height=2) ]

    result = solver.solve_2D_constraints(items, {}, constraints, constraints2D, N)

    # print solution
    for i in range(len(items)):
        print(f"Row {i+1}:", end=" ")
        for j in range(N):
            print(f"{result[i, j+1]}", end=" ")
        print()

    # check if the solution satisfies the constraints
    for constraint in constraints2D:
        if not constraint.check_constraint(result, len(items), N):
            print(f"Constraint {constraint} is not satisfied.")
        else:
            print(f"Constraint {constraint} is satisfied.")

    # Test 2 - mix 1D and 2D constraints
    print("=============== Test 2 - mix 1D and 2D constraints ===============")
    segment1 = Segment('segment1', 'test-property', "item1", "item3", "item5", "item7", "item9", "item11", "item13", "item15", "item17", "item19")
    segment2 = Segment('segment2', 'test-property', "item2", "item4", "item6", "item8", "item10", "item12", "item14", "item16")
    segments_id_dict = {segment1.id: segment1, segment2.id: segment2}

    constraints1 = [
        MinItemsPerSegmentConstraint(segment_id='segment1', min_items=2, window_size=5),
        MaxItemsPerSegmentConstraint(segment_id='segment1', max_items=4, window_size=5),
    ]
    constraints2 = [GlobalMinItemsPerSegmentConstraint(segmentation_property='test-property', min_items=1, weight=1.0, window_size=2)]
    constraints1D = [constraints1, constraints2, []]

    result = solver.solve_2D_constraints(items, segments_id_dict, constraints1D, constraints2D, N)

    # print solution
    for i in range(len(items)):
        print(f"Row {i+1}:", end=" ")
        for j in range(N):
            print(f"{result[i, j+1]}", end=" ")
        print()

    # check if the solution satisfies the constraints
    for constraint in constraints2D:
        if not constraint.check_constraint(result, len(items), N):
            print(f"Constraint {constraint} is not satisfied.")
        else:
            print(f"Constraint {constraint} is satisfied.")

    for i, constraints in enumerate(constraints1D):
        for constraint in constraints:
            if not constraint.check_constraint({j+1: result[i, j+1] for j in range(N)}, items[i], segments_id_dict):
                print(f"Constraint {constraint} is not satisfied.")
            else:
                print(f"Constraint {constraint} is satisfied.")

    # large amount of items test
    print("=============== Test 3 - large amount of items, mixed constraints ===============")
    N = 20
    items1 = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    items2 = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201, 2)}
    items3 = {f'item-{i}': random.uniform(0, 1) for i in range(0, 201, 2)}
    items4 = {f'item-{i}': random.uniform(0, 1) for i in range(1, 301, 3)}
    items = [items1, items2, items3, items4]
    segment1 = Segment('segment1', 'test-property', *list(items1.keys())[:50])
    segment2 = Segment('segment2', 'test-property', *list(items1.keys())[50:])
    segments = {segment1.id: segment1, segment2.id: segment2}
    constraints = [[GlobalMinItemsPerSegmentConstraint('test-property', 1, 3)], [], [], []]
    constraints2D = [ItemUniqueness2D(width=10, height=3)]

    start_time = time.time()
    result = solver.solve_2D_constraints(items, segments, constraints, constraints2D, N)
    elapsed_time = (time.time() - start_time)*1000
    print(f"Elapsed time for large test: {elapsed_time:.4f} milliseconds")

    # print solution
    for i in range(len(items)):
        print(f"Row {i+1}:", end=" ")
        for j in range(N):
            print(f"{result[i, j+1]}".ljust(10), end=" ")
        print()

    # check if the solution satisfies the constraints
    for constraint in constraints2D:
        if not constraint.check_constraint(result, len(items), N):
            print(f"Constraint {constraint} is not satisfied.")
        else:
            print(f"Constraint {constraint} is satisfied.")

    for i, constraints in enumerate(constraints):
        for constraint in constraints:
            if not constraint.check_constraint({j+1: result[i, j+1] for j in range(N)}, items[i], segments):
                print(f"Constraint {constraint} is not satisfied.")
            else:
                print(f"Constraint {constraint} is satisfied.")

def ILP_2D_constraints_test_preprocessing():
    solver = IlpSolver(verbose=False)

    # test preprocessing works for 2D item uniqueness constraint
    print("=============== Test 1 - item uniqueness only ===============")
    N = 10
    items1 = {f'item-{i}': i+1 for i in range(0, 15)}
    items2 = {f'item-{i}': 20 - i for i in range(5, 20)}
    items3 = {f'item-{i}': i+1 for i in range(0, 15)}
    items = [items1, items2, items3]
    constraints = [[], [], []]
    segments = {}
    item_segment_map = {}
    constraints2D = [ItemUniqueness2D(width=5, height=2)]
    filtered_items = solver.preprocess_items_2D(items, segments, constraints, constraints2D, N, item_segment_map)

    print(f"Filtered items lens: {[len(filtered) for filtered in filtered_items]}")
    for i, filtered in enumerate(filtered_items):
        print(f"Row {i+1}: {filtered}")

    start_time = time.time()
    result = solver.solve_2D_constraints(filtered_items, segments, constraints, constraints2D, N)
    elapsed_time = (time.time() - start_time) * 1000
    print(f"Elapsed time for large test: {elapsed_time:.4f} milliseconds")
    check_constraints_and_print_2D(result, items, segments, constraints, constraints2D, N)

    # test preprocessing works for 2D item uniqueness constraint with a large amount of items
    print("=============== Test 2 - item uniqueness only, large amount of items ===============")
    N = 20
    items1 = {f'item-{i}': random.uniform(0, 1) for i in range(0, 150)}
    items2 = {f'item-{i}': random.uniform(0, 1) for i in range(50, 200)}
    items3 = {f'item-{i}': random.uniform(0, 1) for i in range(0, 150)}
    items4 = {f'item-{i}': random.uniform(0, 1) for i in range(50, 200)}
    items = [items1, items2, items3, items4]
    constraints = [[], [], [], []]
    segments = {}
    item_segment_map = {}
    constraints2D = [ItemUniqueness2D(width=10, height=3)]
    filtered_items = solver.preprocess_items_2D(items, segments, constraints, constraints2D, N, item_segment_map)

    print(f"Filtered items lens: {[len(filtered) for filtered in filtered_items]}")
    start_time = time.time()
    result = solver.solve_2D_constraints(filtered_items, segments, constraints, constraints2D, N)
    elapsed_time = (time.time() - start_time) * 1000
    print(f"Elapsed time for preprocessing large test: {elapsed_time:.4f} milliseconds, score: {count_2D_score(result, items, N)}")
    check_constraints_and_print_2D(result, items, segments, constraints, constraints2D, N)
    start_time = time.time()
    result = solver.solve_2D_constraints(items, segments, constraints, constraints2D, N)
    elapsed_time = (time.time() - start_time) * 1000
    print(f"Elapsed time for large test without preprocessing: {elapsed_time:.4f} milliseconds, score: {count_2D_score(result, items, N)}")

    # test 3 - mix 1D and 2D constraints
    print("=============== Test 3 - mixed constraints, large amount of items ===============")
    N = 20
    items1 = {f'item-{i}': random.uniform(0, 1) for i in range(0, 400)}
    items2 = {f'item-{i}': random.uniform(0, 1) for i in range(200, 600)}
    items3 = {f'item-{i}': random.uniform(0, 1) for i in range(0, 400)}
    items4 = {f'item-{i}': random.uniform(0, 1) for i in range(200, 600)}
    items = [items1, items2, items3, items4]
    segment1 = Segment('segment1', 'test-property', *[f'item-{i}' for i in range(0, 600, 2)])
    segment2 = Segment('segment2', 'test-property', *[f'item-{i}' for i in range(1, 600, 2)])
    segments = {segment1.id: segment1, segment2.id: segment2}
    item_segment_map = {item_id: seg_id for seg_id, segment in segments.items() for item_id in segment}
    constraints1 = [GlobalMaxItemsPerSegmentConstraint(segmentation_property='test-property', max_items=3, window_size=5)]
    constraints2 = [GlobalMinItemsPerSegmentConstraint(segmentation_property='test-property', min_items=1, weight=1.0, window_size=2)]
    constraints3 = [ MinItemsPerSegmentConstraint(segment_id='segment1', min_items=2, window_size=5) ]
    constraints4 = [ MaxItemsPerSegmentConstraint(segment_id='segment1', max_items=1, window_size=5) ]
    constraints1D = [constraints1, constraints2, constraints3, constraints4]
    constraints2D = [ ItemUniqueness2D(width=10, height=2) ]
    filtered_items = solver.preprocess_items_2D(items, segments, constraints1D, constraints2D, N, item_segment_map)

    print(f"Filtered items lens: {[len(filtered) for filtered in filtered_items]}")
    start_time = time.time()
    result = solver.solve_2D_constraints(filtered_items, segments, constraints1D, constraints2D, N)
    elapsed_time = (time.time() - start_time) * 1000
    print(
        f"Elapsed time for preprocessing large test: {elapsed_time:.4f} milliseconds, score: {count_2D_score(result, items, N)}")
    check_constraints_and_print_2D(result, items, segments, constraints1D, constraints2D, N)
    start_time = time.time()
    result = solver.solve_2D_constraints(items, segments, constraints1D, constraints2D, N)
    elapsed_time = (time.time() - start_time) * 1000
    print(
        f"Elapsed time for large test without preprocessing: {elapsed_time:.4f} milliseconds, score: {count_2D_score(result, items, N)}")

def check_constraints_and_print_2D(result, items, segments, constraints, constraints2D, N):
    # print solution
    for i in range(len(items)):
        print(f"Row {i + 1}:", end=" ")
        for j in range(N):
            print(f"{result[i, j + 1]}".ljust(10), end=" ")
        print()

    # check if the solution satisfies the constraints
    for constraint in constraints2D:
        if not constraint.check_constraint(result, len(items), N):
            print(f"Constraint {constraint} is not satisfied.")
        else:
            print(f"Constraint {constraint} is satisfied.")

    for i, constraints in enumerate(constraints):
        for constraint in constraints:
            if not constraint.check_constraint({j + 1: result[i, j + 1] for j in range(N)}, items[i], segments):
                print(f"Constraint {constraint} is not satisfied.")
            else:
                print(f"Constraint {constraint} is satisfied.")

def count_2D_score(result, items, N):
    total_score = 0
    for i in range(len(items)):
        for j in range(N):
            item_id = result[i, j+1]
            if item_id:
                total_score += items[i][item_id]
    return total_score

def check_constraints(recommended_items, items, segments, constraints):
    all_constraints_satisfied = True
    for constraint in constraints:
        if not constraint.check_constraint(recommended_items, items, segments):
            all_constraints_satisfied = False
            break

    return all_constraints_satisfied


def run_test_all_approaches(test_name, solver, preprocessor, items, segments, constraints, N, M, partition_sizes: list,
                            verbose=False, run_normal=True):
    results = {"normal": dict(), "preprocessing": dict(), "preprocessing_first_feasible": dict(), "partitioning": dict(), "partitioning_look_ahead": dict()}
    if run_normal:
        start_time_normal = time.time()
        solution = solver.solve(items, segments, constraints, N)
        results["normal"]["time"] = (time.time() - start_time_normal)*1000
        results["normal"]["constraints_satisfied"] = check_constraints(solution, items, segments, constraints)
        results["normal"]["score"] = sum([items[item_id] for item_id in solution.values()])

    item_segment_map = {item_id: seg_id for seg_id, segment in segments.items() for item_id in segment}

    start_time_preprocessing = time.time()
    filtered_items = preprocessor.preprocess_items(items, segments, segments, constraints, item_segment_map, N)
    solution = solver.solve(filtered_items, segments, constraints, N)
    results["preprocessing"]["time"] = (time.time() - start_time_preprocessing)*1000
    results["preprocessing"]["constraints_satisfied"] = check_constraints(solution, items, segments, constraints)
    results["preprocessing"]["score"] = sum([items[item_id] for item_id in solution.values()])

    start_time_preprocessing = time.time()
    filtered_items = preprocessor.preprocess_items(items, segments, segments, constraints, item_segment_map, N)
    solution = solver.solve(filtered_items, segments, constraints, N, return_first_feasible=True)
    results["preprocessing_first_feasible"]["time"] = (time.time() - start_time_preprocessing)*1000
    results["preprocessing_first_feasible"]["constraints_satisfied"] = check_constraints(solution, items, segments, constraints)
    results["preprocessing_first_feasible"]["score"] = sum([items[item_id] for item_id in solution.values()])

    for p in partition_sizes:
        # try:
        start_time_partitioning = time.time()
        temp_item_segment_map = {item_id: seg_id for seg_id, segment in segments.items() for item_id in segment}
        # filtered_items = solver.preprocess_items(items, segments, segments, constraints, item_segment_map, N)
        solution = solver.solve_by_partitioning(preprocessor, items, segments, constraints, N, partition_size=p,
                                                      item_segment_map=temp_item_segment_map)
        results["partitioning"][f"{p}"] = dict()
        results["partitioning"][f"{p}"]["time"] = (time.time() - start_time_partitioning)*1000
        results["partitioning"][f"{p}"]["constraints_satisfied"] = check_constraints(solution, items, segments, constraints)
        results["partitioning"][f"{p}"]["score"] = sum([items[item_id] for item_id in solution.values()])
        # except Exception as e:
        #     print(f"ERROR: Test Case N:{N}, M: {M}, p:{p}: {e}")

        # solve with look ahead
        start_time_partitioning = time.time()
        temp_item_segment_map = {item_id: seg_id for seg_id, segment in segments.items() for item_id in segment}
        solution = solver.solve_by_partitioning(preprocessor, items, segments, constraints, N, partition_size=p,
                                                      item_segment_map=temp_item_segment_map, look_ahead=True)
        results["partitioning_look_ahead"][f"{p}"] = dict()
        results["partitioning_look_ahead"][f"{p}"]["time"] = (time.time() - start_time_partitioning)*1000
        results["partitioning_look_ahead"][f"{p}"]["constraints_satisfied"] = check_constraints(solution, items, segments, constraints)
        results["partitioning_look_ahead"][f"{p}"]["score"] = sum([items[item_id] for item_id in solution.values()])

    if verbose:
        print(f"\n=== {test_name} ===")
        if run_normal:
            print("Normal:")
            print(f"Time: {results['normal']['time']:.4f} ms")
            print(f"Constraints satisfied: {results['normal']['constraints_satisfied']}")
            print(f"Total score: {results['normal']['score']:.1f}")
        print("Preprocessing:")
        print(f"Time: {results['preprocessing']['time']:.4f} ms")
        print(f"Constraints satisfied: {results['preprocessing']['constraints_satisfied']}")
        print(f"Total score: {results['preprocessing']['score']:.1f}")
        print("Preprocessing + First Feasible:")
        print(f"Time: {results['preprocessing_first_feasible']['time']:.4f} ms")
        print(f"Constraints satisfied: {results['preprocessing_first_feasible']['constraints_satisfied']}")
        print(f"Total score: {results['preprocessing_first_feasible']['score']:.1f}")
        for key, value in results["partitioning"].items():
            print(f"Partition size: {key}")
            print(f"Time: {value['time']:.4f} ms")
            print(f"Constraints satisfied: {value['constraints_satisfied']}")
            print(f"Total score: {value['score']:.1f}")
    return results


# compare all 3 approaches (no preprocessing, preprocessing, preprocessing + partitioning) in terms of time efficiency
# and quality of the solution (total score)
def compare_ILP_approaches():
    # small recomm numbers
    num_recomms = [10, 20]
    num_candidates = [100, 200, 300, 500]
    partition_sizes = [5, 8]
    num_segments = 10
    results = dict()
    solver = IlpSolver(verbose=False)
    test_verbose = True
    for M in num_candidates:
        items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
        segment_size = M // num_segments
        segments = {f'segment{i}': Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * segment_size:(i + 1) * segment_size]) for i in range(num_segments)}

        for N in num_recomms:
            if M <= N:
                continue
            constraints = [
                GlobalMinItemsPerSegmentConstraint('test-prop', 1, 10, weight=0.9),
                GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 10, weight=0.9)
            ]
            results[(M, N)] = run_test_all_approaches(f"Test Case N:{N}, M: {M}", solver,
                                                             items, segments, constraints, N, M, partition_sizes, verbose=test_verbose)

    num_recomms = [50, 100, 200]
    num_candidates = [100, 200, 300, 500, 1000, 5000, 10000]
    partition_sizes = [5, 8, 10, 15, 20, 25, 30]
    num_segments = 20
    for M in num_candidates:
        items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
        segment_size = M // num_segments
        segments = {f'segment{i}': Segment(f'segment{i}', 'test-prop', *list(items.keys())[i * segment_size:(i + 1) * segment_size]) for i in range(num_segments)}

        for N in num_recomms:
            if M <= N:
                continue
            run_test_case_normal = True
            if N >= 200 or M >= 300:
                run_test_case_normal = False

            constraints = [
                GlobalMinItemsPerSegmentConstraint('test-prop', 1, 25, weight=0.9),
                GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 30, weight=0.9)
            ]
            results[(M, N)] = run_test_all_approaches(f"Test Case N:{N}, M: {M}", solver, items, segments,
                                                      constraints, N, M, partition_sizes, verbose=test_verbose, run_normal=run_test_case_normal)

    # save results to a file
    with open("../results_ILP_compare_approaches.pkl", "wb") as file:
        pickle.dump(results, file)

def plot_results_all_approaches(results_file: str):
    # Load the results from the file
    results = dict()
    with open(results_file, "rb") as file:
        results = pickle.load(file)

    # Convert the results dictionary into a pandas DataFrame for easier manipulation
    data = []
    for (M, N), results in results.items():
        for approach, result in results.items():
            if approach == "normal":
                if len(result) == 0:
                    continue
                data.append({'M': M, 'N': N, "p": None, 'Approach': 'Normal', 'Time': result['time'], 'ConstraintsSatisfied': result['constraints_satisfied'], 'Score': result['score']})
            elif approach == "preprocessing":
                data.append({'M': M, 'N': N, "p": None, 'Approach': 'Preprocessing', 'Time': result['time'], 'ConstraintsSatisfied': result['constraints_satisfied'], 'Score': result['score']})
            else:
                for partition_size, partition_result in result.items():
                    data.append({'M': M, 'N': N, "p": partition_size, 'Approach': f"Partitioning ({partition_size})", 'Time': partition_result['time'], 'ConstraintsSatisfied': partition_result['constraints_satisfied'], 'Score': partition_result['score']})
    df = pd.DataFrame(data)
    print(df)

    # Set a plotting style
    sns.set_style("whitegrid")

    # 1. Effect of Partition Size (P) on time and score
    # x axis is time, y axis is score, each point represents different value of p, and is annotated
    # plot for N=50 and M=500
    fixed_M = 500
    fixed_N = 50
    df_fixed = df[(df['M'] == fixed_M) & (df['N'] == fixed_N)]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_fixed, x='Time', y='Score', hue='Approach', style='Approach', markers=True)
    for i in range(len(df_fixed)):
        plt.text(df_fixed['Time'].iloc[i], df_fixed['Score'].iloc[i], df_fixed['p'].iloc[i])
    plt.title(f"Effect of Partition Size (P) on Time and Score\n(M={fixed_M}, N={fixed_N})")
    plt.xlabel("Time (milliseconds)")
    plt.ylabel("Total Score")
    plt.tight_layout()
    plt.show()


    # 2. Effect of partition size of constraint satisfaction
    # For each partition size plot the percentage of constraints satisfied (count for all N and M)
    df_partitioning = df[df['Approach'].str.contains("Partitioning")]
    df_partitioning = df_partitioning.groupby(['Approach'], as_index=False)['ConstraintsSatisfied'].mean() * 100
    # rename p to Partition Size and transform the values to not be repeated
    df_partitioning.rename(columns={'Approach': 'Partition Size'}, inplace=True)
    df_partitioning['Partition Size'] = df_partitioning['Partition Size'].apply(lambda x: x.split("(")[1].split(")")[0])
    print(df_partitioning)
    plt.figure(figsize=(16, 10))
    sns.barplot(data=df_partitioning, x='Partition Size', y='ConstraintsSatisfied')
    plt.title(f"Effect of Partition Size (P) on Constraints Satisfaction")
    plt.xlabel("Partition Size (P)")
    plt.ylabel("Constraints Satisfaction (%)")
    plt.show()

    # 3. Effect of N on time for different partition sizes, use fixed M = 500
    fixed_M = 500
    df_fixed = df[(df['M'] == fixed_M)]
    df_fixed = df_fixed[df_fixed['Approach'].str.contains("Partitioning")]
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_fixed, x='N', y='Time', hue='Approach', style='Approach', markers=True)
    plt.title(f"Effect of Number of Recommendations (N) on Time\n(M={fixed_M})")
    plt.xlabel("Number of Recommendations (N)")
    plt.ylabel("Time (milliseconds)")
    plt.tight_layout()
    plt.show()

    # 4. Effect of M on time for different partition sizes, use fixed N = 50
    fixed_N = 50
    df_fixed = df[(df['N'] == fixed_N)]
    df_fixed = df_fixed[df_fixed['Approach'].str.contains("Partitioning")]
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_fixed, x='M', y='Time', hue='Approach', style='Approach', markers=True)
    plt.title(f"Effect of Number of Candidates (M) on Time\n(N={fixed_N})")
    plt.xlabel("Number of Candidates (M)")
    plt.ylabel("Time (milliseconds)")
    plt.tight_layout()
    plt.show()


"""
Experiment with increasing size of candidatte itemes and number of recommendations
with bare ILP solver without any preprocessing or partitioning
"""
def basic_ILP_time_efficiency_test():
    # Increasing N
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segment1 = Segment('segment1', 'test-prop', *list(items.keys())[0:100:4])
    segment2 = Segment('segment2', 'test-prop', *list(items.keys())[1:100:4])
    segment3 = Segment('segment3', 'test-prop', *list(items.keys())[2:100:4])
    segment4 = Segment('segment4', 'test-prop', *list(items.keys())[3:100:4])
    segments = {segment1.id: segment1, segment2.id: segment2, segment3.id: segment3, segment4.id: segment4}
    solver = IlpSolver(verbose=False)

    results = dict()

    for N in [10, 12, 16, 20, 25, 30, 40, 50, 60, 70, 80, 90]:
        constraints = [
            GlobalMaxItemsPerSegmentConstraint('test-prop', 1, 4)
        ]
        time_elapsed = run_test(f"Test Case N:{N}, M: 100", solver, items, segments, constraints, N, verbose=False)
        results[N] = time_elapsed

    print(results)
    # plot results
    plt.figure(figsize=(8, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title("Time Efficiency of ILP Solver for Increasing Number of Recommendations")
    plt.xlabel("Number of Recommendations (N)")
    plt.ylabel("Time (milliseconds)")
    plt.tight_layout()
    plt.grid()
    plt.show()

    # Increasing M
    results = dict()
    for M in [40, 80, 120, 160, 200, 250, 300, 400, 500]:
        items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
        segment1 = Segment('segment1', 'test-prop', *list(items.keys())[0:M:4])
        segment2 = Segment('segment2', 'test-prop', *list(items.keys())[1:M:4])
        segment3 = Segment('segment3', 'test-prop', *list(items.keys())[2:M:4])
        segment4 = Segment('segment4', 'test-prop', *list(items.keys())[3:M:4])
        segments = {segment1.id: segment1, segment2.id: segment2, segment3.id: segment3, segment4.id: segment4}
        constraints = [
            GlobalMaxItemsPerSegmentConstraint('test-prop', 1, 4)
        ]
        time_elapsed = run_test(f"Test Case N:50, M: {M}", solver, items, segments, constraints, 20, verbose=False)
        results[M] = time_elapsed

    print(results)
    # plot results
    plt.figure(figsize=(8, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title("Time Efficiency of ILP Solver for Increasing Number of Candidates")
    plt.xlabel("Number of Candidates (M)")
    plt.ylabel("Time (milliseconds)")
    plt.tight_layout()
    plt.grid()
    plt.show()


def basic_segment_diversity_test():
    solver = IlpSolver(verbose=True)
    segmentation_property = 'test-prop'

    # Test Case 1 - segments with descending scores
    items = {f'item-{i}': i for i in range(1, 101)}
    segment1 = Segment('segment1', segmentation_property, *list(items.keys())[:20])
    segment2 = Segment('segment2', segmentation_property, *list(items.keys())[20:40])
    segment3 = Segment('segment3', segmentation_property, *list(items.keys())[40:60])
    segment4 = Segment('segment4', segmentation_property, *list(items.keys())[60:80])
    segment5 = Segment('segment5', segmentation_property, *list(items.keys())[80:100])
    segments = [segment1, segment2, segment3, segment4, segment5]

    N = 10
    constraints = [
        MinSegmentsConstraint(min_segments=2, window_size=5, segmentation_property=segmentation_property),
        MaxSegmentsConstraint(max_segments=4, window_size=5, segmentation_property=segmentation_property)
    ]
    run_test_preprocessing("Test Case 1", solver, items, segments, constraints, N, verbose=True)

    # Test Case 2 - items with random scores
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segment1 = Segment('segment1', segmentation_property, *list(items.keys())[:20])
    segment2 = Segment('segment2', segmentation_property, *list(items.keys())[20:40])
    segment3 = Segment('segment3', segmentation_property, *list(items.keys())[40:60])
    segment4 = Segment('segment4', segmentation_property, *list(items.keys())[60:80])
    segment5 = Segment('segment5', segmentation_property, *list(items.keys())[80:100])
    segments = [segment1, segment2, segment3, segment4, segment5]

    run_test_preprocessing("Test Case 2", solver, items, segments, constraints, N, verbose=True)

    # Test Case 3 - smaller scale test to debug
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 21)}
    segment1 = Segment('segment1', segmentation_property, *list(items.keys())[:5])
    segment2 = Segment('segment2', segmentation_property, *list(items.keys())[5:10])
    segment3 = Segment('segment3', segmentation_property, *list(items.keys())[10:15])
    segment4 = Segment('segment4', segmentation_property, *list(items.keys())[15:20])
    segments = [segment1, segment2, segment3, segment4]

    N = 6
    constraints = [
        MinSegmentsConstraint(min_segments=2, window_size=5, segmentation_property=segmentation_property),
        MaxSegmentsConstraint(max_segments=2, window_size=3, segmentation_property=segmentation_property)
    ]
    run_test_preprocessing("Test Case 3", solver, items, segments, constraints, N, verbose=True)

    # Test Case 4 - test
    items = {f'item-{i}': i for i in range(1, 101)}
    segments = [Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(10)]
    N = 10
    constraints = [ MaxSegmentsConstraint(max_segments=5, window_size=5, segmentation_property=segmentation_property) ]
    run_test_preprocessing("Test Case 4", solver, items, segments, constraints, N, verbose=True) # every recomm slot should be filled with the most scoring segment

    # Test Case 5 - test MaxSegmentsConstraint with randomly scored items
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = [Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(10)]
    N = 10
    constraints = [ MaxSegmentsConstraint(max_segments=2, window_size=5, segmentation_property=segmentation_property) ]
    run_test_preprocessing("Test Case 5", solver, items, segments, constraints, N, verbose=True)

    # Test Case 6 - test MaxSegmentsConstraint with randomly scored items and draconian constraint
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = [Segment(f'segment{i}', segmentation_property, *list(items.keys())[i * 10:(i + 1) * 10]) for i in
                range(10)]
    N = 10
    constraints = [MaxSegmentsConstraint(max_segments=1, window_size=10, segmentation_property=segmentation_property)]
    run_test_preprocessing("Test Case 6", solver, items, segments, constraints, N, verbose=True)

    # Test Case 7 - test MaxSegmentsConstraint with selected scores for items
    items = {f'item-{i}': i for i in range(1, 101)}
    # segments - each segment will have 10 items, item1 in segment 1, item2 in segment 2, ... item10 in segment 10, item11 in segment 1, ...
    segments = [Segment(f'segment{i}', segmentation_property, *list(items.keys())[i::10]) for i in range(10)]
    N = 10
    constraints = [MaxSegmentsConstraint(max_segments=2, window_size=5, segmentation_property=segmentation_property)]
    run_test_preprocessing("Test Case 7", solver, items, segments, constraints, N, verbose=True)


# compare all 3 approaches (no preprocessing, preprocessing, preprocessing + partitioning) in terms of time efficiency
# graph the effect of increasing N, M, constraint complexity and number of segments in candidate items
def compare_ILP_approaches_speed():
    # effect of increasing N
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segments = {f'segment{i}': Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*10:(i+1)*10]) for i in range(10)}
    solver = IlpSolver(verbose=False)
    # results_increasing_N = dict()
    # for N in [5, 10, 15, 20, 30, 40, 50]:
    #     constraints = [
    #         GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
    #         MinSegmentsConstraint(segmentation_property, 2, 5)
    #     ]
    #     results_increasing_N[N] = run_test_all_approaches(f"Test Case N:{N}, M: 100", solver, items, segments, constraints, N, 100, [10], verbose=False)
    #
    # # plot results
    # plt.figure(figsize=(8, 6))
    # # plot normal approach in blue
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['normal']['time'] for N in results_increasing_N], marker='o', label='Normal', color='blue')
    # # plot preprocessing approach in green
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['preprocessing']['time'] for N in results_increasing_N], marker='o', label='Preprocessing', color='green')
    # # plot partitioning approach in red
    # plt.plot(list(results_increasing_N.keys()), [results_increasing_N[N]['partitioning']['10']['time'] for N in results_increasing_N], marker='o', label='Partitioning (p=10)', color='red')
    #
    # plt.title("Time Efficiency of ILP Solver for Increasing Number of Recommendations.\n Using M=100, |S|=10, C={GlobalMaxItems, MinSegments}")
    # plt.xlabel("Number of Recommendations (N)")
    # plt.ylabel("Time (milliseconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()
    # plt.yticks(range(0, int(results_increasing_N[50]["normal"]["time"]+50), 50))
    # plt.show()
    #
    # # effect of increasing M
    # results_increasing_M = dict()
    # for M in [50, 100, 150, 200, 250, 300]:
    #     items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
    #     segments = {f'segment{i}': Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*(M//10):(i+1)*(M//10)]) for i in range(10)}
    #     constraints = [
    #         GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
    #         MinSegmentsConstraint(segmentation_property, 2, 5)
    #     ]
    #     results_increasing_M[M] = run_test_all_approaches(f"Test Case N:10, M: {M}", solver, items, segments, constraints, 20, M, [10], verbose=False)
    #
    # # plot results
    # plt.figure(figsize=(8, 6))
    # # plot normal approach in blue
    # plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['normal']['time'] for M in results_increasing_M], marker='o', label='Normal', color='blue')
    # # plot preprocessing approach in green
    # plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['preprocessing']['time'] for M in results_increasing_M], marker='o', label='Preprocessing', color='green')
    # # plot partitioning approach in red
    # plt.plot(list(results_increasing_M.keys()), [results_increasing_M[M]['partitioning']['10']['time'] for M in results_increasing_M], marker='o', label='Partitioning (p=10)', color='red')
    #
    # plt.title("Time Efficiency of ILP Solver for Increasing Number of Candidates.\n Using N=20, |S|=10, C={GlobalMaxItems, MinSegments}")
    # plt.xlabel("Number of Candidates (M)")
    # plt.ylabel("Time (milliseconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()
    # plt.yticks(range(0, int(results_increasing_M[250]["normal"]["time"]+50), 50))
    # plt.show()
    #
    # # effect of increasing number of segments in candidate items
    # results_increasing_S = dict()
    # M = 200
    # N = 20
    # for S in [5, 10, 15, 20, 25, 30, 50]:
    #     print(f"Running test for S={S}")
    #     items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
    #     segments = {f'segment{i}': Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*(M//S):(i+1)*(M//S)]) for i in range(S)}
    #     constraints = [
    #         GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
    #         MinSegmentsConstraint(segmentation_property, 2, 5)
    #     ]
    #     results_increasing_S[S] = run_test_all_approaches(f"Test Case N:{N}, M: {M}", solver, items, segments, constraints, N, M, [10], verbose=False)
    #
    # # plot results
    # plt.figure(figsize=(8, 6))
    # # plot normal approach in blue
    # plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['normal']['time'] for S in results_increasing_S], marker='o', label='Normal', color='blue')
    # # plot preprocessing approach in green
    # plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['preprocessing']['time'] for S in results_increasing_S], marker='o', label='Preprocessing', color='green')
    # # plot partitioning approach in red
    # plt.plot(list(results_increasing_S.keys()), [results_increasing_S[S]['partitioning']['10']['time'] for S in results_increasing_S], marker='o', label='Partitioning (p=10)', color='red')
    #
    # plt.title("Time Efficiency of ILP Solver for Increasing Number of Segments in Candidate Items.\n Using N=20, M=200, C={GlobalMaxItems, MinSegments}")
    # plt.xlabel("Number of Segments in Candidate Items (|S|)")
    # plt.ylabel("Time (milliseconds)")
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()
    # plt.yticks(range(0, int(results_increasing_S[25]["normal"]["time"]+50), 50))
    # plt.show()

    # effect of increasing complexity of constraints
    constraints1 = [ GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5) ]
    constraints2 = [ GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
                     MinSegmentsConstraint(segmentation_property, 2, 5)
    ]
    constraints3 = [ GlobalMaxItemsPerSegmentConstraint(segmentation_property, 2, 5),
                     GlobalMinItemsPerSegmentConstraint(segmentation_property, 1, 15),
                     MinSegmentsConstraint(segmentation_property, 2, 5),
    ]
    constraints4 = [ GlobalMaxItemsPerSegmentConstraint(segmentation_property, 2, 5),
                     GlobalMinItemsPerSegmentConstraint(segmentation_property, 1, 15),
                     MinSegmentsConstraint(segmentation_property, 2, 5),
                     MaxSegmentsConstraint(segmentation_property, 2, 3)
    ]
    constraints = [constraints1, constraints2, constraints3]
    results_increasing_C = dict()
    N = 20
    M = 200
    S = 10
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
    segments = {f'segment{i}': Segment(f'segment{i}', segmentation_property, *list(items.keys())[i*(M//S):(i+1)*(M//S)]) for i in range(S)}
    for i, constraint in enumerate(constraints):
        print(f"Running test for constraint {i+1}")
        results_increasing_C[i] = run_test_all_approaches(f"Test Case N:{N}, M: {M}", solver, items, segments, constraint, N, M, [10], verbose=False)

    # plot results
    plt.figure(figsize=(8, 6))
    # plot normal approach in blue
    plt.plot(range(len(constraints)), [results_increasing_C[i]['normal']['time'] for i in results_increasing_C], marker='o', label='Normal', color='blue')
    # plot preprocessing approach in green
    plt.plot(range(len(constraints)), [results_increasing_C[i]['preprocessing']['time'] for i in results_increasing_C], marker='o', label='Preprocessing', color='green')
    # plot partitioning approach in red
    plt.plot(range(len(constraints)), [results_increasing_C[i]['partitioning']['10']['time'] for i in results_increasing_C], marker='o', label='Partitioning (p=10)', color='red')

    plt.title("Time Efficiency of ILP Solver for Increasing Complexity of Constraints.\n Using N=20, M=200, |S|=10")
    plt.xlabel("Number of Constraints")
    plt.ylabel("Time (milliseconds)")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.yticks(range(0, int(results_increasing_C[2]["normal"]["time"]+50), 50))
    plt.show()


def ilp_return_first_feasible_test():
    # compare results of first feasible and optimal in terms of score and time
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 101)}
    segment1 = Segment('segment1', 'test-prop', *list(items.keys())[0:100:4])
    segment2 = Segment('segment2', 'test-prop', *list(items.keys())[1:100:4])
    segment3 = Segment('segment3', 'test-prop', *list(items.keys())[2:100:4])
    segment4 = Segment('segment4', 'test-prop', *list(items.keys())[3:100:4])
    segments = [segment1, segment2, segment3, segment4]
    solver = IlpSolver(verbose=False)
    N = 20
    constraints = [
        GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 5),
        MinSegmentsConstraint('test-prop', 2, 4)
    ]
    run_test_preprocessing("Test Case 1 First Feasible", solver, items, segments, constraints, N, verbose=True, return_first_feasible=True)
    run_test_preprocessing("Test Case 1 Optimal", solver, items, segments, constraints, N, verbose=True)

    # test case 2
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201)}
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*20:(i+1)*20]) for i in range(10)]
    N = 20
    constraints = [
        GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 10),
        MinSegmentsConstraint('test-prop', 2, 4)
    ]
    run_test_preprocessing("Test Case 2 First Feasible", solver, items, segments, constraints, N, verbose=True, return_first_feasible=True)
    run_test_preprocessing("Test Case 2 Optimal", solver, items, segments, constraints, N, verbose=True)

    constraints = [GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 10)]
    run_test_preprocessing("Test Case 2 First Feasible", solver, items, segments, constraints, N, verbose=True, return_first_feasible=True)
    run_test_preprocessing("Test Case 2 Optimal", solver, items, segments, constraints, N, verbose=True)

def ILP_timeout_test():
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201)}
    # 2 segmentation properties
    segments1 = {f'segment-{i}': Segment(f'segment-{i}', 'test-prop1', *list(items.keys())[i*20:(i+1)*20]) for i in range(10)}
    # segments 2 -> divide into 2 segments - odd and even items
    segments2 = {'segment-even': Segment('segment-even', 'test-prop2', *list(items.keys())[::2]),
                 'segment-odd': Segment('segment-odd', 'test-prop2', *list(items.keys())[1::2])}
    segments = {**segments1, **segments2}
    N = 20
    M = 200
    constraints = [
        GlobalMaxItemsPerSegmentConstraint('test-prop1', 2, 10, weight=0.9),
        MinSegmentsConstraint('test-prop2', 2, 4, weight=0.9)
    ]
    solver1 = IlpSolver(verbose=True, time_limit=0.1)
    solver2 = IlpSolver(verbose=True, time_limit=2)

    recomms1 = solver1.solve(items, segments, constraints, N)
    recomms2 = solver2.solve(items, segments, constraints, N)

    score1 = sum([items[item_id] for item_id in recomms1.values()])
    score2 = sum([items[item_id] for item_id in recomms2.values()])

    print(f"Score 1: {score1}, Score 2: {score2}")

def ILP_num_threads_test():
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, 201)}
    segments = {f'segment-{i}': Segment(f'segment-{i}', 'test-prop', *list(items.keys())[i*20:(i+1)*20]) for i in range(10)}
    N = 20
    constraints = [
        GlobalMaxItemsPerSegmentConstraint('test-prop', 2, 10),
        MinSegmentsConstraint('test-prop', 2, 4)
    ]
    solver = IlpSolver(verbose=True)
    results = dict()
    for num_threads in [1, 2, 4, 8]:
        print(f"Running test with {num_threads} threads")
        start = time.time()
        recomms = solver.solve(items, segments, constraints, N, num_threads=num_threads)
        time_elapsed = (time.time() - start) * 1000
        results[num_threads] = time_elapsed
        score = sum([items[item_id] for item_id in recomms.values()])
        print(f"Score: {score}")
        print(f"Time elapsed: {time_elapsed:.4f} ms")



if __name__ == "__main__":
    # main()
    # ILP_time_efficiency()
    # ILP_time_efficiency(constraint_weight=0.9)
    # ILP_time_efficiency(constraint_weight=0.9, use_preprocessing=True)
    # ILP_basic_test()
    # ILP_solve_with_already_recommeded_items_test()
    # ILP_partitioning_test()
    # ILP_partitioning_time_efficiency()
    # plot_results_ILP_partitioning('results_ILP_partitioning_time_efficiency.txt')
    # ILP_2D_constraints_test()
    ILP_solve_for_overlapping_segments()
    # compare_ILP_approaches()
    # basic_ILP_time_efficiency_test()
    # plot_results_all_approaches('results_ILP_compare_approaches.pkl')
    # ILP_2D_constraints_test_preprocessing()
    # basic_segment_diversity_test()
    # compare_ILP_approaches_speed()
    # ilp_return_first_feasible_test()
    # ILP_timeout_test()
    # ILP_num_threads_test()
