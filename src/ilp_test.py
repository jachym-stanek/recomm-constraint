import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from algorithms.ILP import ILP
from segmentation import Segment
from src.constraints.constraint import Constraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint, \
    ItemFromSegmentAtPositionConstraint, ItemAtPositionConstraint, SegmentationMinDiversity, SegmentationMaxDiversity


def run_test(test_name, solver, items, segments, constraints, N, using_soft_constraints=False):
    print(f"\n=== {test_name} ===")
    start_time = time.time()
    segments_id_dict = {seg.id: seg for seg in segments}
    recommended_items = solver.solve(items, segments, constraints, N)

    # Check constraints
    if recommended_items:
        all_constraints_satisfied = True
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments_id_dict):
                print(f"Constraint {constraint} is not satisfied.")
                all_constraints_satisfied = False
        if all_constraints_satisfied or using_soft_constraints:
            print(f"All constraints are satisfied for {test_name}.")
            print("Recommended Items:")
            total_score = 0
            for position, item_id in recommended_items.items():
                score = items[item_id]
                total_score += score
                item_segments = [seg.id for seg in segments if item_id in seg]
                print(f"Position {position}: {item_id} (Item segments: {item_segments} Score: {score:.1f})")
            print(f"Total Score: {total_score:.1f}")
    else:
        print(f"No solution found for {test_name}.")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    return elapsed_time * 1000  # Convert to milliseconds


def main():
    solver = ILP()

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
    segments = [seg1, seg2, seg3, seg4]

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
    segments = [segment1, segment2, segment3, segment4, segment5]

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
    segments = [segment1, segment2, segment3, segment4]
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
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*100:(i+1)*100]) for i in range(10)]
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
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*10:(i+1)*10]) for i in range(10)]
    constraints = [
        MinItemsPerSegmentConstraint(segment_id=f'segment0', min_items=1, window_size=5, weight=1.0),
        MinItemsPerSegmentConstraint(segment_id=f'segment1', min_items=1, window_size=5, weight=0.9),
        MinItemsPerSegmentConstraint(segment_id=f'segment2', min_items=1, window_size=5, weight=0.8),
        MinItemsPerSegmentConstraint(segment_id=f'segment3', min_items=1, window_size=5, weight=0.7),
        MinItemsPerSegmentConstraint(segment_id=f'segment4', min_items=1, window_size=5, weight=0.6),
        MinItemsPerSegmentConstraint(segment_id=f'segment5', min_items=1, window_size=5, weight=0.5),
    ]
    run_test("Test Case 7", solver, items, segments, constraints, N, using_soft_constraints=True)

    # Test Case 8: 100 candidate items for N=10, hard constraints - min diversity
    N = 10
    items = {f'item-{i}': max(random.uniform(0, 1) - i*0.01, 0) for i in range(1, 101)} # Decreasing scores to check diversity
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*25:(i+1)*25]) for i in range(4)]
    constraints = [
        SegmentationMinDiversity(segmentation_property='test-prop', min_items=2, weight=1.0)
    ]
    run_test("Test Case 8", solver, items, segments, constraints, N) # should include 2 items from each segment and maximize score by including items from earlier segments

    # Test Case 9: 100 candidate items for N=10, hard constraints - max diversity
    N = 10
    items = {f'item-{i}': max(random.uniform(0, 1) - i*0.01, 0) for i in range(1, 101)} # Decreasing scores to check diversity
    segments = [Segment(f'segment{i}', 'test-prop', *list(items.keys())[i*20:(i+1)*20]) for i in range(5)]
    constraints = [
        SegmentationMaxDiversity(segmentation_property='test-prop', max_items=2, weight=1.0)
    ]
    run_test("Test Case 9", solver, items, segments, constraints, N) # should include 2 items from each segment even if it reduces the total score


def ILP_time_efficiency(constraint_weight=1.0):
    solver = ILP()
    num_recomms = [5, 10, 20, 50, 100, 200, 500] # N
    num_candidates = [10, 50, 100, 200, 500, 1000] # M
    num_constraints = [1, 2, 3, 4, 5, 8, 10, 15] # C
    num_segments = {10: 2, 50: 10, 100: 10, 200: 20, 500: 50, 1000: 100} # M -> S
    elapsed_times = dict() # (N, M, C) -> elapsed_time

    for N in num_recomms:
        for M in num_candidates:
            if M < N:
                continue
            items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M+1)}
            S = num_segments[M]
            segments = segments = [Segment(f'segment{i}', 'prop', *list(items.keys())[i*S:(i+1)*S]) for i in range(S)]
            available_constraints = [
                                        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=N//5,
                                                                     window_size=N, weight=constraint_weight) for i in range(S)
                                    ] + [
                                        MaxItemsPerSegmentConstraint(segment_id=f'segment{i}', max_items=N//5+1,
                                                                     window_size=N, weight=constraint_weight) for i in range(S)
                                    ] + [
                                        MinItemsPerSegmentConstraint(segment_id=f'segment{i}', min_items=2,
                                                                     window_size=5, weight=constraint_weight) for i in range(S)
                                    ] + [
                                        MaxItemsPerSegmentConstraint(segment_id=f'segment{i}', max_items=3,
                                                                     window_size=5, weight=constraint_weight) for i in range(S)
                                    ]
            for C in num_constraints:
                constraints = random.choices(available_constraints, k=C)
                elapsed_time = run_test(f"Test Case ({N}, {M}, {C})", solver, items, segments, constraints, N, using_soft_constraints=True)
                elapsed_times[(N, M, C)] = elapsed_time

    plot_results(elapsed_times)


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


if __name__ == "__main__":
    main()
    # ILP_time_efficiency()
    # ILP_time_efficiency(constraint_weight=0.9)
