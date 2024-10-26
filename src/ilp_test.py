import random
import time

from typing_extensions import runtime

from algorithms.ILP import ILP
from segmentation import Segment
from src.constraints.constraint import Constraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint, \
    ItemFromSegmentAtPositionConstraint, ItemAtPositionConstraint


def run_test(test_name, solver, items, segments, constraints, N):
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
        if all_constraints_satisfied:
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

    return elapsed_time


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


if __name__ == "__main__":
    main()
