import random

from gurobipy import Model, GRB, quicksum
from typing import Dict, List

from segmentation import Segmentation
from constraints.constraint import Constraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint, \
    ItemFromSegmentAtPositionConstraint, ItemAtPositionConstraint



def solve_recommender_system(items: Dict[str, float],
                             segmentations: List[Segmentation],
                             constraints: List[Constraint],
                             N: int):
    model = Model("RecommenderSystem")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    item_ids = list(items.keys())
    positions = list(range(1, N + 1))
    segments = {seg.id: seg for seg in segmentations}

    # Create decision variables x[i,p] for item i at position p
    x = model.addVars(item_ids, positions, vtype=GRB.BINARY, name="x")

    # Objective function: Maximize the total score of selected items
    model.setObjective(
        quicksum(items[i] * x[i, p] for i in item_ids for p in positions),
        GRB.MAXIMIZE
    )

    # Constraint 1: Each item is selected at most once
    for i in item_ids:
        model.addConstr(
            quicksum(x[i, p] for p in positions) <= 1,
            name=f"ItemOnce_{i}"
        )

    # Constraint 2: Each position has at most one item
    for p in positions:
        model.addConstr(
            quicksum(x[i, p] for i in item_ids) <= 1,
            name=f"PositionOnce_{p}"
        )

    # Constraint 3: Exactly N items are selected
    model.addConstr(
        quicksum(x[i, p] for i in item_ids for p in positions) == N,
        name="TotalItems"
    )

    # Process each constraint in the constraints list
    for constraint in constraints:
        if isinstance(constraint, MinItemsPerSegmentConstraint):
            segment_id = constraint.segment_id
            min_items = constraint.min_items
            segment_items = segments[segment_id]
            model.addConstr(
                quicksum(x[i, p] for i in segment_items for p in positions) >= min_items,
                name=f"MinItems_{segment_id}"
            )

        elif isinstance(constraint, MaxItemsPerSegmentConstraint):
            segment_id = constraint.segment_id
            max_items = constraint.max_items
            segment_items = segments[segment_id]
            model.addConstr(
                quicksum(x[i, p] for i in segment_items for p in positions) <= max_items,
                name=f"MaxItems_{segment_id}"
            )

        elif isinstance(constraint, ItemFromSegmentAtPositionConstraint):
            segment_id = constraint.segment_id
            position = constraint.position
            segment_items = segments[segment_id]
            model.addConstr(
                quicksum(x[i, position] for i in segment_items) >= 1,
                name=f"SegmentAtPosition_{segment_id}_{position}"
            )

        elif isinstance(constraint, ItemAtPositionConstraint):
            item_id = constraint.item_id
            position = constraint.position
            model.addConstr(
                x[item_id, position] == 1,
                name=f"ItemAtPosition_{item_id}_{position}"
            )

        else:
            raise ValueError(f"Unknown constraint type: {type(constraint)}")

    # Optimize the model
    model.optimize()

    # Check if the model found an optimal solution
    if model.Status == GRB.OPTIMAL:
        # Extract the solution
        solution = {}
        for i in item_ids:
            for p in positions:
                if x[i, p].X > 0.5:
                    solution[p] = i  # Map position to item
        # Return the recommended items sorted by position
        return {k: solution[k] for k in sorted(solution)}
    else:
        print("No optimal solution found.")
        return None

if __name__ == "__main__":
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
    seg1 = Segmentation('genre1', 'genre', 'item1', 'item2', 'item3', 'item4', 'item5')
    seg2 = Segmentation('genre2', 'genre', 'item6', 'item7', 'item8', 'item9', 'item10')
    seg3 = Segmentation('genre3', 'genre', 'item11', 'item12', 'item13', 'item14', 'item15')
    seg4 = Segmentation('genre4', 'genre', 'item16', 'item17', 'item18', 'item19', 'item20')

    segmentations = [seg1, seg2, seg3, seg4]

    # Prepare segments dictionary
    segments = {seg.id: seg for seg in segmentations}

    # Test Case 1: Single Constraint
    print("\n=== Test Case 1: Single Constraint ===")
    constraints = [
        MinItemsPerSegmentConstraint(segment_id='genre2', min_items=2)
    ]

    N = 5

    recommended_items = solve_recommender_system(items, segmentations, constraints, N)

    # Check constraints
    if recommended_items:
        all_constraints_satisfied = True
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments):
                print(f"Constraint {constraint} is not satisfied.")
                all_constraints_satisfied = False
        if all_constraints_satisfied:
            print("All constraints are satisfied for Test Case 1.")
            print("Recommended Items:")
            total_score = 0
            for position, item_id in recommended_items.items():
                score = items[item_id]
                total_score += score
                print(f"Position {position}: {item_id} (Score: {score:.1f})")
            print(f"Total Score: {total_score:.1f}")
    else:
        print("No solution found for Test Case 1.")

    # Test Case 2: Multiple Constraints
    print("\n=== Test Case 2: Multiple Constraints ===")
    constraints = [
        ItemFromSegmentAtPositionConstraint(segment_id='genre1', position=1),
        MaxItemsPerSegmentConstraint(segment_id='genre3', max_items=2),
        ItemAtPositionConstraint(item_id='item6', position=3)
    ]

    N = 7

    recommended_items = solve_recommender_system(items, segmentations, constraints, N)

    # Check constraints
    if recommended_items:
        all_constraints_satisfied = True
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments):
                print(f"Constraint {constraint} is not satisfied.")
                all_constraints_satisfied = False
        if all_constraints_satisfied:
            print("All constraints are satisfied for Test Case 2.")
            print("Recommended Items:")
            total_score = 0
            for position, item_id in recommended_items.items():
                score = items[item_id]
                total_score += score
                print(f"Position {position}: {item_id} (Score: {score:.1f})")
            print(f"Total Score: {total_score:.1f}")
    else:
        print("No solution found for Test Case 2.")

    # Test Case 3: Hard Constraints
    print("\n=== Test Case 3: Hard Constraints ===")
    constraints = [
        ItemFromSegmentAtPositionConstraint(segment_id='genre1', position=1),
        MinItemsPerSegmentConstraint(segment_id='genre3', min_items=2),
        MaxItemsPerSegmentConstraint(segment_id='genre3', max_items=4),
        ItemAtPositionConstraint(item_id='item6', position=3),
        MaxItemsPerSegmentConstraint(segment_id='genre1', max_items=2)
    ]

    N = 5

    recommended_items = solve_recommender_system(items, segmentations, constraints, N)

    # Check constraints
    if recommended_items:
        all_constraints_satisfied = True
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments):
                print(f"Constraint {constraint} is not satisfied.")
                all_constraints_satisfied = False
        if all_constraints_satisfied:
            print("All constraints are satisfied for Test Case 3.")
            print("Recommended Items:")
            total_score = 0
            for position, item_id in recommended_items.items():
                score = items[item_id]
                total_score += score
                print(f"Position {position}: {item_id} (Score: {score:.1f})")
            print(f"Total Score: {total_score:.1f}")
    else:
        print("No solution found for Test Case 3.")
