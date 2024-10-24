import time

from gurobipy import Model, GRB, quicksum
from typing import Dict, List

from src.algorithms.algorithm import Algorithm
from src.segmentation import Segment
from src.constraints.constraint import Constraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint, \
    ItemFromSegmentAtPositionConstraint, ItemAtPositionConstraint


class ILP(Algorithm):
    def __init__(self, name="ILP", description="Integer Linear Programming Solver"):
        super().__init__(name, description)

    def solve(self, items: Dict[str, float], segmentations: List[Segment], constraints: List[Constraint], N: int):
        start = time.time()

        print(f"[{self.name}] Solving ILP with {len(items)} candidate items, {len(segmentations)} segmentations, {len(constraints)} constraints, count={N}.")

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
                raise ValueError(f"[{self.name}]Unsupported constraint type: {type(constraint)}")

        # Optimize the model
        model.optimize()
        result = None

        # Check if the model found an optimal solution
        if model.Status == GRB.OPTIMAL:
            # Extract the solution
            solution = {}
            for i in item_ids:
                for p in positions:
                    if x[i, p].X > 0.5:
                        solution[p] = i  # Map position to item
            # Return the recommended items sorted by position
            result = {k: solution[k] for k in sorted(solution)}
        else:
            print(f"[{self.name}]No optimal solution found.")

        end = time.time()
        print(f"[{self.name}] Finished in {(end - start) * 1000:.2f} ms")

        return result
