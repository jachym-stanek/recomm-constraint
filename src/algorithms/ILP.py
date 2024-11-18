import time

from gurobipy import Model, GRB, quicksum
from typing import Dict, List, Set

from src.algorithms.algorithm import Algorithm
from src.segmentation import Segment
from src.constraints.constraint import Constraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint, \
    ItemFromSegmentAtPositionConstraint, ItemAtPositionConstraint


class ILP(Algorithm):
    def __init__(self, name="ILP", description="Integer Linear Programming Solver", verbose=True):
        super().__init__(name, description, verbose)

    def solve(self, items: Dict[str, float], segments: List[Segment], constraints: List[Constraint], N: int):
        start = time.time()

        if self.verbose:
            print(f"[{self.name}] Solving ILP with {len(items)} candidate items, {len(segments)} segments, {len(constraints)} constraints, count={N}.")

        model = Model("RecommenderSystem")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        item_ids = list(items.keys())
        positions = list(range(1, N + 1))
        segments_dict = {seg.id: seg for seg in segments}

        # Create decision variables x[i,p] for item i at position p
        x = model.addVars(item_ids, positions, vtype=GRB.BINARY, name="x")

        # Initialize penalties list
        model._penalties = []

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

        # Penalty scaling factor K (total possible score)
        K = sum(items.values())

        # Process each constraint in the constraints list
        for constraint in constraints:
            constraint.add_to_model(model, x, items, segments_dict, positions, N, K)

        # Objective function: Maximize total score - total penalty
        total_score = quicksum(items[i] * x[i, p] for i in item_ids for p in positions)
        total_penalty = quicksum(penalty_coeff * s for s, penalty_coeff in model._penalties)
        model.setObjective(total_score - total_penalty, GRB.MAXIMIZE)

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
        elif self.verbose:
            print(f"[{self.name}] No optimal solution found.")

        end = time.time()

        if self.verbose:
            print(f"[{self.name}] Finished in {(end - start) * 1000:.2f} ms")

        return result


    """
    Filter the number of candidates in order to reduce the number of ILP variables
    
    for every max_items constraint:
        Take only the top max_items items from the segment  TODO: invent a way to handle for overlapping segments
    for remaining items:
        sort by score
        take top N items
        keep taking until all min_items constraints have enough items
    
    """
    def preprocess_items(self, items: Dict[str, float],
                         segments: Dict[str, Segment],
                         constraints: List[Constraint],
                         item_segment_map: Dict[str, str], # map item_id to segment_id
                         N: int):
        start = time.time()

        # Step 1: Process MaxItemsPerSegmentConstraints
        # Initialize a set to hold the candidate items
        removed_items = set()

        # For each MaxItemsPerSegmentConstraint, compute the maximum number of items
        for constraint in constraints:
            if isinstance(constraint, MaxItemsPerSegmentConstraint):
                segment_id = constraint.segment_id
                max_items = constraint.max_items
                window_size = constraint.window_size

                # Compute the maximum possible number of items from the segment
                max_total_items = self.compute_limit_items(N, window_size, max_items)
                # Get the segment's items
                segment_items = list(segments[segment_id])
                sorted_items = sorted(segment_items, key=lambda x: items[x])
                removed = set(sorted_items[:-max_total_items]) # Remove the lowest scoring items, keep the top max_total_items
                removed_items.update(removed)
                print(f"[ILP] Removing {len(removed)} items from segment {segment_id}, segment num items: {len(segments[segment_id])}, max_items: {max_items}, max_total_items: {max_total_items}")

        # Order the remaining items by score
        remaining_items = {item_id: score for item_id, score in items.items() if item_id not in removed_items}
        sorted_items = sorted(remaining_items.items(), key=lambda x: x[1], reverse=True)

        # Step 2: Process MinItemsPerSegmentConstraints
        # Compute the total minimum required items
        min_required_items = {segment_id: 0 for segment_id in segments.keys()}
        added_items_per_segment = {segment_id: set() for segment_id in segments.keys()}
        min_satisfied = {}
        for constraint in constraints:
            if isinstance(constraint, MinItemsPerSegmentConstraint):
                segment_id = constraint.segment_id
                min_required_items[segment_id] = self.compute_limit_items(N, constraint.window_size, constraint.min_items)
                min_satisfied[segment_id] = False

        candidate_items = dict()
        idx = 0
        while True:
            item, score = sorted_items[idx]
            item_segment = item_segment_map.get(item)

            # decide if to add item
            if len(candidate_items) < N or (item_segment in min_satisfied and not min_satisfied[item_segment]):
                candidate_items[item] = score

                if item_segment is not None:
                    added_items_per_segment[item_segment].add(item)

                if item_segment in min_satisfied:
                    if len(added_items_per_segment[item_segment]) >= min_required_items[item_segment]:
                        min_satisfied[item_segment] = True
                        print(f"[ILP] Segment {item_segment} has enough items {len(added_items_per_segment[item_segment])} >= {min_required_items[item_segment]}")

            if len(candidate_items) >= N and all(min_satisfied.values()): # all min items constraints are satisfied and we have enough items
                break

            idx += 1
            if idx >= len(sorted_items):
                break

        # remove items from segments
        for segment_id, items in added_items_per_segment.items():
            segments[segment_id] = Segment(segment_id, segments[segment_id].segmentation_property, *items)

        print(f"[ILP] Total candidate items after preprocessing: {len(candidate_items)} time: {(time.time() - start)*1000} milliseconds")
        return candidate_items, list(segments.values())

    # hard limit for MaxItemsPerSegmentConstraint, worst case for MinItemsPerSegmentConstraint
    def compute_limit_items(self, N, W, M):
        max_items = 0
        p = 1
        while p <= N:
            # Set M positions to 'X'
            for _ in range(M):
                if p <= N:
                    max_items += 1
                    p += 1
                else:
                    break
            # Set W - M positions to 'O'
            p += W - M

        return max_items
