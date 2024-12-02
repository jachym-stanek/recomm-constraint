import time

from gurobipy import Model, GRB, quicksum
from typing import Dict, List, Set

from src.algorithms.algorithm import Algorithm
from src.segmentation import Segment
from src.constraints.constraint import Constraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint, \
    ItemFromSegmentAtPositionConstraint, ItemAtPositionConstraint, SegmentationMinDiversity, SegmentationMaxDiversity, \
    Constraint2D, ItemUniqueness2D


class ILP(Algorithm):
    def __init__(self, name="ILP", description="Integer Linear Programming Solver", verbose=True):
        super().__init__(name, description, verbose)

    def solve_by_partitioning(self, items: Dict[str, float], segments: Dict[str, Segment], constraints: List[Constraint], N: int,
                              partition_size: int, item_segment_map: Dict[str, str]):
        start_time = time.time()

        if self.verbose:
            print(f"[{self.name}] Solving ILP using partitioning with {len(items)} candidate items, {len(segments)} segments,"
                  f" {len(constraints)} constraints, count={N}, partition size: {partition_size}.")

        if N < partition_size:
            return self.solve(items, segments, constraints, N)

        candidates = items.copy()
        final_result = {}
        already_recommended_items = []
        partition_start = 0

        while len(final_result) < N:
            # print("===================================== Partitioning ======================================")
            partition_count = min(partition_size, N - len(final_result))
            partition_candidates = self.preprocess_items(candidates, segments, constraints, N=partition_count, item_segment_map=item_segment_map)
            partition_constraints = []
            for constraint in constraints:
                if isinstance(constraint, MaxItemsPerSegmentConstraint):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count else constraint.window_size
                    partition_constraints.append(MaxItemsPerSegmentConstraint(constraint.segment_id, constraint.max_items,
                                                                              window_size, weight=constraint.weight))
                elif isinstance(constraint, SegmentationMaxDiversity):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count else constraint.window_size
                    partition_constraints.append(SegmentationMaxDiversity(constraint.segmentation_property, constraint.max_items,
                                                                          window_size, weight=constraint.weight))
                elif isinstance(constraint, MinItemsPerSegmentConstraint):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count + len(already_recommended_items) else constraint.window_size
                    partition_constraint_weight = 0.9 if constraint.weight == 1 and constraint.window_size > partition_count + len(already_recommended_items) else constraint.weight  # make min constraints soft if they are hard
                    partition_constraints.append(MinItemsPerSegmentConstraint(constraint.segment_id, constraint.min_items,
                                                                              window_size, weight=partition_constraint_weight))
                elif isinstance(constraint, SegmentationMinDiversity):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count else constraint.window_size
                    partition_constraint_weight = 0.9 if constraint.weight == 1 and len(already_recommended_items) + partition_count < N else constraint.weight
                    partition_constraints.append(SegmentationMinDiversity(constraint.segmentation_property, constraint.min_items,
                                                                      window_size, weight=partition_constraint_weight))
                else:
                    partition_constraints.append(constraint)

            # already_recommended_segments = [item_segment_map.get(item) for item in already_recommended_items]
            # candidates_segments = [item_segment_map.get(item) for item in partition_candidates.keys()]
            # already_recommended_segments_count = {segment: already_recommended_segments.count(segment) for segment in set(already_recommended_segments)}
            # candidates_segments_count = {segment: candidates_segments.count(segment) for segment in set(candidates_segments)}
            # print(f"[ILP] Already recommended segments: {already_recommended_segments}, \nCandidate segments: {candidates_segments}")
            # print(f"[ILP] Already recommended segments count: {dict(sorted(already_recommended_segments_count.items()))}, \nCandidate segments count: {dict(sorted(candidates_segments_count.items()))}")

            partition_result = self.solve(partition_candidates, segments, partition_constraints, partition_count, already_recommended_items)

            # Add the inner result to the final result and remove the recommended items from the candidate list
            for position, item in partition_result.items():
                final_result[position+partition_start] = item
                already_recommended_items.append(item)
                candidates.pop(item)

            partition_start += partition_size

        end_time = time.time()

        if self.verbose:
            print(f"[{self.name}] Partitioning solution finished in {(end_time - start_time) * 1000:.2f} ms")

        return final_result

    def solve(self, items: Dict[str, float], segments: Dict[str, Segment], constraints: List[Constraint], N: int,
              already_recommended_items: List[str] = None):
        start = time.time()

        if self.verbose:
            print(f"[{self.name}] Solving ILP with {len(items)} candidate items, {len(segments)} segments, {len(constraints)} constraints, count={N}.")

        model = Model("RecommenderSystem")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        item_ids = list(items.keys())
        positions = list(range(1, N + 1))

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
            constraint.add_to_model(model, x, items, segments, positions, N, K, already_recommended_items)

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
                         N: int, verbose=False) -> Dict[str, float]:
        start = time.time()

        # Step 1: Process MaxItemsPerSegmentConstraints
        # Initialize a set to hold the candidate items
        removed_items = set()
        segment_bare_minimums = {}  # used for MinItemsPerSegmentConstraint

        # For each MaxItemsPerSegmentConstraint, compute the maximum number of items
        for constraint in constraints:
            if isinstance(constraint, MaxItemsPerSegmentConstraint):
                self.preprocess_MaxItemsPerSegmentConstraint(constraint, N, items, segments, removed_items)
            elif isinstance(constraint, SegmentationMaxDiversity):
                for c in constraint.constraints:
                    self.preprocess_MaxItemsPerSegmentConstraint(c, N, items, segments, removed_items)
            elif isinstance(constraint, MinItemsPerSegmentConstraint):
                self.compute_bare_minumum_for_MinItemsPerSegmentConstraint(constraint, N, segment_bare_minimums)
            elif isinstance(constraint, SegmentationMinDiversity):
                for c in constraint.constraints:
                    self.compute_bare_minumum_for_MinItemsPerSegmentConstraint(c, N, segment_bare_minimums)

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
                self.preprocess_MinItemsPerSegmentConstraint(constraint, N, segment_bare_minimums, min_required_items, min_satisfied)
            elif isinstance(constraint, SegmentationMinDiversity):
                for c in constraint.constraints:
                    self.preprocess_MinItemsPerSegmentConstraint(c, N, segment_bare_minimums, min_required_items, min_satisfied)

        candidate_items = dict()
        idx = 0
        while True:
            item, score = sorted_items[idx]
            item_segment = item_segment_map.get(item)

            # decide if to add item - if we do not have enough items or there is a minimum constraint that is not satisfied
            if (len(candidate_items) < N or item_segment in min_satisfied) and not (item_segment in min_satisfied and min_satisfied[item_segment]): # do not take items that have not satisfied min constraints
                candidate_items[item] = score

                if item_segment is not None:
                    added_items_per_segment[item_segment].add(item)

                if item_segment in min_satisfied:
                    if len(added_items_per_segment[item_segment]) >= min_required_items[item_segment]:
                        min_satisfied[item_segment] = True
                        # print(f"[ILP] Segment {item_segment} has enough items {len(added_items_per_segment[item_segment])} >= {min_required_items[item_segment]}")

            if len(candidate_items) >= N and all(min_satisfied.values()): # all min items constraints are satisfied and we have enough items
                break

            idx += 1
            if idx >= len(sorted_items):
                break

        if verbose:
            print(f"[ILP] Total candidate items after preprocessing: {len(candidate_items)} time: {(time.time() - start)*1000} milliseconds")
        return candidate_items

    def preprocess_MaxItemsPerSegmentConstraint(self, constraint: MaxItemsPerSegmentConstraint, N: int, items: Dict[str, float],
                                                segments: Dict[str, Segment], removed_items: Set[str]):
        segment_id = constraint.segment_id
        max_items = constraint.max_items
        window_size = constraint.window_size

        # Compute the maximum possible number of items from the segment
        max_total_items = self.compute_limit_items(N, window_size, max_items)
        # Get the segment's items
        segment_items = [i for i in segments[segment_id] if i in items]
        sorted_items = sorted(segment_items, key=lambda x: items[x])
        removed = set(sorted_items[:-max_total_items])  # Remove the lowest scoring items, keep the top max_total_items
        removed_items.update(removed)
        # print(f"[ILP] Removing {len(removed)} items from segment {segment_id}, segment num items: {len(segments[segment_id])}, max_items: {max_items}, max_total_items: {max_total_items}")

    def compute_bare_minumum_for_MinItemsPerSegmentConstraint(self, constraint: MinItemsPerSegmentConstraint, N: int, segment_bare_minimums: Dict[str, int]):
        segment_id = constraint.segment_id
        min_items = constraint.min_items
        window_size = constraint.window_size
        segment_bare_minimums[segment_id] = max(segment_bare_minimums.get(segment_id, 0), self.compute_limit_items(N, window_size, min_items))

    def preprocess_MinItemsPerSegmentConstraint(self, constraint: MinItemsPerSegmentConstraint, N: int, segment_bare_minimums: Dict[str, int],
                                                min_required_items: Dict[str, int], min_satisfied: Dict[str, bool]):
        segment_id = constraint.segment_id
        sum_bare_minimums = sum([v for v in segment_bare_minimums.values()])
        min_required_items[segment_id] = segment_bare_minimums.get(segment_id, 0) + N - sum_bare_minimums
        min_satisfied[segment_id] = False

    # hard limit for MaxItemsPerSegmentConstraint, worst case for MinItemsPerSegmentConstraint
    def compute_limit_items(self, N, W, item_limit_per_window):
        max_items_limit = 0
        p = 1
        while p <= N:
            # Set m positions to 'X'
            for _ in range(item_limit_per_window):
                if p <= N:
                    max_items_limit += 1
                    p += 1
                else:
                    break
            # Set W - m positions to 'O'
            p += W - item_limit_per_window

        return max_items_limit

