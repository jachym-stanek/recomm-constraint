import copy
import math
import time

from gurobipy import Model, GRB, quicksum
from typing import Dict, List, Set

from src.algorithms.algorithm import Algorithm
from src.segmentation import Segment
from src.constraints.constraint import Constraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint, \
    ItemFromSegmentAtPositionConstraint, ItemAtPositionConstraint, GlobalMinItemsPerSegmentConstraint, \
    GlobalMaxItemsPerSegmentConstraint, \
    Constraint2D, ItemUniqueness2D, MaxSegmentsConstraint, MinSegmentsConstraint


class IlpSolver(Algorithm):
    def __init__(self, name="ILP", description="Integer Linear Programming Solver", verbose=True):
        super().__init__(name, description, verbose)

    def solve_by_partitioning(self, items: Dict[str, float], segments: Dict[str, Segment], constraints: List[Constraint], N: int,
                              partition_size: int, item_segment_map: Dict[str, str], use_doubling: bool = False):
        start_time = time.time()

        temp_item_segment_map = item_segment_map.copy()
        items_remaining_per_segment = {segment_id: Segment(segment_id, segment.segmentation_property, *set(segment)) for segment_id, segment in segments.items()}

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
            if self.verbose:
                print("===================================== Partitioning ======================================")
            if use_doubling:
                partition_count = min(partition_size * 2, N - len(final_result))
            else:
                partition_count = min(partition_size, N - len(final_result))

            remaining_recomm_len = N - len(final_result)
            partition_constraints = []
            for constraint in constraints:
                if isinstance(constraint, MaxItemsPerSegmentConstraint):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count else constraint.window_size
                    partition_constraints.append(MaxItemsPerSegmentConstraint(constraint.segment_id, constraint.max_items,
                                                                              window_size, weight=constraint.weight))
                elif isinstance(constraint, GlobalMaxItemsPerSegmentConstraint):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count else constraint.window_size
                    partition_constraints.append(GlobalMaxItemsPerSegmentConstraint(constraint.segmentation_property, constraint.max_items,
                                                                                    window_size, weight=constraint.weight))
                elif isinstance(constraint, MinItemsPerSegmentConstraint):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count + len(already_recommended_items) else constraint.window_size
                    partition_constraint_weight = 0.9 if constraint.weight == 1 and constraint.window_size > partition_count + len(already_recommended_items) else constraint.weight  # make min constraints soft if they are hard
                    partition_constraints.append(MinItemsPerSegmentConstraint(constraint.segment_id, constraint.min_items,
                                                                              window_size, weight=partition_constraint_weight))
                elif isinstance(constraint, GlobalMinItemsPerSegmentConstraint):
                    window_size = min(constraint.window_size, partition_count + len(already_recommended_items)) if constraint.window_size > partition_count else constraint.window_size
                    partition_constraint_weight = 0.9 if constraint.weight == 1 and len(already_recommended_items) + partition_count < N else constraint.weight
                    partition_constraints.append(GlobalMinItemsPerSegmentConstraint(constraint.segmentation_property, constraint.min_items,
                                                                                    window_size, weight=partition_constraint_weight))
                # TODO: also handle changing window size for SegmentDiversityConstraints
                else:
                    partition_constraints.append(constraint)

            partition_candidates = self.preprocess_items(candidates, items_remaining_per_segment, segments, constraints, N=partition_count,
                                                         item_segment_map=item_segment_map,
                                                         remaining_recomm_len=remaining_recomm_len,
                                                         previous_recommended_items=already_recommended_items)

            # already_recommended_segments = [temp_item_segment_map.get(item) for item in already_recommended_items]
            # candidates_segments = [temp_item_segment_map.get(item) for item in partition_candidates.keys()]
            # already_recommended_segments_count = {segment: already_recommended_segments.count(segment) for segment in set(already_recommended_segments)}
            # candidates_segments_count = {segment: candidates_segments.count(segment) for segment in set(candidates_segments)}
            # print(f"[ILP] Already recommended segments: {already_recommended_segments}, \nCandidate segments: {candidates_segments}")
            # print(f"Already recommended segments count: {dict(sorted(already_recommended_segments_count.items(), key=lambda x: (x[0].rstrip('0123456789'), int(x[0][len(x[0].rstrip('0123456789')):]))))}, \nCandidate segments count:           {dict(sorted(candidates_segments_count.items(), key=lambda x: (x[0].rstrip('0123456789'), int(x[0][len(x[0].rstrip('0123456789')):]))))}")

            partition_result = self.solve(partition_candidates, segments, partition_constraints, partition_count, already_recommended_items)

            # take only half
            if use_doubling:
                taken_until = min(partition_size, len(partition_result))
                partition_result = {k: v for k, v in partition_result.items() if k <= taken_until}

            # Add the inner result to the final result and remove the recommended items from the candidate list
            for position, item in partition_result.items():
                final_result[position+partition_start] = item
                already_recommended_items.append(item)
                candidates.pop(item)
                item_segment = item_segment_map.get(item) # TODO: remove from all possible segments
                if item_segment is not None:
                    items_remaining_per_segment[item_segment].remove(item)
                if item in item_segment_map:
                    item_segment_map.pop(item)


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

        # Create decision variables x[i,0,p] for item i at position p, 1D case so row index is 0 (for compatibility with 2D)
        x = model.addVars(item_ids, [0], positions, vtype=GRB.BINARY, name="x")

        # Initialize penalties list
        model._penalties = []

        # Constraint 1: Each item is selected at most once
        for i in item_ids:
            model.addConstr(
                quicksum(x[i, 0, p] for p in positions) <= 1,
                name=f"ItemOnce_{i}"
            )

        # Constraint 2: Each position has at most one item
        for p in positions:
            model.addConstr(
                quicksum(x[i, 0, p] for i in item_ids) <= 1,
                name=f"PositionOnce_{p}"
            )

        # Constraint 3: Exactly N items are selected
        model.addConstr(
            quicksum(x[i, 0, p] for i in item_ids for p in positions) == N,
            name="TotalItems"
        )

        # Penalty scaling factor K (total possible score)
        K = sum(items.values())

        # Process each constraint in the constraints list
        for constraint in constraints:
            constraint.add_to_ilp_model(model, x, items, segments, 0, positions, N, K, already_recommended_items)

        # Objective function: Maximize total score - total penalty
        total_score = quicksum(items[i] * x[i, 0, p] for i in item_ids for p in positions)
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
                    if x[i, 0, p].X > 0.5:
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
    # TODO: multiple segments per item
    def preprocess_items(self, items: Dict[str, float],
                         items_remaining_per_segment: Dict[str, Segment],
                         segments: Dict[str, Segment],
                         constraints: List[Constraint],
                         item_segment_map: Dict[str, list], # map item_id to all item segment ids
                         N: int, remaining_recomm_len: int = None,
                         previous_recommended_items: List[str] = None) -> Dict[str, float]:
        start = time.time()

        # Step 1: Process MaxItemsPerSegmentConstraints
        # Initialize a set to hold the candidate items
        removed_items = set()
        segment_bare_minimums = {}  # used for MinItemsPerSegmentConstraint
        # at least minimum_number_of_segments must have at least minimum_items_per_segment items
        minimum_number_of_segments = 0
        minimum_items_per_segment = 0

        # For each MaxItemsPerSegmentConstraint, compute the maximum number of items
        for constraint in constraints:
            if isinstance(constraint, MaxItemsPerSegmentConstraint):
                self.preprocess_MaxItemsPerSegmentConstraint(constraint, N, items, segments, removed_items, already_recommended_items=previous_recommended_items)
            elif isinstance(constraint, GlobalMaxItemsPerSegmentConstraint):
                constraint.initialize_constraint_from_segments(items_remaining_per_segment)
                for c in constraint.constraints:
                    self.preprocess_MaxItemsPerSegmentConstraint(c, N, items, segments, removed_items, already_recommended_items=previous_recommended_items)
            elif isinstance(constraint, MinItemsPerSegmentConstraint):
                self.compute_bare_minumum_for_MinItemsPerSegmentConstraint(constraint, N, segment_bare_minimums)
            elif isinstance(constraint, GlobalMinItemsPerSegmentConstraint):
                constraint.initialize_constraint_from_segments(items_remaining_per_segment)
                for c in constraint.constraints:
                    self.compute_bare_minumum_for_MinItemsPerSegmentConstraint(c, N, segment_bare_minimums)
            elif isinstance(constraint, MinSegmentsConstraint):
                minimum_number_of_segments = constraint.min_segments
                minimum_items_per_segment = self.compute_limit_items(N, constraint.window_size, constraint.min_segments)
            elif isinstance(constraint, MaxSegmentsConstraint): # no preprecessing needed
                seqment_quotient = math.ceil(constraint.window_size / constraint.max_segments)
                minimum_number_of_segments = max(minimum_items_per_segment, constraint.max_segments)
                minimum_items_per_segment = max(minimum_items_per_segment, min(seqment_quotient+1, N))

        # Order the remaining items by score
        remaining_items = {item_id: score for item_id, score in items.items() if item_id not in removed_items}
        sorted_items = sorted(remaining_items.items(), key=lambda x: x[1], reverse=True)

        # Step 2: Process MinItemsPerSegmentConstraints
        # Compute the total minimum required items
        min_required_items = {segment_id: 0 for segment_id in items_remaining_per_segment.keys()}
        added_items_per_segment = {segment_id: set() for segment_id in items_remaining_per_segment.keys()}
        min_satisfied = {}
        for constraint in constraints:
            if isinstance(constraint, MinItemsPerSegmentConstraint):
                self.preprocess_MinItemsPerSegmentConstraint(constraint, N, segment_bare_minimums, min_required_items,
                                                             min_satisfied, len(items_remaining_per_segment[constraint.segment_id]), remaining_recomm_len)
            elif isinstance(constraint, GlobalMinItemsPerSegmentConstraint):
                for c in constraint.constraints:
                    self.preprocess_MinItemsPerSegmentConstraint(c, N, segment_bare_minimums, min_required_items, min_satisfied,
                                                                 len(items_remaining_per_segment[c.segment_id]), remaining_recomm_len)

        candidate_items = dict()
        idx = 0
        while True:
            item, score = sorted_items[idx]
            item_segments = item_segment_map.get(item)

            if item_segments is None:
                item_segments = [None]

            if type(item_segments) is not list:
                item_segments = [item_segments]

            for item_segment in item_segments:
                # decide if to add item - if we do not have enough items or there is a minimum constraint that is not satisfied
                if self._item_decision_function(remaining_recomm_len, candidate_items, N, item_segment, min_satisfied, added_items_per_segment) or \
                    self._segment_decision_function(item_segment, minimum_number_of_segments, minimum_items_per_segment,
                                                    added_items_per_segment):
                    candidate_items[item] = score

                    if item_segment is not None:
                        added_items_per_segment[item_segment].add(item)

                    if item_segment in min_satisfied:
                        if len(added_items_per_segment[item_segment]) >= min_required_items[item_segment]:
                            min_satisfied[item_segment] = True
                            # print(f"[ILP] Segment {item_segment} has enough items {len(added_items_per_segment[item_segment])} >= {min_required_items[item_segment]}")
                    break

            # all min items constraints are satisfied, we have enough items and enough segments
            if (len(candidate_items) >= N and all(min_satisfied.values()) and
                    self._enough_segments_with_min_items(added_items_per_segment, minimum_number_of_segments, minimum_items_per_segment)):
                break

            idx += 1
            if idx >= len(sorted_items):
                break

        # fallback if preprocessing still filtered out too many items (can happen when a lot of high scoring items are
        # removed and then only small scoring items are left from the segment with min items constraint)
        if len(candidate_items) < N:
            for item, score in sorted_items:
                if item not in candidate_items:
                    candidate_items[item] = score
                if len(candidate_items) >= N:
                    break

        if self.verbose:
            print(f"[ILP] Total candidate items after preprocessing: {len(candidate_items)} time: {(time.time() - start)*1000} milliseconds")
            number_of_added_items_per_segment = {segment_id: len(items) for segment_id, items in added_items_per_segment.items()}
            print(f"Number of added items per segment: {number_of_added_items_per_segment}")
        return candidate_items

    def _item_decision_function(self, remaining_recomm_len, candidate_items, N, item_segment, min_satisfied, added_items_per_segment) -> bool:
        # do not take items that have not satisfied min constraints (if we have more partitions on the way
        case1 = (remaining_recomm_len is not None and (len(candidate_items) < N or item_segment in min_satisfied) and not
                (item_segment in min_satisfied and min_satisfied[item_segment])
                    or (all(min_satisfied.values()) and len(candidate_items) < N))
        # no consecutive recomms -> cam afford to be more greedy
        case2 = (remaining_recomm_len is None and (len(candidate_items) < N or (item_segment in min_satisfied and not min_satisfied[item_segment])))
        not_overselecting_segment = (item_segment is None or len(added_items_per_segment[item_segment]) < N)
        return (case1 or case2) and not_overselecting_segment

    def _enough_segments_with_min_items(self, added_items_per_segment, min_segments, min_items) -> bool:
        return len([seg for seg, items in added_items_per_segment.items() if len(items) >= min_items]) >= min_segments

    def _segment_decision_function(self, segment, min_segments, min_items, added_items_per_segment) -> bool:
        segments_with_min_items = [seg for seg, items in added_items_per_segment.items() if len(items) >= min_items]
        if segment in segments_with_min_items:
            return False
        elif len(segments_with_min_items) < min_segments:
            return True
        return False

    def preprocess_MaxItemsPerSegmentConstraint(self, constraint: MaxItemsPerSegmentConstraint, N: int, items: Dict[str, float],
                                                segments: Dict[str, Segment], removed_items: Set[str], already_recommended_items: List[str] = None):
        segment_id = constraint.segment_id
        max_items = constraint.max_items
        window_size = constraint.window_size

        # Compute the maximum possible number of items from the segment
        max_total_items = self.compute_limit_items(N, window_size, max_items)
        if already_recommended_items is not None and window_size > N:
            index = window_size - N
            count_items_in_previous_recomm = len([item for item in already_recommended_items[-index:] if item in segments[segment_id]])
            max_total_items = max(0, max_items - count_items_in_previous_recomm)
            # print(f"[ILP] Segment {segment_id}, max_items: {max_items}, window_size: {window_size}, max_total_items: {max_total_items}, count_items_in_previous_recomm: {count_items_in_previous_recomm}")
        # Get the segment's items
        segment_items = [i for i in segments[segment_id] if i in items]
        sorted_items = sorted(segment_items, key=lambda x: items[x], reverse=True)
        # TODO dont remove items from overlapping segments
        removed = set(sorted_items[max_total_items:])  # Remove the lowest scoring items, keep the top max_total_items
        # print(f"Removed {len(removed)} items")
        removed_items.update(removed)
        # print(f"[ILP] Removing {len(removed)} items from segment {segment_id}, segment num items: {len(segments[segment_id])}, max_items: {max_items}, max_total_items: {max_total_items}")

    def compute_bare_minumum_for_MinItemsPerSegmentConstraint(self, constraint: MinItemsPerSegmentConstraint, N: int,
                                                              segment_bare_minimums=None):
        if segment_bare_minimums is None:
            segment_bare_minimums = {}
        segment_id = constraint.segment_id
        min_items = constraint.min_items
        window_size = constraint.window_size
        segment_bare_minimums[segment_id] = max(segment_bare_minimums.get(segment_id, 0), self.compute_limit_items(N, window_size, min_items))

    def preprocess_MinItemsPerSegmentConstraint(self, constraint: MinItemsPerSegmentConstraint, N: int, segment_bare_minimums: Dict[str, int],
                                                min_required_items: Dict[str, int], min_satisfied: Dict[str, bool], num_remaining_segment_items: int,
                                                remaining_recomm_len: int =None):
        segment_id = constraint.segment_id
        sum_bare_minimums = sum([v for v in segment_bare_minimums.values()])
        # bare minimum + remaining items = bm + (N - bm - all other bare minimums) = bm + N - sum all bare minimums
        required_minimum = max(segment_bare_minimums.get(segment_id, 0), segment_bare_minimums.get(segment_id, 0) + N - sum_bare_minimums)
        if remaining_recomm_len is None:
            min_required_items[segment_id] = required_minimum
            min_satisfied[segment_id] = False
            return
        num_items_to_be_needed = self.compute_limit_items(remaining_recomm_len, constraint.window_size, constraint.min_items)   # how much will we need if there are remaining partitions
        # do not take too many items if there will be not enough to satisfy future constraints
        if num_items_to_be_needed + required_minimum > num_remaining_segment_items:
            min_required_items[segment_id] = max(segment_bare_minimums.get(segment_id, 0), num_remaining_segment_items - num_items_to_be_needed)
        else:
            min_required_items[segment_id] = required_minimum
        min_satisfied[segment_id] = False

    # hard limit for MaxItemsPerSegmentConstraint, worst case for MinItemsPerSegmentConstraint
    # this is the deprecated version, useful for understanding the computation
    def compute_limit_items_old(self, N: int, W: int, item_limit_per_window: int):
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

    def compute_limit_items(self, N: int, W: int, item_limit_per_window: int):
        full_windows = N // W
        remaining_positions = N % W
        max_items_limit = (item_limit_per_window * full_windows) + min(remaining_positions, item_limit_per_window)
        return max_items_limit


    def solve_2D_constraints(self, items: List[Dict[str, float]], segments: Dict[str, Segment], constraints: List[List[Constraint]],
                             constraints2D: List[Constraint2D], N: int):
        start = time.time()

        if self.verbose:
            print(
                f"[{self.name}] Solving ILP with 2D constraints, {len(items)} candidate item pools, {len(segments)} segments,"
                f" {sum(len(c) for c in constraints)} 1D constraints, {len(constraints2D)} 2D constraints, row length={N}.")

        model = Model("RecommenderSystem")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Initialize penalties list
        model._penalties = []
        total_score = 0
        total_penalty = 0
        positions = list(range(1, N + 1)) # positions numbered from 1 to N for convenience
        X = dict() # decision variables for all rows

        # Add row constraints
        for r, item_pool in enumerate(items):
            item_ids = list(item_pool.keys())

            # Create decision variables x[i,r,p] for item i in row r at position p
            x = model.addVars(item_ids, [r], positions, vtype=GRB.BINARY, name="x")
            X.update(x)

            # Constraint 1: Each item is selected at most once per row
            for i in item_ids:
                model.addConstr(
                    quicksum(x[i, r, p] for p in positions) <= 1,
                    name=f"ItemOnce_{i}"
                )

            # Constraint 2: Each row position has at most one item
            for p in positions:
                model.addConstr(
                    quicksum(x[i, r, p] for i in item_ids) <= 1,
                    name=f"PositionOnce_{p}"
                )

            # Constraint 3: Exactly N items are selected per row
            model.addConstr(
                quicksum(x[i, r, p] for i in item_ids for p in positions) == N,
                name="TotalItems"
            )

            # Penalty scaling factor K (total possible score per row)
            K = sum(item_pool.values())

            # Process each constraint in the constraints list for the current row
            for constraint in constraints[r]:
                constraint.add_to_ilp_model(model, x, items[r], segments, r, positions, N, K, already_recommended_items=None)

            # Objective function: Maximize total score - total penalty
            total_score += quicksum(item_pool[i] * x[i, r, p] for i in item_ids for p in positions)
            total_penalty += quicksum(penalty_coeff * s for s, penalty_coeff in model._penalties)

        # Process 2D constraints
        for constraint in constraints2D:
            constraint.add_to_model(model, X, items, positions, len(items), N)

        model.setObjective(total_score - total_penalty, GRB.MAXIMIZE)

        # Optimize the model
        model.optimize()
        result = None

        # Check if the model found an optimal solution
        if model.Status == GRB.OPTIMAL:
            # Extract the solution
            solution = {}
            for r, item_pool in enumerate(items):
                item_ids = list(item_pool.keys())
                for i in item_ids:
                    for p in positions:
                        if X[i, r, p].X > 0.5:
                            solution[(r, p)] = i
            # Return the recommended items sorted by position
            result = {k: solution[k] for k in sorted(solution)}
        elif self.verbose:
            print(f"[{self.name}] No optimal solution found.")

        end = time.time()

        if self.verbose:
            print(f"[{self.name}] Finished in {(end - start) * 1000:.2f} ms")

        return result

    def preprocess_items_2D(self, items: List[Dict[str, float]], segments: Dict[str, Segment], constraints: List[List[Constraint]],
                              constraints2D: List[Constraint2D], N: int, item_segment_map: Dict[str, str]) -> List[Dict[str, float]]:
        start = time.time()

        candidate_items = [dict() for _ in range(len(items))]

        # First preprocess items for each row using each row's constraints
        for r, item_pool in enumerate(items):
            candidate_items[r] = self.preprocess_items(item_pool, segments, segments, constraints[r], item_segment_map,
                                                       N)

        # ensure each item uniqueness 2D constraint has enough items
        for constraint in constraints2D:
            if isinstance(constraint, ItemUniqueness2D):
                self.preprocess_item_uniqueness(constraint, items, candidate_items, N)

        if self.verbose:
            print(f"[ILP] Time taken for 2D item preprocessing: {(time.time() - start)*1000} milliseconds")

        return candidate_items

    # check if there is enough items to satisfy the item uniqueness constraint
    # if not add enough items into candidates to satisfy the constraint
    def preprocess_item_uniqueness(self, constraint: ItemUniqueness2D, item_pools: List[Dict[str, float]],
                                   candidate_items: List[Dict[str, float]], N: int):
        W = constraint.width
        H = constraint.height
        R = len(candidate_items)

        for r in range(R):
            row_above_start = max(0, r - H + 1)
            row_below_end = min(R, r + H)
            number_of_unique_items = self.count_num_unique(candidate_items[r], candidate_items, r, row_above_start, row_below_end)

            # keep adding items until there is enough to satisfy the constraint, or we run out of items
            ordered_items = sorted(item_pools[r].items(), key=lambda x: x[1], reverse=True)
            for item, score in ordered_items:
                if number_of_unique_items >= N:
                    break
                if item in candidate_items[r]:
                    continue
                if self.is_item_unique(item, candidate_items, r, row_above_start, row_below_end):
                    candidate_items[r][item] = score
                    number_of_unique_items += 1

    def count_num_unique(self, items: Dict[str, float], candidate_items: List[Dict[str, float]], r: int, row_above_start: int,
                         row_below_end: int) -> int:
        num_unique = 0
        for item in items:
            if self.is_item_unique(item, candidate_items, r, row_above_start, row_below_end):
                num_unique += 1
        return num_unique

    def is_item_unique(self, item_id, candidate_items: List[Dict[str, float]], r: int, row_above_start: int,
                       row_below_end: int) -> bool:
        for i in range(row_above_start, row_below_end):
            if i == r:
                continue
            if item_id in candidate_items[i]:
                return False
        return True
