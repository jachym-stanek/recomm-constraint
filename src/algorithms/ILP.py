import time

from gurobipy import Model

from src.algorithms.algorithm import Algorithm
from src.constraints import *
from src.algorithms.Preprocessor import ItemPreprocessor


class IlpSolver(Algorithm):
    def __init__(self, name="ILP", description="Integer Linear Programming Solver", verbose=True, time_limit=None):
        super().__init__(name, description, verbose)
        self.time_limit = time_limit # in seconds, None for no limit

    def solve_by_slicing(self, preprocessor: ItemPreprocessor, items: Dict[str, float], segments: Dict[str, Segment],
                         constraints: List[Constraint], N: int, slice_size: int, look_ahead: bool = False):

        start_time = time.time()

        if self.verbose:
            print(f"[{self.name}] Solving ILP using slicing with {len(items)} candidate items, {len(segments)} segments,"
                  f" {len(constraints)} constraints, count={N}, slice size: {slice_size}.")

        if N < slice_size:
            return self.solve(items, segments, constraints, N)

        remaining_items = items.copy()
        final_result = {}
        already_recommended_items = []
        slice_start = 0

        while len(final_result) < N:
            if self.verbose:
                print("===================================== Solving ILP for a slice ======================================")
            slice_N = min(slice_size, N - len(final_result)) # full slice or remaining items
            if look_ahead:
                slice_N = slice_size * 2
            slice_constraints = self._get_constraint_for_a_slice(constraints, slice_N, already_recommended_items)
            slice_candidates = preprocessor.preprocess_items(remaining_items, segments, slice_constraints, slice_N)
            slice_segments = self._get_slice_segments(segments, slice_candidates, already_recommended_items)

            if self.verbose:
                print(f"Already recommended items: {already_recommended_items}")
                # get segments for the already recommended items
                already_recommended_items_to_segments = {}
                for item in already_recommended_items:
                    item_segments = []
                    for segment in segments.values():
                        if item in segment:
                            item_segments.append(segment.id)
                    already_recommended_items_to_segments[item] = item_segments
                for item in already_recommended_items:
                    print(f"{already_recommended_items_to_segments[item]}", end=" ")
                print()

            slice_result = self.solve(slice_candidates, slice_segments, slice_constraints, slice_N, already_recommended_items)

            if slice_result is None:
                if self.verbose:
                    print(f"[{self.name}] No solution found for the slice, returning empty solution.")
                final_result = None
                break

            # take only half
            if look_ahead:
                taken_until = min(slice_size, len(slice_result))
                slice_result = {k: v for k, v in slice_result.items() if k <= taken_until}

            # Add the inner result to the final result and remove the recommended items from the candidate list
            for position, item in slice_result.items():
                final_result[position+slice_start] = item
                already_recommended_items.append(item)
                remaining_items.pop(item)

            slice_start += slice_N

        end_time = time.time()

        if self.verbose:
            print(f"[{self.name}] Partitioning solution finished in {(end_time - start_time) * 1000:.2f} ms")

        return final_result

    """
    We need to adjust constraints for a slice:
     * Hard min constraints might not be possible to be satisfied (e.g. window size is 5, min segments is 4, and slice size is 3)
       thus in this case we need to make them soft in order not to fail (this is not an issue for hard max constraints)
     * If slice size is smaller than window size, we need to also adjust the window size
    """
    def _get_constraint_for_a_slice(self, constraints: List[Constraint], slice_N: int, already_recommended_items: List[str]):
        slice_constraints = []

        for constraint in constraints:
            if isinstance(constraint, MaxItemsPerSegmentConstraint):
                window_size = min(constraint.window_size, slice_N + len(already_recommended_items))
                slice_constraints.append(MaxItemsPerSegmentConstraint(constraint.segment_id, constraint.property, constraint.max_items,
                                                                          window_size, weight=constraint.weight))
            elif isinstance(constraint, GlobalMaxItemsPerSegmentConstraint):
                window_size = min(constraint.window_size, slice_N + len(already_recommended_items))
                slice_constraints.append(GlobalMaxItemsPerSegmentConstraint(constraint.segmentation_property, constraint.max_items,
                                                       window_size, weight=constraint.weight))
            elif isinstance(constraint, MinItemsPerSegmentConstraint):
                window_size = min(constraint.window_size, slice_N + len(already_recommended_items))
                slice_constraint_weight = 0.9 if constraint.weight == 1 and constraint.window_size > slice_N + len(already_recommended_items) else constraint.weight  # make min constraints soft if they are hard
                slice_constraints.append(MinItemsPerSegmentConstraint(constraint.segment_id, constraint.property, constraint.min_items,
                                                                          window_size,
                                                                          weight=slice_constraint_weight))
            elif isinstance(constraint, GlobalMinItemsPerSegmentConstraint):
                window_size = min(constraint.window_size, slice_N + len(already_recommended_items))
                slice_constraint_weight = 0.9 if constraint.weight == 1 and constraint.window_size > slice_N + len(already_recommended_items) else constraint.weight
                slice_constraints.append(GlobalMinItemsPerSegmentConstraint(constraint.segmentation_property, constraint.min_items,
                                                       window_size, weight=slice_constraint_weight))
            elif isinstance(constraint, MaxSegmentsConstraint):
                window_size = min(constraint.window_size, slice_N + len(already_recommended_items))
                slice_constraints.append(MaxSegmentsConstraint(constraint.segmentation_property, constraint.max_segments,
                                                       window_size, weight=constraint.weight))
            elif isinstance(constraint, MinSegmentsConstraint):
                window_size = min(constraint.window_size, slice_N + len(already_recommended_items))
                slice_constraint_weight = 0.9 if constraint.weight == 1 and constraint.window_size > slice_N + len(already_recommended_items) else constraint.weight
                slice_constraints.append(MinSegmentsConstraint(constraint.segmentation_property, constraint.min_segments,
                                                       window_size, weight=slice_constraint_weight))
            else:
                slice_constraints.append(constraint)

        return slice_constraints

    """
    remove items that are not candidates or already recommended from segments
    """
    def _get_slice_segments(self, segments: Dict[str, Segment], items: Dict[str, float], already_recommended_items: List[str]):
        needed_items = set(items.keys()).union(set(already_recommended_items))
        reduced_segments = {}
        for segment_label, segment in segments.items():
            reduced_segments[segment_label] = Segment(segment.id, segment.property, *set(segment) & needed_items)
        return reduced_segments

    def _remove_item_from_segments(self, segments: Dict[str, Segment], item: str):
        for segment in segments.values():
            if item in segment:
                segment.remove(item)
        return segments

    def _deep_copy_segments(self, segments: Dict[str, Segment]):
        """
        Create a deep copy of the segments dictionary.
        :param segments: dict<seg_label, Segment>
        :return: dict<seg_label, Segment>
        """
        return {seg_label: Segment(segment.id, segment.property, *segment) for seg_label, segment in segments.items()}

    def solve(self, items: Dict[str, float], segments: Dict[str, Segment], constraints: List[Constraint], N: int,
              already_recommended_items: List[str] = None, return_first_feasible: bool = False, num_threads: int = 0):
        start = time.time()

        if self.verbose:
            print(f"[{self.name}] Solving ILP with {len(items)} candidate items, {len(segments)} segments, {len(constraints)} constraints, count={N}.")

        model = Model("RecommenderSystem")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        if self.time_limit is not None:
            if self.verbose:
                print(f"[{self.name}] Setting time limit to {self.time_limit} seconds.")
            model.setParam("TimeLimit", self.time_limit)

        # set the number of threads to use
        if num_threads > 0:
            model.setParam("Threads", num_threads)
            if self.verbose:
                print(f"[{self.name}] Setting number of threads to {num_threads}.")

        if return_first_feasible:
            model.setParam("SolutionLimit", 1)

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

        # Penalty scaling factor K (sum of N most scoring items = max possible score)
        K = sum(sorted(items.values(), reverse=True)[:N])

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
        if (model.Status == GRB.OPTIMAL or (return_first_feasible and model.Status == GRB.SOLUTION_LIMIT)
               or (self.time_limit is not None and model.Status == GRB.TIME_LIMIT)) and model.SolCount > 0:
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
