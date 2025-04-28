import time

from gurobipy import Model

from src.algorithms.algorithm import Algorithm
from src.constraints import *
from src.algorithms.Preprocessor import ItemPreprocessor


class IlpSolver(Algorithm):
    def __init__(self, name="ILP", description="Integer Linear Programming Solver", verbose=True, time_limit=None):
        super().__init__(name, description, verbose)
        self.time_limit = time_limit # in seconds, None for no limit

    def solve_by_slicing(self, item_preprocessor: ItemPreprocessor, items: Dict[str, float], segments: Dict[str, Segment],
                         constraints: List[Constraint], N: int, partition_size: int,
                         item_segment_map: Dict[str, str], look_ahead: bool = False):
        start_time = time.time()

        temp_item_segment_map = item_segment_map.copy()
        items_remaining_per_segment = {segment_id: Segment(segment_id, segment.segmentation_property, *set(segment)) for segment_id, segment in segments.items()}

        if self.verbose:
            print(f"[{self.name}] Solving ILP using slicing with {len(items)} candidate items, {len(segments)} segments,"
                  f" {len(constraints)} constraints, count={N}, partition size: {partition_size}.")

        if N < partition_size:
            return self.solve(items, segments, constraints, N)

        candidates = items.copy()
        final_result = {}
        already_recommended_items = []
        partition_start = 0

        while len(final_result) < N:
            if self.verbose:
                print("===================================== Slicing ======================================")
            if look_ahead:
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

            partition_candidates = item_preprocessor.preprocess_items(candidates, items_remaining_per_segment, segments, constraints, N=partition_count,
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
            if look_ahead:
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
