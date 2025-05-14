from typing import Dict, List, Set

from gurobipy import GRB, quicksum

from src.segmentation import Segment


class Constraint:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight  # weight in [0, 1]

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        raise NotImplementedError("Must implement add_to_model method.")

    def add_to_cp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        raise NotImplementedError(f"Must implement add_to_model_cp method for type {type(self)}.")

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        raise NotImplementedError("Must implement check_constraint method.")

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        """
        Return value in [0, 1] reflecting how well solution meets the constraint
        1 means fully satisfied, 0 worst violation
        """
        raise NotImplementedError("Must implement satisfaction_ratio method.")


class Constraint2D:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight  # weight in [0, 1]

    def add_to_model(self, model, x, items, positions, num_rows, num_cols):
        raise NotImplementedError("Must implement add_to_model method.")

    def check_constraint(self, solution, num_rows, num_cols):
        raise NotImplementedError("Must implement check_constraint method.")


class MinItemsPerSegmentConstraint(Constraint):
    def __init__(self, segment_id, item_property, min_items, window_size, name="MinItemsPerSegment", weight=1.0):
        name = f"{name}_{segment_id}_{item_property}_{min_items}_{window_size}"
        super().__init__(name, weight)
        self.segment_id = segment_id
        self.property = item_property
        self.min_items = min_items
        self.window_size = window_size
        self.label = f"{self.segment_id}-{self.property}"

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_items = segments[self.label]

        if self.weight < 1.0:
            s = _add_slack_variable(model, K, self.weight, self.name)

        # constraint on recomm position
        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]
            if self.weight < 1.0:
                # Soft constraint: Introduce slack variable
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) + s >= self.min_items,
                    name=f"{self.name}_{i}"
                )
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) >= self.min_items,
                    name=f"{self.name}_{i}"
                )

        # constraint including the already recommended items
        if already_recommended_items: # TODO: refactor this to merge with the previous loop
            counter_start = self.window_size - N if (N < self.window_size < len(already_recommended_items) + N) else 1
            counter_end = min(self.window_size, len(already_recommended_items) + 1)
            for i in range(counter_start, counter_end):
                recomm_positions = positions[:self.window_size-i] # positions in the recommendation that are not already recommended

                # count the number of items from the segment in the already recommended items that are in the window
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)

                if self.weight < 1.0:
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions)
                        + num_already_recommended + s >= self.min_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )
                else:
                    # Hard constraint
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions)
                        + num_already_recommended >= self.min_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )

    def add_to_cp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # Use the Segment object for this constraint.
        seg = segments[self.label]
        for i in range(N - self.window_size + 1):
            window = positions[i : i + self.window_size]
            # Sum over all candidate items in the segment for positions in the window.
            model.Add(sum(x[item, p] for item in items if item in seg for p in window) >= self.min_items)

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        # Get the indicator list for this segment.
        indicator_list = solver_data["segment_indicator"][self.label]
        for i in range(N - self.window_size + 1):
            b_vars = []
            for p in range(i, i + self.window_size):
                b = model.NewIntVar(0, 1, f"min_{self.label}_{i}_{p}")
                model.AddElement(assign[p], indicator_list, b)
                b_vars.append(b)
            model.Add(sum(b_vars) >= self.min_items)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        if type(solution) is dict:
            solution = list(solution.values())

        N = len(solution)
        segment_items = segments[self.label]
        for i in range(N - self.window_size + 1):
            window = solution[i:i + self.window_size]
            count = sum(1 for item_id in window if item_id in segment_items)
            if count < self.min_items:
                return False

        # check the already recommended items
        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items))
            for i in range(counter_start, counter_end):
                recomm_positions = solution[:self.window_size-i]
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)
                count = sum(1 for item_id in recomm_positions if item_id in segment_items) + num_already_recommended
                if count < self.min_items:
                    return False
        return True

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        if isinstance(solution, dict):
            solution = list(solution.values())

        segment_items = segments[self.label]
        N = len(solution)
        worst_deficit = 0

        # sliding windows over the recommendation
        for i in range(N - self.window_size + 1):
            window = solution[i: i + self.window_size]
            cnt = sum(1 for it in window if it in segment_items)
            worst_deficit = max(worst_deficit, max(0, self.min_items - cnt))

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items))
            for i in range(counter_start, counter_end):
                recom_window = solution[: self.window_size - i]
                already_cnt = sum(1 for it in already_recommended_items[-i:] if it in segment_items)
                cnt = sum(1 for it in recom_window if it in segment_items) + already_cnt
                worst_deficit = max(worst_deficit, max(0, self.min_items - cnt))

        return _linear_ratio(worst_deficit, self.min_items)

    def __repr__(self):
        return f"{self.name}(segment_id={self.segment_id}, property={self.property}, min_items={self.min_items}, window_size={self.window_size})"


class MaxItemsPerSegmentConstraint(Constraint):
    def __init__(self, segment_id, item_property, max_items, window_size, name="MaxItemsPerSegment", weight=1.0):
        name = f"{name}_{segment_id}_{item_property}_{max_items}_{window_size}"
        super().__init__(name, weight)
        self.segment_id = segment_id
        self.property = item_property
        self.max_items = max_items
        self.window_size = window_size
        self.label = f"{self.segment_id}-{self.property}"

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_items = segments[self.label]

        if self.weight < 1.0:
            s = _add_slack_variable(model, K, self.weight, self.name)

        # constraint on recomm position
        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]
            if self.weight < 1.0:
                # Soft constraint: Introduce slack variable
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) - s <= self.max_items,
                    name=f"{self.name}_{i}"
                )
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) <= self.max_items,
                    name=f"{self.name}_{i}"
                )

        # constraint including the already recommended items
        if already_recommended_items:
            counter_start = self.window_size - N if (N < self.window_size < len(already_recommended_items) + N) else 1
            counter_end = min(self.window_size, len(already_recommended_items) + 1) # take either last W-1 or |AR| positions
            for i in range(counter_start, counter_end):
                recomm_positions = positions[:self.window_size-i]

                # count the number of items from the segment in the already recommended items that are in the window
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)

                if self.weight < 1.0:
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions) + num_already_recommended - s <= self.max_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )
                else:
                    # Hard constraint
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions) + num_already_recommended <= self.max_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )

    def add_to_cp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        seg = segments[self.label]
        for i in range(N - self.window_size + 1):
            window = positions[i : i + self.window_size]
            model.Add(sum(x[item, p] for item in items if item in seg for p in window) <= self.max_items)

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        indicator_list = solver_data["segment_indicator"][self.label]
        for i in range(N - self.window_size + 1):
            b_vars = []
            for p in range(i, i + self.window_size):
                b = model.NewIntVar(0, 1, f"max_{self.segment_id}_{i}_{p}")
                model.AddElement(assign[p], indicator_list, b)
                b_vars.append(b)
            model.Add(sum(b_vars) <= self.max_items)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        if type(solution) is dict:
            solution = list(solution.values())

        N = len(solution)
        segment_items = segments[self.label]
        for i in range(N - self.window_size + 1):
            window = solution[i:i + self.window_size]
            count = sum(1 for item_id in window if item_id in segment_items)
            if count > self.max_items:
                return False

        # check the already recommended items
        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items))
            for i in range(counter_start, counter_end):
                recomm_positions = solution[:self.window_size-i]
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)
                count = sum(1 for item_id in recomm_positions if item_id in segment_items) + num_already_recommended
                if count > self.max_items:
                    return False
        return True

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        if isinstance(solution, dict):
            solution = list(solution.values())

        segment_items = segments[self.label]
        N = len(solution)
        worst_excess = 0

        for i in range(N - self.window_size + 1):
            window = solution[i: i + self.window_size]
            cnt = sum(1 for it in window if it in segment_items)
            worst_excess = max(worst_excess, max(0, cnt - self.max_items))

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items))
            for i in range(counter_start, counter_end):
                recom_window = solution[: self.window_size - i]
                already_cnt = sum(1 for it in already_recommended_items[-i:] if it in segment_items)
                cnt = sum(1 for it in recom_window if it in segment_items) + already_cnt
                worst_excess = max(worst_excess, max(0, cnt - self.max_items))

        return _linear_ratio(worst_excess, self.max_items)

    def __repr__(self):
        return f"{self.name}(segment_id={self.segment_id}, property={self.property}, max_items={self.max_items}, window_size={self.window_size})"


class ItemFromSegmentAtPositionConstraint(Constraint):
    def __init__(self, segment_id, item_property, position, name="ItemFromSegmentAtPosition", weight=1.0):
        name = f"{name}_{segment_id}_{item_property}_{position}"
        super().__init__(name, weight)
        self.segment_id = segment_id
        self.property = item_property
        self.position = position
        self.label = f"{self.segment_id}-{self.property}"

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_items = segments[self.label]
        if self.weight < 1.0:
            # Soft constraint: Introduce slack variable
            s = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{self.position}")
            model.addConstr(
                quicksum(x[i, row, self.position] for i in segment_items) + s >= 1,
                name=f"{self.name}_{self.position}"
            )
            penalty_coeff = K * self.weight / (1 - self.weight)
            model._penalties.append((s, penalty_coeff))
        else:
            # Hard constraint
            model.addConstr(
                quicksum(x[i, row, self.position] for i in segment_items) >= 1,
                name=f"{self.name}_{self.position}"
            )

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        indicator_list = solver_data["segment_indicator"][self.label]
        b = model.NewIntVar(0, 1, f"fromseg_{self.label}_{self.position}")
        model.AddElement(assign[self.position], indicator_list, b)
        model.Add(b == 1)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        segment_items = segments[self.label]
        if type(solution) is dict:
            item_id = solution.get(self.position)
        else:
            item_id = solution[self.position-1] # position is 1-indexed
        return item_id in segment_items

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        return 1.0 if self.check_constraint(solution, items, segments, already_recommended_items) else 0.0

    def __repr__(self):
        return f"{self.name}(segment_id={self.segment_id}, property={self.property}, position={self.position})"


class ItemAtPositionConstraint(Constraint):
    def __init__(self, item_id, position, name="ItemAtPosition", weight=1.0):
        name = f"{name}_{item_id}_{position}"
        super().__init__(name, weight)
        self.item_id = item_id
        self.position = position

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        if self.weight < 1.0:
            # Soft constraint: Introduce slack variable
            s = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{self.position}")
            model.addConstr(
                x[self.item_id, row, self.position] + s >= 1,
                name=f"{self.name}_{self.position}"
            )
            penalty_coeff = K * self.weight / (1 - self.weight)
            model._penalties.append((s, penalty_coeff))
        else:
            # Hard constraint
            model.addConstr(
                x[self.item_id, row, self.position] >= 1,
                name=f"{self.name}_{self.position}"
            )

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        candidate_val = solver_data["candidate_to_int"][self.item_id]
        model.Add(assign[self.position] == candidate_val)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        if type(solution) is dict:
            return solution.get(self.position) == self.item_id
        else:
            return solution[self.position-1] == self.item_id

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        return 1.0 if self.check_constraint(solution, items, segments, already_recommended_items) else 0.0

    def __repr__(self):
        return f"{self.name}(item_id={self.item_id}, position={self.position})"


class GlobalMinItemsPerSegmentConstraint(Constraint):
    """
    Minimum nuber of items from each segment that belongs to segmentation of target property
    E.g. Final recommendation should contain at least 2 items from every genre present in the candidate items
    """

    def __init__(self, segmentation_property, min_items, window_size, weight=1.0, name="GlobalMinItemsPerSegment", verbose=False):
        name = f"{name}_{segmentation_property}_{min_items}_{window_size}"
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.min_items = min_items
        self.window_size = window_size
        self.constraints = [] # List of MinItemsPerSegmentConstraint for each segment with min_items and window_size = N
        self.verbose = verbose

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # ensure that the constraints are empty
        if self.constraints:
            self.constraints = []

        # create MinItemsPerSegmentConstraint for each segment
        for segment_label in segments:
            if segments[segment_label].property == self.segmentation_property:
                segment_id = segments[segment_label].id
                if self.verbose:
                    print(f"[{self.name}]Adding constraint for segment {segment_id}, property {self.segmentation_property}")
                constraint = MinItemsPerSegmentConstraint(segment_id, self.segmentation_property, self.min_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_ilp_model(model, x, items, segments, row, positions, N, K, already_recommended_items)

    def add_to_cp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # Loop through all segments and add a per-segment minimum if the property matches.
        for seg in segments.values():
            if seg.property == self.segmentation_property:
                constraint = MinItemsPerSegmentConstraint(seg.id, seg.property, self.min_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_cp_model(model, x, items, segments, row, positions, N, K, already_recommended_items)

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        seg_dict = solver_data["global_segments"].get(self.segmentation_property, {})
        for seg_id, indicator_list in seg_dict.items():
            for i in range(N - self.window_size + 1):
                b_vars = []
                for p in range(i, i + self.window_size):
                    b = model.NewIntVar(0, 1, f"globalmin_{seg_id}_{i}_{p}")
                    model.AddElement(assign[p], indicator_list, b)
                    b_vars.append(b)
                model.Add(sum(b_vars) >= self.min_items)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        if not self.constraints:
            self.constraints = self.sub_constraints_from_segments(segments)
        return all(constraint.check_constraint(solution, items, segments, already_recommended_items) for constraint in self.constraints)

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        if not self.constraints:
            self.constraints = self.sub_constraints_from_segments(segments)
        return sum(c.satisfaction_ratio(solution, items, segments, already_recommended_items) for c in self.constraints) / len(self.constraints)

    def sub_constraints_from_segments(self, segments):
        constraints = []
        for segment_label in segments:
            if segments[segment_label].property== self.segmentation_property:
                segment_id = segments[segment_label].id
                constraint = MinItemsPerSegmentConstraint(segment_id, self.segmentation_property, self.min_items, self.window_size, weight=self.weight)
                constraints.append(constraint)

        return constraints

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, min_items={self.min_items}, window_size={self.window_size})"


class GlobalMaxItemsPerSegmentConstraint(Constraint):
    """
    Maximum nuber of items from each segment that belongs to segmentation of target property
    E.g. Final recommendation should contain at most 2 items from every genre present in the candidate items
    """

    def __init__(self, segmentation_property, max_items, window_size, weight=1.0, name="GlobalMaxItemsPerSegment", verbose=False):
        name = f"{name}_{segmentation_property}_{max_items}_{window_size}"
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.max_items = max_items
        self.window_size = window_size
        self.constraints = [] # List of MaxItemsPerSegmentConstraint for each segment with max_items and window_size = N
        self.verbose = verbose

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # ensure that the constraints are empty
        if self.constraints:
            self.constraints = []

        # create MaxItemsPerSegmentConstraint for each segment
        for segment_label in segments:
            if segments[segment_label].property == self.segmentation_property:
                segment_id = segments[segment_label].id
                if self.verbose:
                    print(f"[{self.name}]Adding constraint for segment {segment_id}, property {self.segmentation_property}")
                constraint = MaxItemsPerSegmentConstraint(segment_id, self.segmentation_property, self.max_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_ilp_model(model, x, items, segments, row, positions, N, K, already_recommended_items)

    def add_to_cp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # Loop through all segments and add a per-segment maximum if the property matches.
        for seg in segments.values():
            if seg.property == self.segmentation_property:
                constraint = MaxItemsPerSegmentConstraint(seg.id, seg.property, self.max_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_cp_model(model, x, items, segments, row, positions, N, K, already_recommended_items)

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        seg_dict = solver_data["global_segments"].get(self.segmentation_property, {})
        for seg_id, indicator_list in seg_dict.items():
            for i in range(N - self.window_size + 1):
                b_vars = []
                for p in range(i, i + self.window_size):
                    b = model.NewIntVar(0, 1, f"globalmax_{seg_id}_{i}_{p}")
                    model.AddElement(assign[p], indicator_list, b)
                    b_vars.append(b)
                model.Add(sum(b_vars) <= self.max_items)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        if not self.constraints:
            self.constraints = self.sub_constraints_from_segments(segments)
        return all(constraint.check_constraint(solution, items, segments, already_recommended_items) for constraint in self.constraints)

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        if not self.constraints:
            self.constraints = self.sub_constraints_from_segments(segments)
        return sum(c.satisfaction_ratio(solution, items, segments, already_recommended_items) for c in self.constraints) / len(self.constraints)

    def sub_constraints_from_segments(self, segments):
        constraints = []
        for segment_label in segments:
            if segments[segment_label].property == self.segmentation_property:
                segment_id = segments[segment_label].id
                constraint = MaxItemsPerSegmentConstraint(segment_id, self.segmentation_property, self.max_items, self.window_size, weight=self.weight)
                constraints.append(constraint)

        return constraints

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, max_items={self.max_items}, window_size={self.window_size})"


class MinSegmentsConstraint(Constraint):
    """
    Minimum number of segments that should be represented in the final recommendation
    E.g. Final recommendation should contain at least 2 different genres in every window of size W
    """

    def __init__(self, segmentation_property, min_segments, window_size, name="MinSegmentDiversity", weight=1.0, verbose=False):
        name = f"{name}_{segmentation_property}_{min_segments}_{window_size}"
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.min_segments = min_segments
        self.window_size = window_size
        self.verbose = verbose

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_labels = [segment_label for segment_label, s in segments.items() if s.property == self.segmentation_property]

        # binary variable for each segment and window y_{segment_id, i} = 1 if segment_id is represented in the window
        window_starts = range(N - self.window_size + 1)
        y = model.addVars([row], segment_labels, window_starts, vtype=GRB.BINARY, name=f"y_{self.name}")

        if self.weight < 1.0:
            s = _add_slack_variable(model, K, self.weight, self.name)

        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]

            # set constraints on y (ensure y=1 if any segment item is in the window and y=0 otherwise)
            for segment_label in segment_labels:
                segment_items = segments[segment_label]
                model.addConstr(
                    y[row, segment_label, i] <= quicksum(x[item_id, row, p] for item_id in items if item_id in segment_items for p in window),
                    name=f"y_{self.name}_{row}_{segment_label}_{i}"
                )
                for item_id in items:
                    if item_id in segment_items:
                        for p in window:
                            model.addConstr(
                                y[row, segment_label, i] >= x[item_id, row, p],
                                name=f"y_{self.name}_{row}_{segment_label}_{i}_{item_id}"
                            )

            # constraint on the number of segments in the window
            if self.weight < 1.0:
                model.addConstr(
                    quicksum(y[row, segment_label, i] for segment_label in segment_labels) + s >= self.min_segments,
                    name=f"{self.name}_{i}"
                )
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(y[row, segment_label, i] for segment_label in segment_labels) >= self.min_segments,
                    name=f"{self.name}_{i}"
                )

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items) + 1)
            for i in range(counter_start, counter_end):
                recomm_positions = positions[:self.window_size - i]
                constant = {}
                for segment_label in segment_labels:
                    already_present = 1 if any(
                        item_id in segments[segment_label] for item_id in already_recommended_items[-i:]) else 0
                    constant[segment_label] = already_present
                const_sum = sum(constant.values())
                new_y_vars = {}
                for segment_label in segment_labels:
                    if constant[segment_label] == 0:
                        new_y_vars[segment_label] = model.addVar(vtype=GRB.BINARY,
                                                              name=f"y_{self.name}_already_{segment_label}_{i}")
                        model.addConstr(
                            new_y_vars[segment_label] <= quicksum(
                                x[item_id, row, p] for item_id in items if item_id in segments[segment_label] for p in recomm_positions),
                            name=f"y_{self.name}_already_constr1_{segment_label}_{i}"
                        )
                        for item_id in segments[segment_label]:
                            if item_id in items:
                                for p in recomm_positions:
                                    model.addConstr(
                                        new_y_vars[segment_label] >= x[item_id, row, p],
                                        name=f"y_{self.name}_already_constr2_{segment_label}_{i}_{item_id}"
                                    )
                # For the effective window, the count is constant (from already recommended items) plus contributions from the new part.
                if self.weight < 1.0:
                    model.addConstr(
                        quicksum(new_y_vars[segment_label] for segment_label in new_y_vars) + const_sum + s >= self.min_segments,
                        name=f"{self.name}_already_{i}"
                    )
                else:
                    model.addConstr(
                        quicksum(new_y_vars[segment_label] for segment_label in new_y_vars) + const_sum >= self.min_segments,
                        name=f"{self.name}_already_{i}"
                    )

    def add_to_cp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # Consider only segments that match the property.
        seg_list = [seg for seg in segments.values() if seg.segmentation_property == self.segmentation_property]
        for i in range(N - self.window_size + 1):
            window = positions[i: i + self.window_size]
            y_vars = {}
            for seg in seg_list:
                y_var = model.NewBoolVar(f"minseg_{seg.id}_{i}")
                model.Add(sum(x[item, p] for item in items if item in seg for p in window) >= 1).OnlyEnforceIf(y_var)
                model.Add(sum(x[item, p] for item in items if item in seg for p in window) == 0).OnlyEnforceIf(y_var.Not())
                y_vars[seg.id] = y_var
            model.Add(sum(y_vars[seg.id] for seg in seg_list) >= self.min_segments)

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        # Use global_segments to get all segments with the given property.
        seg_dict = solver_data["global_segments"].get(self.segmentation_property, {})
        for i in range(N - self.window_size + 1):
            y_vars = []
            for seg_id, indicator_list in seg_dict.items():
                b_vars = []
                for p in range(i, i + self.window_size):
                    b = model.NewIntVar(0, 1, f"minseg_{seg_id}_{i}_{p}")
                    model.AddElement(assign[p], indicator_list, b)
                    b_vars.append(b)
                y = model.NewBoolVar(f"minseg_y_{seg_id}_{i}")
                model.Add(sum(b_vars) >= 1).OnlyEnforceIf(y)
                model.Add(sum(b_vars) == 0).OnlyEnforceIf(y.Not())
                y_vars.append(y)
            model.Add(sum(y_vars) >= self.min_segments)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        if type(solution) is dict:
            solution = list(solution.values())

        segment_labels = [segment_label for segment_label, s in segments.items() if s.property == self.segmentation_property]

        N = len(solution)
        for i in range(N - self.window_size + 1):
            window = solution[i:i + self.window_size]
            segments_in_window = _segments_in_window(window, segments, segment_labels)
            if len(segments_in_window) < self.min_segments:
                return False

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items) + 1)
            for i in range(counter_start, counter_end):
                recomm_positions = solution[:self.window_size - i]
                already_recommended_item_in_window = already_recommended_items[-i:]
                combined_window = recomm_positions + already_recommended_item_in_window
                segments_in_window = _segments_in_window(combined_window, segments, segment_labels)
                if len(segments_in_window) < self.min_segments:
                    return False

        return True

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        if isinstance(solution, dict):
            solution = list(solution.values())

        segment_labels = [segment_label for segment_label, s in segments.items() if s.property == self.segmentation_property]
        N = len(solution)
        worst_deficit = 0

        for i in range(N - self.window_size + 1):
            window = solution[i: i + self.window_size]
            deficit = self.min_segments - len(_segments_in_window(window, segments, segment_labels))
            worst_deficit = max(worst_deficit, max(0, deficit))

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items)) + 1
            for i in range(counter_start, counter_end):
                recom_window = solution[: self.window_size - i]
                combined = recom_window + already_recommended_items[-i:]
                deficit = self.min_segments - len(_segments_in_window(combined, segments, segment_labels))
                worst_deficit = max(worst_deficit, max(0, deficit))

        return _linear_ratio(worst_deficit, self.min_segments)

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, min_segments={self.min_segments}, window_size={self.window_size})"


class MaxSegmentsConstraint(Constraint):
    """
    Maximum number of segments that should be represented in the final recommendation
    E.g. Final recommendation should contain at most 2 different genres in every window of size W
    """

    def __init__(self, segmentation_property, max_segments, window_size, name="MaxSegmentDiversity", weight=1.0, verbose=False):
        name = f"{name}_{segmentation_property}_{max_segments}_{window_size}"
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.max_segments = max_segments
        self.window_size = window_size
        self.verbose = verbose

    def add_to_ilp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_labels = [segment_label for segment_label, s in segments.items() if s.property == self.segmentation_property]

        # binary variable for each segment and window y_{segment_id, i} = 1 if segment_id is represented in the window
        window_starts = range(N - self.window_size + 1)
        y = model.addVars([row], segment_labels, window_starts, vtype=GRB.BINARY, name=f"y_{self.name}")

        if self.weight < 1.0:
            s = _add_slack_variable(model, K, self.weight, self.name)

        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]

            # set constraints on y (ensure y=1 if any segment item is in the window and y=0 otherwise)
            for segment_label in segment_labels:
                segment_items = segments[segment_label]
                model.addConstr(
                    y[row, segment_label, i] <= quicksum(x[item_id, row, p] for item_id in items if item_id in segment_items for p in window),
                    name=f"y_{self.name}_{row}_{segment_label}_{i}"
                )
                for item_id in items:
                    if item_id in segment_items:
                        for p in window:
                            model.addConstr(
                                y[row, segment_label, i] >= x[item_id, row, p],
                                name=f"y_{self.name}_{row}_{segment_label}_{i}_{item_id}"
                            )

            # constraint on the number of segments in the window
            if self.weight < 1.0:
                model.addConstr(
                    quicksum(y[row, segment_label, i] for segment_label in segment_labels) - s <= self.max_segments,
                    name=f"{self.name}_{i}"
                )
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(y[row, segment_label, i] for segment_label in segment_labels) <= self.max_segments,
                    name=f"{self.name}_{i}"
                )

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items) + 1)
            for i in range(counter_start, counter_end):
                recomm_positions = positions[:self.window_size - i]
                constant = {}
                for segment_label in segment_labels:
                    already_present = 1 if any(
                        item_id in segments[segment_label] for item_id in already_recommended_items[-i:]) else 0
                    constant[segment_label] = already_present
                const_sum = sum(constant.values())
                new_y_vars = {}
                for segment_label in segment_labels:
                    if constant[segment_label] == 0: # only create a new var if the segment is not already present
                        new_y_vars[segment_label] = model.addVar(vtype=GRB.BINARY,
                                                              name=f"y_{self.name}_already_{segment_label}_{i}")
                        model.addConstr(
                            new_y_vars[segment_label] <= quicksum(
                                x[item_id, row, p] for item_id in items if item_id in segments[segment_label] for p in recomm_positions),
                            name=f"y_{self.name}_already_constr1_{segment_label}_{i}"
                        )
                        for item_id in segments[segment_label]:
                            if item_id in items:
                                for p in recomm_positions:
                                    model.addConstr(
                                        new_y_vars[segment_label] >= x[item_id, row, p],
                                        name=f"y_{self.name}_already_constr2_{segment_label}_{i}_{item_id}"
                                    )
                # For the effective window, the count is constant (from already recommended items) plus contributions from the new part.
                if self.weight < 1.0:
                    model.addConstr(
                        const_sum + quicksum(new_y_vars[seg] for seg in new_y_vars) - s <= self.max_segments,
                        name=f"{self.name}_already_{i}"
                    )
                else:
                    model.addConstr(
                        const_sum + quicksum(new_y_vars[seg] for seg in new_y_vars) <= self.max_segments,
                        name=f"{self.name}_already_{i}"
                    )

    def add_to_cp_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        seg_list = [seg for seg in segments.values() if seg.segmentation_property == self.segmentation_property]
        for i in range(N - self.window_size + 1):
            window = positions[i: i + self.window_size]
            y_vars = {}
            for seg in seg_list:
                y_var = model.NewBoolVar(f"maxseg_{seg.id}_{i}")
                model.Add(sum(x[item, p] for item in items if item in seg for p in window) >= 1).OnlyEnforceIf(y_var)
                model.Add(sum(x[item, p] for item in items if item in seg for p in window) == 0).OnlyEnforceIf(y_var.Not())
                y_vars[seg.id] = y_var
            model.Add(sum(y_vars[seg.id] for seg in seg_list) <= self.max_segments)

    def add_to_permutation_cp_model(self, model, assign, solver_data, N):
        seg_dict = solver_data["global_segments"].get(self.segmentation_property, {})
        for i in range(N - self.window_size + 1):
            y_vars = []
            for seg_id, indicator_list in seg_dict.items():
                b_vars = []
                for p in range(i, i + self.window_size):
                    b = model.NewIntVar(0, 1, f"maxseg_{seg_id}_{i}_{p}")
                    model.AddElement(assign[p], indicator_list, b)
                    b_vars.append(b)
                y = model.NewBoolVar(f"maxseg_y_{seg_id}_{i}")
                model.Add(sum(b_vars) >= 1).OnlyEnforceIf(y)
                model.Add(sum(b_vars) == 0).OnlyEnforceIf(y.Not())
                y_vars.append(y)
            model.Add(sum(y_vars) <= self.max_segments)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        if type(solution) is dict:
            solution = list(solution.values())

        segment_labels = [segment_label for segment_label, s in segments.items() if s.property == self.segmentation_property]

        N = len(solution)
        for i in range(N - self.window_size + 1):
            window = solution[i:i + self.window_size]
            segments_in_window = _segments_in_window(window, segments, segment_labels)
            if len(segments_in_window) > self.max_segments:
                return False

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items) + 1)
            for i in range(counter_start, counter_end):
                recomm_positions = solution[:self.window_size - i]
                already_recommended_item_in_window = already_recommended_items[-i:]
                combined_window = recomm_positions + already_recommended_item_in_window
                segments_in_window = _segments_in_window(combined_window, segments, segment_labels)
                if len(segments_in_window) > self.max_segments:
                    return False

        return True

    def satisfaction_ratio(self, solution, items, segments, already_recommended_items=None) -> float:
        if isinstance(solution, dict):
            solution = list(solution.values())

        segment_labels = [segment_label for segment_label, s in segments.items() if s.property == self.segmentation_property]
        N = len(solution)
        worst_excess = 0

        for i in range(N - self.window_size + 1):
            window = solution[i: i + self.window_size]
            excess = len(_segments_in_window(window, segments, segment_labels)) - self.max_segments
            worst_excess = max(worst_excess, max(0, excess))

        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items)) + 1
            for i in range(counter_start, counter_end):
                recom_window = solution[: self.window_size - i]
                combined = recom_window + already_recommended_items[-i:]
                excess = len(_segments_in_window(combined, segments, segment_labels)) - self.max_segments
                worst_excess = max(worst_excess, max(0, excess))

        return _linear_ratio(worst_excess, self.max_segments)

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, max_segments={self.max_segments}, window_size={self.window_size})"

class ItemUniqueness2D(Constraint2D):
    def __init__(self, width, height, name="ItemUniqueness2D", weight=1.0):
        super().__init__(name, weight)
        self.width = width    # 2D sliding window width
        self.height = height  # 2D sliding window height

    """
    In every window of size width x height, each item can appear at most once
    Each row of the output matrix is filled with items from a different item pool
    """
    def add_to_model(self, model, x, items, positions, num_rows, num_cols):
        for window_start_row in range(num_rows - self.height + 1):
            for window_start_col in range(num_cols - self.width + 1):
                window_positions = positions[window_start_col:window_start_col + self.width]
                window_rows = range(window_start_row, window_start_row + self.height)
                for row in window_rows:
                    for i in items[row].keys():
                        # every item can appear at most once in the window (items can be repeated in different row item pools)
                        model.addConstr(
                            quicksum(x[i, r, p] for r in window_rows for p in window_positions if i in items[r].keys()) <= 1,
                            name=f"{self.name}_{window_start_row}_{window_start_col}_{i}"
                        )

    def check_constraint(self, solution: dict, num_rows, num_cols):
        for window_start_row in range(num_rows - self.height + 1):
            for window_start_col in range(num_cols - self.width + 1):
                items_in_window = set()
                for r in range(window_start_row, window_start_row + self.height):
                    for p in range(window_start_col + 1, window_start_col + self.width + 1):
                        item_id = solution.get((r, p))
                        if item_id is not None:
                            if item_id in items_in_window:
                                return False
                            items_in_window.add(item_id)
        return True

    def __repr__(self):
        return f"{self.name}(width={self.width}, height={self.height})"



########################################### Helper functions ###########################################

def _add_slack_variable(model, K, weight, name):
    s = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{name}")
    penalty_coeff = K * weight / (1 - weight)
    model._penalties.append((s, penalty_coeff))

    return s


# Map an integer deficit/excess in the range [0 bound] onto [0,1]
def _linear_ratio(deficit: int, bound: int) -> float:
    if bound <= 0:
        return 1.0  # degenerate, treat as always satisfied
    return max(0.0, 1.0 - deficit / bound)


def _segments_in_window(window, segments, segment_labels):
    segs = set()
    for item in window:
        for s in segment_labels:
            if item in segments[s]:
                segs.add(s)
    return segs
