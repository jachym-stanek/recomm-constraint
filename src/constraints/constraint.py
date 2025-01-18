from typing import Dict, List, Set

from gurobipy import Model, GRB, quicksum

from src.segmentation import Segment


class Constraint:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight  # weight in [0, 1]

    def add_to_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        raise NotImplementedError("Must implement add_to_model method.")

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        raise NotImplementedError("Must implement check_constraint method.")


class Constraint2D:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight  # weight in [0, 1]

    def add_to_model(self, model, x, items, positions, num_rows, num_cols):
        raise NotImplementedError("Must implement add_to_model method.")

    def check_constraint(self, solution, num_rows, num_cols):
        raise NotImplementedError("Must implement check_constraint method.")


class MinItemsPerSegmentConstraint(Constraint):
    def __init__(self, segment_id, min_items, window_size, name="MinItemsPerSegment", weight=1.0):
        super().__init__(f"{name}_{segment_id}", weight)
        self.segment_id = segment_id
        self.min_items = min_items
        self.window_size = window_size

    def add_to_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_items = segments[self.segment_id]

        # constraint on recomm position
        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]
            if self.weight < 1.0:
                # Soft constraint: Introduce slack variable
                s = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{i}")
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) + s >= self.min_items,
                    name=f"{self.name}_{i}"
                )
                penalty_coeff = K * self.weight / (1 - self.weight)
                model._penalties.append((s, penalty_coeff))
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) >= self.min_items,
                    name=f"{self.name}_{i}"
                )

        # constraint including the already recommended items
        if already_recommended_items: # TODO: refactor this to merge with the previous loop
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items) + 1)
            for i in range(counter_start, counter_end):
                recomm_positions = positions[:self.window_size-i] # positions in the recommendation that are not already recommended

                # count the number of items from the segment in the already recommended items that are in the window
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)

                if self.weight < 1.0:
                    # Soft constraint: Introduce slack variable
                    s = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_already_recommended_{i}")
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions)
                        + num_already_recommended + s >= self.min_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )
                    penalty_coeff = K * self.weight / (1 - self.weight)
                    model._penalties.append((s, penalty_coeff))
                else:
                    # Hard constraint
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions)
                        + num_already_recommended >= self.min_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        N = len(solution)
        segment_items = segments[self.segment_id]
        for i in range(N - self.window_size + 1):
            window = list(solution.values())[i:i + self.window_size]
            count = sum(1 for item_id in window if item_id in segment_items)
            if count < self.min_items:
                return False

        # check the already recommended items
        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items))
            for i in range(counter_start, counter_end):
                recomm_positions = list(solution.values())[:self.window_size-i]
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)
                count = sum(1 for item_id in recomm_positions if item_id in segment_items) + num_already_recommended
                if count < self.min_items:
                    return False
        return True

    def __repr__(self):
        return f"{self.name}(segment_id={self.segment_id}, min_items={self.min_items}, window_size={self.window_size})"


class MaxItemsPerSegmentConstraint(Constraint):
    def __init__(self, segment_id, max_items, window_size, name="MaxItemsPerSegment", weight=1.0):
        super().__init__(f"{name}_{segment_id}", weight)
        self.segment_id = segment_id
        self.max_items = max_items
        self.window_size = window_size

    def add_to_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_items = segments[self.segment_id]

        # constraint on recomm position
        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]
            if self.weight < 1.0:
                # Soft constraint: Introduce slack variable
                s = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{i}")
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) - s <= self.max_items,
                    name=f"{self.name}_{i}"
                )
                penalty_coeff = K * self.weight / (1 - self.weight)
                model._penalties.append((s, penalty_coeff))
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(x[i, row, p] for i in items if i in segment_items for p in window) <= self.max_items,
                    name=f"{self.name}_{i}"
                )

        # constraint including the already recommended items
        if already_recommended_items:
            counter_start = self.window_size - N if (N < self.window_size < len(already_recommended_items) + N) else 1
            counter_end = min(self.window_size, len(already_recommended_items))
            for i in range(counter_start, counter_end):
                recomm_positions = positions[:self.window_size-i]

                # count the number of items from the segment in the already recommended items that are in the window
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)

                if self.weight < 1.0:
                    # Soft constraint: Introduce slack variable
                    s = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_already_recommended_{i}")
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions) + num_already_recommended - s <= self.max_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )
                    penalty_coeff = K * self.weight / (1 - self.weight)
                    model._penalties.append((s, penalty_coeff))
                else:
                    # Hard constraint
                    model.addConstr(
                        quicksum(x[i, row, p] for i in items if i in segment_items for p in recomm_positions) + num_already_recommended <= self.max_items,
                        name=f"{self.name}_already_recommended_{i}"
                    )

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        N = len(solution)
        segment_items = segments[self.segment_id]
        for i in range(N - self.window_size + 1):
            window = list(solution.values())[i:i + self.window_size]
            count = sum(1 for item_id in window if item_id in segment_items)
            if count > self.max_items:
                return False

        # check the already recommended items
        if already_recommended_items:
            counter_start = self.window_size - N if self.window_size > N else 1
            counter_end = min(self.window_size, len(already_recommended_items))
            for i in range(counter_start, counter_end):
                recomm_positions = list(solution.values())[:self.window_size-i]
                num_already_recommended = sum(1 for item_id in already_recommended_items[-i:] if item_id in segment_items)
                count = sum(1 for item_id in recomm_positions if item_id in segment_items) + num_already_recommended
                if count > self.max_items:
                    return False
        return True

    def __repr__(self):
        return f"{self.name}(segment_id={self.segment_id}, max_items={self.max_items}, window_size={self.window_size})"


class ItemFromSegmentAtPositionConstraint(Constraint):
    def __init__(self, segment_id, position, name="ItemFromSegmentAtPosition", weight=1.0):
        super().__init__(f"{name}_{segment_id}", weight)
        self.segment_id = segment_id
        self.position = position

    def add_to_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        segment_items = segments[self.segment_id]
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

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        segment_items = segments[self.segment_id]
        item_id = solution.get(self.position)
        return item_id in segment_items

    def __repr__(self):
        return f"{self.name}(segment_id={self.segment_id}, position={self.position})"


class ItemAtPositionConstraint(Constraint):
    def __init__(self, item_id, position, name="ItemAtPosition", weight=1.0):
        super().__init__(name, weight)
        self.item_id = item_id
        self.position = position

    def add_to_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
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

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        return solution.get(self.position) == self.item_id

    def __repr__(self):
        return f"{self.name}(item_id={self.item_id}, position={self.position})"


"""
Minimum nuber of items from each segment that belongs to segmentation of target property
E.g. Final recommendation should contain at least 2 items from every genre
"""
class MinSegmentsPerSegmentationConstraint(Constraint):
    def __init__(self, segmentation_property, min_items, window_size, weight=1.0, name="MinSegmentsPerSegmentation", verbose=False):
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.min_items = min_items
        self.window_size = window_size
        self.constraints = [] # List of MinItemsPerSegmentConstraint for each segment with min_items and window_size = N
        self.verbose = verbose

    def add_to_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # create MinItemsPerSegmentConstraint for each segment
        for segment_id in segments:
            if segments[segment_id].property == self.segmentation_property:
                if self.verbose:
                    print(f"[MinSegmentsPerSegmentation]Adding constraint for segment {segment_id}, property {self.segmentation_property}")
                constraint = MinItemsPerSegmentConstraint(segment_id, self.min_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_model(model, x, items, segments, row, positions, N, K, already_recommended_items)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        return all(constraint.check_constraint(solution, items, segments, already_recommended_items) for constraint in self.constraints)

    def initialize_constraint_from_segments(self, segments):
        for segment_id in segments:
            if segments[segment_id].property== self.segmentation_property:
                constraint = MinItemsPerSegmentConstraint(segment_id, self.min_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, min_items={self.min_items})"


class MaxSegmentsPerSegmentationConstraint(Constraint):
    def __init__(self, segmentation_property, max_items, window_size, weight=1.0, name="MaxSegmentsPerSegmentation", verbose=False):
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.max_items = max_items
        self.window_size = window_size
        self.constraints = [] # List of MaxItemsPerSegmentConstraint for each segment with max_items and window_size = N
        self.verbose = verbose

    def add_to_model(self, model, x, items, segments, row, positions, N, K, already_recommended_items=None):
        # create MaxItemsPerSegmentConstraint for each segment
        for segment_id in segments:
            if segments[segment_id].property == self.segmentation_property:
                if self.verbose:
                    print(f"[MaxSegmentsPerSegmentation]Adding constraint for segment {segment_id}, property {self.segmentation_property}")
                constraint = MaxItemsPerSegmentConstraint(segment_id, self.max_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_model(model, x, items, segments, row, positions, N, K, already_recommended_items)

    def check_constraint(self, solution, items, segments, already_recommended_items=None):
        return all(constraint.check_constraint(solution, items, segments, already_recommended_items) for constraint in self.constraints)

    def initialize_constraint_from_segments(self, segments):
        for segment_id in segments:
            if segments[segment_id].property == self.segmentation_property:
                constraint = MaxItemsPerSegmentConstraint(segment_id, self.max_items, self.window_size, weight=self.weight)
                self.constraints.append(constraint)

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, max_items={self.max_items})"


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

    def check_constraint(self, solution, num_rows, num_cols):
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
