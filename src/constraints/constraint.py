from gurobipy import Model, GRB, quicksum


class Constraint:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight  # weight in [0, 1]

    def add_to_model(self, model, x, items, segments, positions, N, K):
        raise NotImplementedError("Must implement add_to_model method.")

    def check_constraint(self, solution, items, segments):
        raise NotImplementedError("Must implement check_constraint method.")


class MinItemsPerSegmentConstraint(Constraint):
    def __init__(self, segment_id, min_items, window_size, name="MinItemsPerSegment", weight=1.0):
        super().__init__(f"{name}_{segment_id}", weight)
        self.segment_id = segment_id
        self.min_items = min_items
        self.window_size = window_size

    def add_to_model(self, model, x, items, segments, positions, N, K):
        segment_items = segments[self.segment_id]
        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]
            if self.weight < 1.0:
                # Soft constraint: Introduce slack variable
                s = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{i}")
                model.addConstr(
                    quicksum(x[i, p] for i in segment_items for p in window) + s >= self.min_items,
                    name=f"{self.name}_{i}"
                )
                penalty_coeff = K * self.weight / (1 - self.weight)
                model._penalties.append((s, penalty_coeff))
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(x[i, p] for i in segment_items for p in window) >= self.min_items,
                    name=f"{self.name}_{i}"
                )

    def check_constraint(self, solution, items, segments):
        N = len(solution)
        segment_items = segments[self.segment_id]
        for i in range(N - self.window_size + 1):
            window = list(solution.values())[i:i + self.window_size]
            count = sum(1 for item_id in window if item_id in segment_items)
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

    def add_to_model(self, model, x, items, segments, positions, N, K):
        segment_items = segments[self.segment_id]
        for i in range(N - self.window_size + 1):
            window = positions[i:i + self.window_size]
            if self.weight < 1.0:
                # Soft constraint: Introduce slack variable
                s = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{i}")
                model.addConstr(
                    quicksum(x[i, p] for i in segment_items for p in window) - s <= self.max_items,
                    name=f"{self.name}_{i}"
                )
                penalty_coeff = K * self.weight / (1 - self.weight)
                model._penalties.append((s, penalty_coeff))
            else:
                # Hard constraint
                model.addConstr(
                    quicksum(x[i, p] for i in segment_items for p in window) <= self.max_items,
                    name=f"{self.name}_{i}"
                )

    def check_constraint(self, solution, items, segments):
        N = len(solution)
        segment_items = segments[self.segment_id]
        for i in range(N - self.window_size + 1):
            window = list(solution.values())[i:i + self.window_size]
            count = sum(1 for item_id in window if item_id in segment_items)
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

    def add_to_model(self, model, x, items, segments, positions, N, K):
        segment_items = segments[self.segment_id]
        if self.weight < 1.0:
            # Soft constraint: Introduce slack variable
            s = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{self.position}")
            model.addConstr(
                quicksum(x[i, self.position] for i in segment_items) + s >= 1,
                name=f"{self.name}_{self.position}"
            )
            penalty_coeff = K * self.weight / (1 - self.weight)
            model._penalties.append((s, penalty_coeff))
        else:
            # Hard constraint
            model.addConstr(
                quicksum(x[i, self.position] for i in segment_items) >= 1,
                name=f"{self.name}_{self.position}"
            )

    def check_constraint(self, solution, items, segments):
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

    def add_to_model(self, model, x, items, segments, positions, N, K):
        if self.weight < 1.0:
            # Soft constraint: Introduce slack variable
            s = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"s_{self.name}_{self.position}")
            model.addConstr(
                x[self.item_id, self.position] + s >= 1,
                name=f"{self.name}_{self.position}"
            )
            penalty_coeff = K * self.weight / (1 - self.weight)
            model._penalties.append((s, penalty_coeff))
        else:
            # Hard constraint
            model.addConstr(
                x[self.item_id, self.position] == 1,
                name=f"{self.name}_{self.position}"
            )

    def check_constraint(self, solution, items, segments):
        return solution.get(self.position) == self.item_id

    def __repr__(self):
        return f"{self.name}(item_id={self.item_id}, position={self.position})"


"""
Minimum nuber of items from each segment that belongs to segmentation of target property
E.g. Final recommendation should contain at least 2 items from every genre
"""
class SegmentationMinDiversity(Constraint):
    def __init__(self, segmentation_property, min_items, weight=1.0, name="SegmentationMinDiversity"):
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.min_items = min_items
        self.constraints = [] # List of MinItemsPerSegmentConstraint for each segment with min_items and window_size = N

    def add_to_model(self, model, x, items, segments, positions, N, K):
        # create MinItemsPerSegmentConstraint for each segment
        for segment_id in segments:
            if segments[segment_id].property == self.segmentation_property:
                constraint = MinItemsPerSegmentConstraint(segment_id, self.min_items, N, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_model(model, x, items, segments, positions, N, K)

    def check_constraint(self, solution, items, segments):
        return all(constraint.check_constraint(solution, items, segments) for constraint in self.constraints)

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, min_items={self.min_items})"


class SegmentationMaxDiversity(Constraint):
    def __init__(self, segmentation_property, max_items, weight=1.0, name="SegmentationMaxDiversity"):
        super().__init__(name, weight)
        self.segmentation_property = segmentation_property
        self.max_items = max_items
        self.constraints = [] # List of MaxItemsPerSegmentConstraint for each segment with max_items and window_size = N

    def add_to_model(self, model, x, items, segments, positions, N, K):
        # create MaxItemsPerSegmentConstraint for each segment
        for segment_id in segments:
            if segments[segment_id].property == self.segmentation_property:
                constraint = MaxItemsPerSegmentConstraint(segment_id, self.max_items, N, weight=self.weight)
                self.constraints.append(constraint)
                constraint.add_to_model(model, x, items, segments, positions, N, K)

    def check_constraint(self, solution, items, segments):
        return all(constraint.check_constraint(solution, items, segments) for constraint in self.constraints)

    def __repr__(self):
        return f"{self.name}(segmentation_property={self.segmentation_property}, max_items={self.max_items})"

