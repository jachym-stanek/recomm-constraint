#

class Constraint:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __repr__(self):
        return "Constraint: " + self.name

    def check_constraint(self, solution, items, segments):
        raise NotImplementedError("Must implement check_constraint method.")


class MinItemsPerSegmentConstraint(Constraint):
    def __init__(self, segment_id, min_items, window_size, name="MinItemsPerSegment", weight=1.0):
        super().__init__(name, weight)
        self.segment_id = segment_id
        self.min_items = min_items
        self.window_size = window_size

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
        return f"{self.name}(segment_id={self.segment_id}, min_items={self.min_items})"


class MaxItemsPerSegmentConstraint(Constraint):
    def __init__(self, segment_id, max_items, window_size, name="MaxItemsPerSegment", weight=1.0):
        super().__init__(name, weight)
        self.segment_id = segment_id
        self.max_items = max_items
        self.window_size = window_size

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
        return f"{self.name}(segment_id={self.segment_id}, max_items={self.max_items})"

class ItemFromSegmentAtPositionConstraint(Constraint):
    def __init__(self, segment_id, position, name="ItemFromSegmentAtPosition", weight=1.0):
        super().__init__(name, weight)
        self.segment_id = segment_id
        self.position = position

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

    def check_constraint(self, solution, items, segments):
        return solution.get(self.position) == self.item_id

    def __repr__(self):
        return f"{self.name}(item_id={self.item_id}, position={self.position})"
