import random

from matplotlib.style.core import available

from src.constraints import *


class ConstraintGenerator:

    def __init__(self):
        pass

    def generate_random_constraints(self, num_constraints, num_recommendations, items=None, segment_ids=None,
                                    segmentation_properties=None, weight_type="mixed", exclude_specific=None):
        """
        Generate n random 1D constraints with random parameters.

        Args:
            num_constraints (int): Number of constraints to generate.
            weight_type (str): "mixed" (default), "soft" or "hard".
            exclude_specific (list): exclude specific constraint types.

        Returns:
            list: A list of constraint objects.
        """
        if items is None:
            items = []
        if segment_ids is None:
            segment_ids = []
        if segmentation_properties is None:
            segmentation_properties = []

        generated_constraints = []
        # For ItemAtPositionConstraint, we record used positions to avoid conflicting items.
        used_item_positions = {}  # position -> item_id
        # For per-segment min/max constraints, we keep a dictionary per segment_id.
        min_max_per_segment = {}  # segment_id -> {'min': value, 'max': value}
        # For global min/max constraints, keyed by segmentation_property.
        global_min_max = {}  # segmentation_property -> {'min': value, 'max': value}
        available_types = available_types = [
                MinItemsPerSegmentConstraint,
                MaxItemsPerSegmentConstraint,
                ItemFromSegmentAtPositionConstraint,
                ItemAtPositionConstraint,
                GlobalMinItemsPerSegmentConstraint,
                GlobalMaxItemsPerSegmentConstraint,
                MinSegmentsConstraint,
                MaxSegmentsConstraint
        ]

        if weight_type not in ["soft", "hard", "mixed"]:
            raise ValueError('weight_type must be one of "soft", "hard", or "mixed".')

        # Define available 1D constraint classes.
        if exclude_specific is not None:
            if not isinstance(exclude_specific, (list, tuple)):
                raise TypeError('exclude_specific must be either a list or a tuple but is of type: ', type(exclude_specific))

            available_types = [c for c in available_types if c not in exclude_specific]


        while len(generated_constraints) < num_constraints:
            constraint_class = random.choice(available_types)

            weight_val = 1.0
            if weight_type == "mixed":
                if random.random() < 0.5:
                    weight_val = 1.0
                else:
                    weight_val = random.uniform(0.0, 1.0)
            elif weight_type == "soft":
                weight_val = random.uniform(0.0, 1.0)
            elif weight_type == "hard":
                pass

            window_size = random.randint(1, num_recommendations)

            if constraint_class == MinItemsPerSegmentConstraint:
                if not segment_ids:
                    continue
                segment_id = random.choice(segment_ids)
                # Choose a minimum between 1 and the window_size.
                min_items = random.randint(1, window_size)
                # If a maximum constraint already exists for this segment, adjust min_items if needed.
                if segment_id in min_max_per_segment and 'max' in min_max_per_segment[segment_id]:
                    if min_items > min_max_per_segment[segment_id]['max']:
                        min_items = min_max_per_segment[segment_id]['max']
                c = MinItemsPerSegmentConstraint(segment_id, min_items, window_size, weight=weight_val)
                min_max_per_segment.setdefault(segment_id, {})['min'] = min_items

            elif constraint_class == MaxItemsPerSegmentConstraint:
                if not segment_ids:
                    continue
                segment_id = random.choice(segment_ids)
                # Choose a maximum between 1 and window_size.
                max_items = random.randint(1, window_size)
                if segment_id in min_max_per_segment and 'min' in min_max_per_segment[segment_id]:
                    if max_items < min_max_per_segment[segment_id]['min']:
                        max_items = min_max_per_segment[segment_id]['min']
                c = MaxItemsPerSegmentConstraint(segment_id, max_items, window_size, weight=weight_val)
                min_max_per_segment.setdefault(segment_id, {})['max'] = max_items

            elif constraint_class == ItemFromSegmentAtPositionConstraint:
                if not segment_ids:
                    continue
                segment_id = random.choice(segment_ids)
                position = random.randint(1, num_recommendations)
                c = ItemFromSegmentAtPositionConstraint(segment_id, position, weight=weight_val)

            elif constraint_class == ItemAtPositionConstraint:
                if not items:
                    continue
                item_id = random.choice(items)
                position = random.randint(1, num_recommendations)
                # Prevent contradiction: do not assign a different item at a position already used.
                if position in used_item_positions and used_item_positions[position] != item_id:
                    continue
                used_item_positions[position] = item_id
                c = ItemAtPositionConstraint(item_id, position, weight=weight_val)

            elif constraint_class == GlobalMinItemsPerSegmentConstraint:
                if not segmentation_properties:
                    continue
                segmentation_property = random.choice(segmentation_properties)
                min_items = random.randint(1, window_size)
                c = GlobalMinItemsPerSegmentConstraint(segmentation_property, min_items, window_size, weight=weight_val)
                global_min_max.setdefault(segmentation_property, {})['min'] = min_items

            elif constraint_class == GlobalMaxItemsPerSegmentConstraint:
                if not segmentation_properties:
                    continue
                segmentation_property = random.choice(segmentation_properties)
                max_items = random.randint(1, window_size)
                if segmentation_property in global_min_max and 'min' in global_min_max[segmentation_property]:
                    if max_items < global_min_max[segmentation_property]['min']:
                        max_items = global_min_max[segmentation_property]['min']
                c = GlobalMaxItemsPerSegmentConstraint(segmentation_property, max_items, window_size, weight=weight_val)
                global_min_max.setdefault(segmentation_property, {})['max'] = max_items

            elif constraint_class == MinSegmentsConstraint:
                if not segmentation_properties:
                    continue
                segmentation_property = random.choice(segmentation_properties)
                # When available, use the number of segment_ids to bound the number of segments.
                max_possible = len(segment_ids) if segment_ids else window_size
                min_segments = random.randint(1, max_possible)
                c = MinSegmentsConstraint(segmentation_property, min_segments, window_size, weight=weight_val)

            elif constraint_class == MaxSegmentsConstraint:
                if not segmentation_properties:
                    continue
                segmentation_property = random.choice(segmentation_properties)
                max_possible = len(segment_ids) if segment_ids else window_size
                max_segments = random.randint(1, max_possible)
                c = MaxSegmentsConstraint(segmentation_property, max_segments, window_size, weight=weight_val)
            else:
                continue

            generated_constraints.append(c)

        return generated_constraints


if __name__ == "__main__":
    # Example usage
    num_recomms = 10
    num_constraints = 5
    items = [f'item-{i}' for i in range(1, num_recomms+1)]
    segment_ids = [f'segment-{i}' for i in range(1, 4)]
    segmentation_properties = ['prop1', 'prop2']
    generator = ConstraintGenerator()
    constraints = generator.generate_random_constraints(num_constraints, num_recomms, items, segment_ids,
                                                        segmentation_properties)
    print(f"Generated {num_constraints} random constraints:")
    for c in constraints:
        print(c)


