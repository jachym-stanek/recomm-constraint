import math
import random

from matplotlib.style.core import available

from src.constraints import *


class ConstraintGenerator:

    def __init__(self):
        pass

    def generate_random_constraints(self, num_constraints, num_recommendations, items=None, segments=None,
                                    segmentation_properties=None, weight_type="mixed", exclude_specific=None, min_window_size=1):
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
        if segments is None:
            segments = dict()
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

        if min_window_size > num_recommendations:
            raise ValueError('Parameter min_window_size must be less than or equal to num_recommendations.')

        # Define available 1D constraint classes.
        if exclude_specific is not None:
            if not isinstance(exclude_specific, (list, tuple)):
                raise TypeError('exclude_specific must be either a list or a tuple but is of type: ', type(exclude_specific))

            available_types = [c for c in available_types if c not in exclude_specific]

        max_runs = 1000
        run = 0
        while len(generated_constraints) < num_constraints :
            run += 1
            if run > max_runs:
                break

            constraint_class = random.choice(available_types)

            weight_val = 1.0
            if weight_type == "mixed":
                if random.random() < 0.5:
                    weight_val = 1.0
                else:
                    weight_val = random.uniform(0.1, 1.0)
            elif weight_type == "soft":
                weight_val = random.uniform(0.1, 0.9)
            elif weight_type == "hard":
                pass

            window_size = random.randint(min_window_size, num_recommendations)

            if constraint_class == MinItemsPerSegmentConstraint:
                if not segments:
                    continue
                segment_label = random.choice(list(segments.keys()))
                max_item_per_window = math.ceil(len(segments[segment_label])/(num_recommendations//window_size))
                max_item_per_window = min(max_item_per_window, window_size)
                min_items = random.randint(1, max_item_per_window)
                # If a maximum constraint already exists for this segment, adjust min_items if needed.
                if segment_label in min_max_per_segment and 'max' in min_max_per_segment[segment_label]:
                    if min_items > min_max_per_segment[segment_label]['max']:
                        min_items = min_max_per_segment[segment_label]['max']
                segment_id = segments[segment_label].id
                segment_property = segments[segment_label].property
                c = MinItemsPerSegmentConstraint(segment_id, segment_property, min_items, window_size, weight=weight_val)
                min_max_per_segment.setdefault(segment_id, {})['min'] = min_items

            elif constraint_class == MaxItemsPerSegmentConstraint:
                if not segments:
                    continue
                segment_label = random.choice(list(segments.keys()))
                max_item_per_window = math.ceil(len(segments[segment_label])/(num_recommendations//window_size))
                max_item_per_window = min(max_item_per_window, window_size)
                max_items = random.randint(1, max_item_per_window)
                if segment_label in min_max_per_segment and 'min' in min_max_per_segment[segment_label]:
                    if max_items < min_max_per_segment[segment_label]['min']:
                        max_items = min_max_per_segment[segment_label]['min']
                segment_id = segments[segment_label].id
                segment_property = segments[segment_label].property
                c = MaxItemsPerSegmentConstraint(segment_id, segment_property, max_items, window_size, weight=weight_val)
                min_max_per_segment.setdefault(segment_id, {})['max'] = max_items

            elif constraint_class == ItemFromSegmentAtPositionConstraint:
                if not segments:
                    continue
                segment_label = random.choice(list(segments.keys()))
                position = random.randint(1, num_recommendations)
                segment_id = segments[segment_label].id
                segment_property = segments[segment_label].property
                c = ItemFromSegmentAtPositionConstraint(segment_id, segment_property, position, weight=weight_val)

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
                if len(segments) > 0:
                    num_segments = sum(1 for seg in segments.values() if seg.property == segmentation_property) # make sure constraint is feasible
                    max_per_window = math.ceil(window_size//num_segments) if num_segments > 0 else 1
                else:
                    max_per_window = window_size
                min_items = random.randint(1, max_per_window)
                c = GlobalMinItemsPerSegmentConstraint(segmentation_property, min_items, window_size, weight=weight_val)
                global_min_max.setdefault(segmentation_property, {})['min'] = min_items

            elif constraint_class == GlobalMaxItemsPerSegmentConstraint:
                if not segmentation_properties:
                    continue
                segmentation_property = random.choice(segmentation_properties)
                max_items = random.randint(1, window_size)  # maximum constraint wont have an issue with infeasibility
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
                max_possible = sum([1 for seg in segments.values() if seg.property == segmentation_property]) if len(segments.values()) > 0 else window_size
                max_possible = min(max_possible, window_size)
                min_segments = random.randint(1, max_possible)
                c = MinSegmentsConstraint(segmentation_property, min_segments, window_size, weight=weight_val)

            elif constraint_class == MaxSegmentsConstraint:
                if not segmentation_properties:
                    continue
                segmentation_property = random.choice(segmentation_properties)
                max_segments = random.randint(1, window_size)
                c = MaxSegmentsConstraint(segmentation_property, max_segments, window_size, weight=weight_val)
            else:
                continue

            generated_constraints.append(c)

        print(f"[ConstraintGenerator] Generated {len(generated_constraints)} constraints: {constraints}.")
        return generated_constraints


if __name__ == "__main__":
    # Example usage
    num_recomms = 10
    num_constraints = 5
    items = [f'item-{i}' for i in range(1, num_recomms+1)]
    segmentation_properties = ['prop1', 'prop2']
    segments_list = [
        Segment('segment1', 'prop1', *items[:5]),
        Segment('segment2', 'prop1', *items[5:8]),
        Segment('segment1', 'prop2', *items[5:10])
    ]
    segments = {segment.label: segment for segment in segments_list}
    generator = ConstraintGenerator()
    constraints = generator.generate_random_constraints(num_constraints, num_recomms, items, segments,
                                                        segmentation_properties)
    print(f"Generated {num_constraints} random constraints:")
    for c in constraints:
        print(c)

    # Test the generated constraints with min window size and without position constraints
    constraints = generator.generate_random_constraints(num_constraints, num_recomms, items, segments,
                                                        segmentation_properties, min_window_size=5, weight_type="soft",
                                                        exclude_specific=[ItemAtPositionConstraint, ItemFromSegmentAtPositionConstraint]
                                                        )
    print(f"Generated {num_constraints} random constraints with min window size and without position constraints:")
    for c in constraints:
        print(c)


