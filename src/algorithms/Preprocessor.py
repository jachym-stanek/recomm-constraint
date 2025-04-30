import time
import math
from typing import Tuple

from src.algorithms.algorithm import Algorithm
from src.constraints import *


class _SegmentClass(object):
    def __init__(self, segments: list, N: int):
        self._segments = segments
        self._items = list()
        self._min_items = 0
        self._max_items = N

    @property
    def segments(self):
        return self._segments

    @property
    def items(self):
        return self._items

    @property
    def min_items(self):
        return self._min_items

    @min_items.setter
    def min_items(self, value):
        self._min_items = max(self._min_items, value)

    @property
    def max_items(self):
        return self._max_items

    @max_items.setter
    def max_items(self, value):
        self._max_items = min(self._max_items, value)

    def contains_segment(self, segment):
        return segment in self._segments

    def append(self, item):
        self._items.append(item)

    def __contains__(self, item):
        return item in self._items

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"SegmentClass{self._segments}"


class ItemPreprocessor(Algorithm):
    def __init__(self, name="ItemPreprocessor", description="Item Preprocessor", verbose=True):
        super().__init__(name, description, verbose)

    def preprocess_items(self, items: Dict[str, float],
                         segments: Dict[str, Segment],
                         constraints: List[Constraint],
                         N: int) -> Dict[str, float]:

        start_time = time.time()

        segment_classes, item2segments = self._preprocess_segments(segments, N)

        items_added_per_segment_class = {segment_class: 0 for segment_class in segment_classes.values()}
        minimum_satisfied = {segment_class: True for segment_class in segment_classes.values()}
        filtered_items = {}

        for constraint in constraints:
            if isinstance(constraint, MaxItemsPerSegmentConstraint):
                self._preprocess_MaxItemsPerSegmentConstraint(constraint, N, segment_classes, minimum_satisfied)
            elif isinstance(constraint, GlobalMaxItemsPerSegmentConstraint):
                for c in constraint.sub_constraints_from_segments(segments):
                    self._preprocess_MaxItemsPerSegmentConstraint(c, N, segment_classes, minimum_satisfied)
            elif isinstance(constraint, MinItemsPerSegmentConstraint):
                self._preprocess_MinItemsPerSegmentConstraint(constraint, N, segment_classes, minimum_satisfied)
            elif isinstance(constraint, GlobalMinItemsPerSegmentConstraint):
                for c in constraint.sub_constraints_from_segments(segments):
                    self._preprocess_MinItemsPerSegmentConstraint(c, N, segment_classes, minimum_satisfied)
            elif isinstance(constraint, ItemAtPositionConstraint):
                item_id = constraint.item_id
                if item_id in items:
                    filtered_items[item_id] = items[item_id]
            elif isinstance(constraint, ItemFromSegmentAtPositionConstraint):
                self._preprocess_ItemFromSegmentAtPositionConstraint(constraint, N, segment_classes, minimum_satisfied)
            elif isinstance(constraint, MinSegmentsConstraint) or isinstance(constraint, MaxSegmentsConstraint):
                self._preprocess_segment_constraint(constraint, N, segment_classes, minimum_satisfied, segments)

        sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)

        idx = 0
        while True:
            item, score = sorted_items[idx]
            item_segments = item2segments.get(item)
            item_segment_class = self._get_item_class(item_segments, segment_classes)

            # decide if to add item
            if self._item_decision_function(item_segment_class, items_added_per_segment_class, filtered_items, N):
                filtered_items[item] = score

                if item_segment_class is not None:
                    if item_segment_class not in items_added_per_segment_class:
                        items_added_per_segment_class[item_segment_class] = 0
                    items_added_per_segment_class[item_segment_class] += 1

                    if (items_added_per_segment_class[item_segment_class] >= item_segment_class.min_items or
                            items_added_per_segment_class[item_segment_class] == len(item_segment_class)):
                        minimum_satisfied[item_segment_class] = True

            # stopping criteria
            # all min items constraints are satisfied, we have enough items and enough segments
            if len(filtered_items) >= N and all(minimum_satisfied.values()):
                break

            idx += 1
            if idx >= len(sorted_items):
                break

        # fallback if preprocessing still filtered out too many items (can happen when a lot of high scoring items are
        # removed and then only small scoring items are left from the segment with min items constraint)
        if len(filtered_items) < N:
            for item, score in sorted_items:
                if item not in filtered_items:
                    filtered_items[item] = score
                if len(filtered_items) >= N:
                    break

        if self.verbose:
            print(
                f"[Preprocessor] Total candidate items after preprocessing: {len(filtered_items)} time: {(time.time() - start_time) * 1000} milliseconds")
            print(f"Number of added items per segment class: {items_added_per_segment_class}")
        return filtered_items

    @staticmethod
    def _preprocess_segments(segments: Dict[str, Segment], N: int) -> Tuple[Dict[Tuple, _SegmentClass], Dict[str, list]]:
        """
        Extract segment classes (= all items that belong to the same segments will form a segment class) and item2segments map
        """
        segment_classes = {}
        item2segments = {}

        for seg_label, segment in segments.items():
            for item_id in segment:
                if item_id in item2segments:
                    item2segments[item_id].append(seg_label)
                else:
                    item2segments[item_id] = [seg_label]

        # create segment classes based on which segments the items belong to
        for item_id, seg_ids in item2segments.items():
            ordered_seg_ids = tuple(sorted(seg_ids))
            if ordered_seg_ids not in segment_classes:
                segment_classes[ordered_seg_ids] = _SegmentClass(list(ordered_seg_ids), N)
            segment_classes[ordered_seg_ids].append(item_id)

        return segment_classes, item2segments

    @staticmethod
    def _get_item_class(item_segments, segment_classes):
        """
        Get the segment class for the item
        """
        if item_segments is None:
            return None
        ordered_seg_ids = tuple(sorted(item_segments))
        if ordered_seg_ids in segment_classes:
            return segment_classes[ordered_seg_ids]
        return None

    @staticmethod
    def _compute_limit_items(N: int, W: int, item_limit_per_window: int):
        full_windows = N // W
        remaining_positions = N % W
        max_items_limit = (item_limit_per_window * full_windows) + min(remaining_positions, item_limit_per_window)
        return max_items_limit

    def _preprocess_MaxItemsPerSegmentConstraint(self, constraint: MaxItemsPerSegmentConstraint, N: int,
                                                 segment_classes: Dict[Tuple, _SegmentClass], minimum_satisfied: Dict[_SegmentClass, bool]):
        segment_label = constraint.label
        max_items = constraint.max_items
        window_size = constraint.window_size

        # Compute the maximum possible number of items from the segment
        max_total_items = self._compute_limit_items(N, window_size, max_items)

        # Set the maximum to all Segment classes containing the segment
        for seg_class in segment_classes.values():
            if seg_class.contains_segment(segment_label):
                seg_class.min_items = max_total_items
                seg_class.max_items = max_total_items
                minimum_satisfied[seg_class] = False

    def _preprocess_MinItemsPerSegmentConstraint(self, constraint: MinItemsPerSegmentConstraint, N: int,
                                                 segment_classes: Dict[Tuple, _SegmentClass], minimum_satisfied: Dict[_SegmentClass, bool]):
        segment_label = constraint.label
        min_items = constraint.min_items
        window_size = constraint.window_size

        # Compute the minimum possible number of items from the segment
        min_total_items = self._compute_limit_items(N, window_size, min_items)

        # Set the minimum to all Segment classes containing the segment
        for seg_class in segment_classes.values():
            if seg_class.contains_segment(segment_label):
                seg_class.min_items = min_total_items
                minimum_satisfied[seg_class] = False

    def _preprocess_ItemFromSegmentAtPositionConstraint(self, constraint: ItemFromSegmentAtPositionConstraint, N: int,
                                   segment_classes: Dict[Tuple, _SegmentClass], minimum_satisfied: Dict[_SegmentClass, bool]):
        segment_label = constraint.label

        # all classes containing the segment have to have minimum at least 1 item
        for seg_class in segment_classes.values():
            if seg_class.contains_segment(segment_label):
                seg_class.min_items = 1
                minimum_satisfied[seg_class] = False

    def _preprocess_segment_constraint(self, constraint: Constraint, N: int, segment_classes: Dict[Tuple, _SegmentClass],
                                       minimum_satisfied: Dict[_SegmentClass, bool], segments: Dict[str, Segment]):
        segmentation_property = constraint.segmentation_property
        # if any segment constraint is set, then set all segment classes minimums to N
        for seg_class in segment_classes.values():
            # if the segment class contains any segment with the segmentation property, set the minimum items to N
            if any(seg_class.contains_segment(seg.label) for seg in segments.values() if seg.property == segmentation_property):
                seg_class.min_items = N
                minimum_satisfied[seg_class] = False

    def _item_decision_function(self, item_segment_class: _SegmentClass,
                                items_added_per_segment_class: Dict[_SegmentClass, int], filtered_items, N) -> bool:

        # if item does not belong to any segment class, add it
        if item_segment_class is None and len(filtered_items) < N:
            return True
        elif item_segment_class is None:
            return False
        # item is added if the segment class has not satisfied the minimum items constraint
        if items_added_per_segment_class[item_segment_class] < item_segment_class.min_items:
            return True
        # if we are not overselecting the segment class, add it
        if items_added_per_segment_class[item_segment_class] < item_segment_class.max_items:
            return True

        return False
