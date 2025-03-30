import time
import math

from src.algorithms.algorithm import Algorithm
from src.constraints import *


class ItemPreprocessor(Algorithm):
    def __init__(self, name="ItemPreprocessor", description="Item Preprocessor", verbose=True):
        super().__init__(name, description, verbose)

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
                         item_segment_map: Dict[str, list],  # map item_id to all item segment ids
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
                self.preprocess_MaxItemsPerSegmentConstraint(constraint, N, items, segments, removed_items,
                                                             already_recommended_items=previous_recommended_items)
            elif isinstance(constraint, GlobalMaxItemsPerSegmentConstraint):
                constraint.initialize_constraint_from_segments(items_remaining_per_segment)
                for c in constraint.constraints:
                    self.preprocess_MaxItemsPerSegmentConstraint(c, N, items, segments, removed_items,
                                                                 already_recommended_items=previous_recommended_items)
            elif isinstance(constraint, MinItemsPerSegmentConstraint):
                self.compute_bare_minumum_for_MinItemsPerSegmentConstraint(constraint, N, segment_bare_minimums)
            elif isinstance(constraint, GlobalMinItemsPerSegmentConstraint):
                constraint.initialize_constraint_from_segments(items_remaining_per_segment)
                for c in constraint.constraints:
                    self.compute_bare_minumum_for_MinItemsPerSegmentConstraint(c, N, segment_bare_minimums)
            elif isinstance(constraint, MinSegmentsConstraint):
                minimum_number_of_segments = constraint.min_segments
                minimum_items_per_segment = self.compute_limit_items(N, constraint.window_size, constraint.min_segments)
            elif isinstance(constraint, MaxSegmentsConstraint):  # no preprecessing needed
                seqment_quotient = math.ceil(constraint.window_size / constraint.max_segments)
                minimum_number_of_segments = max(minimum_items_per_segment, constraint.max_segments)
                minimum_items_per_segment = max(minimum_items_per_segment, min(seqment_quotient + 1, N))

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
                                                             min_satisfied,
                                                             len(items_remaining_per_segment[constraint.segment_id]),
                                                             remaining_recomm_len)
            elif isinstance(constraint, GlobalMinItemsPerSegmentConstraint):
                for c in constraint.constraints:
                    self.preprocess_MinItemsPerSegmentConstraint(c, N, segment_bare_minimums, min_required_items,
                                                                 min_satisfied,
                                                                 len(items_remaining_per_segment[c.segment_id]),
                                                                 remaining_recomm_len)

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
                if self._item_decision_function(remaining_recomm_len, candidate_items, N, item_segment, min_satisfied,
                                                added_items_per_segment) or \
                        self._segment_decision_function(item_segment, minimum_number_of_segments,
                                                        minimum_items_per_segment,
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
                    self._enough_segments_with_min_items(added_items_per_segment, minimum_number_of_segments,
                                                         minimum_items_per_segment)):
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
            print(
                f"[ILP] Total candidate items after preprocessing: {len(candidate_items)} time: {(time.time() - start) * 1000} milliseconds")
            number_of_added_items_per_segment = {segment_id: len(items) for segment_id, items in
                                                 added_items_per_segment.items()}
            print(f"Number of added items per segment: {number_of_added_items_per_segment}")
        return candidate_items

    def _item_decision_function(self, remaining_recomm_len, candidate_items, N, item_segment, min_satisfied,
                                added_items_per_segment) -> bool:
        # do not take items that have not satisfied min constraints (if we have more partitions on the way
        case1 = (remaining_recomm_len is not None and (
                    len(candidate_items) < N or item_segment in min_satisfied) and not
                 (item_segment in min_satisfied and min_satisfied[item_segment])
                 or (all(min_satisfied.values()) and len(candidate_items) < N))
        # no consecutive recomms -> cam afford to be more greedy
        case2 = (remaining_recomm_len is None and (
                    len(candidate_items) < N or (item_segment in min_satisfied and not min_satisfied[item_segment])))
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

    def preprocess_MaxItemsPerSegmentConstraint(self, constraint: MaxItemsPerSegmentConstraint, N: int,
                                                items: Dict[str, float],
                                                segments: Dict[str, Segment], removed_items: Set[str],
                                                already_recommended_items: List[str] = None):
        segment_id = constraint.segment_id
        max_items = constraint.max_items
        window_size = constraint.window_size

        # Compute the maximum possible number of items from the segment
        max_total_items = self.compute_limit_items(N, window_size, max_items)
        if already_recommended_items is not None and window_size > N:
            index = window_size - N
            count_items_in_previous_recomm = len(
                [item for item in already_recommended_items[-index:] if item in segments[segment_id]])
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
        segment_bare_minimums[segment_id] = max(segment_bare_minimums.get(segment_id, 0),
                                                self.compute_limit_items(N, window_size, min_items))

    def preprocess_MinItemsPerSegmentConstraint(self, constraint: MinItemsPerSegmentConstraint, N: int,
                                                segment_bare_minimums: Dict[str, int],
                                                min_required_items: Dict[str, int], min_satisfied: Dict[str, bool],
                                                num_remaining_segment_items: int,
                                                remaining_recomm_len: int = None):
        segment_id = constraint.segment_id
        sum_bare_minimums = sum([v for v in segment_bare_minimums.values()])
        # bare minimum + remaining items = bm + (N - bm - all other bare minimums) = bm + N - sum all bare minimums
        required_minimum = max(segment_bare_minimums.get(segment_id, 0),
                               segment_bare_minimums.get(segment_id, 0) + N - sum_bare_minimums)
        if remaining_recomm_len is None:
            min_required_items[segment_id] = required_minimum
            min_satisfied[segment_id] = False
            return
        num_items_to_be_needed = self.compute_limit_items(remaining_recomm_len, constraint.window_size,
                                                          constraint.min_items)  # how much will we need if there are remaining partitions
        # do not take too many items if there will be not enough to satisfy future constraints
        if num_items_to_be_needed + required_minimum > num_remaining_segment_items:
            min_required_items[segment_id] = max(segment_bare_minimums.get(segment_id, 0),
                                                 num_remaining_segment_items - num_items_to_be_needed)
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
            number_of_unique_items = self.count_num_unique(candidate_items[r], candidate_items, r, row_above_start,
                                                           row_below_end)

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

    def count_num_unique(self, items: Dict[str, float], candidate_items: List[Dict[str, float]], r: int,
                         row_above_start: int,
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
