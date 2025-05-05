def create_item_segment_map_from_segments(segments):
    item_segment_map = {}

    for seg_id, segment in segments.items():
        for item_id in segment:
            if item_id in item_segment_map:
                item_segment_map[item_id].append(seg_id)
            else:
                item_segment_map[item_id] = [seg_id]
    return item_segment_map

def check_solution(test_name, constraints, recommended_items, items, segments, using_soft_constraints=False, verbose=False):
    total_score = 0
    all_constraints_satisfied = True

    if recommended_items:
        for constraint in constraints:
            if not constraint.check_constraint(recommended_items, items, segments):
                all_constraints_satisfied = False
                print(f"Constraint {constraint} is not satisfied.")
        if all_constraints_satisfied or using_soft_constraints:
            print(f"All constraints are satisfied for test {test_name}.")
        for position, item_id in recommended_items.items():
            score = items[item_id]
            total_score += score
            item_segments = [seg for seg in segments if item_id in segments[seg]]
            if verbose:
                print(f"Position {position}: {item_id} (Item segments: {item_segments} Score: {score:.1f})")
        print(f"Total Score: {total_score:.1f}")
    else:
        print(f"No solution found for {test_name}.")

    return total_score, all_constraints_satisfied

def remove_already_recommended_items_from_candidates(already_recommended, items):
    return {item_id: score for item_id, score in items.items() if item_id not in already_recommended}

def total_satisfaction(solution, items, segments, constraints, already_recommended_items=None):
    satisfaction = 0
    weight = 0

    for constraint in constraints:
        s = constraint.satisfaction_ratio(solution, items, segments, already_recommended_items)
        if s is not None:
            satisfaction += s * constraint.weight
            weight += constraint.weight
    if weight > 0:
        satisfaction /= weight
    return satisfaction
