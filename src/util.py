def create_item_segment_map_from_segments(segments):
    item_segment_map = {}

    for seg_id, segment in segments.items():
        for item_id in segment:
            if item_id in item_segment_map:
                item_segment_map[item_id].append(seg_id)
            else:
                item_segment_map[item_id] = [seg_id]
    return item_segment_map
