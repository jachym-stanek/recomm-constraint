# segmentation.py

import csv
import json
from tqdm import tqdm


class Segment(set):
    def __init__(self, segment_id, segmentation_property, *args):
        super().__init__(args)
        self.segment_id = segment_id
        self.segmentation_property = segmentation_property

    def __repr__(self):
        return (f"Segmentation[segment id='{self.segment_id}', "
                f"segmentation property='{self.segmentation_property}', "
                f"num elements={len(self)}]")

    @property
    def id(self):
        return self.segment_id

    @property
    def property(self):
        return self.segmentation_property


class SegmentationExtractor:
    def __init__(self, dataset_info_path, items_file_path):
        self.segments = None

        # Load dataset information
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)

        # Get item properties that can be segmented
        self.item_properties = self.dataset_info.get('item_properties', [])

        # Load items data from the CSV file
        self.items = self._load_items(items_file_path)

    def _load_items(self, items_file_path):
        items = []
        # Try to automatically detect the delimiter
        with open(items_file_path, 'r') as f:
            # Peek at the first line to check for delimiter type
            sample_line = f.readline()
            delimiter = ',' if ',' in sample_line else '\t'
            f.seek(0)  # Reset file pointer to the beginning

            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Convert list-like strings to actual lists
                for prop in self.item_properties:
                    if prop in row and row[prop].startswith('[') and row[prop].endswith(']'):
                        row[prop] = eval(row[prop])  # Convert string representation of list to actual list
                items.append(row)
        return items

    def extract_segments(self, segmentation_property):
        if segmentation_property not in self.item_properties:
            raise ValueError(f"Property '{segmentation_property}' is not a valid item property.")

        # Create a dictionary to hold segments
        segments = {}
        for item in tqdm(self.items, desc=f"Extracting segments for '{segmentation_property}'", unit="item"):
            item_id = item['item_id']
            property_value = item.get(segmentation_property)

            if isinstance(property_value, list):
                # Add item to each segment corresponding to the list elements
                for value in property_value:
                    if value not in segments:
                        segments[value] = Segment(segment_id=value, segmentation_property=segmentation_property)
                    segments[value].add(item_id)
            else:
                # Add item to the segment for the single value
                if property_value not in segments:
                    segments[property_value] = Segment(segment_id=property_value,
                                                       segmentation_property=segmentation_property)
                segments[property_value].add(item_id)

        # Return a list of Segmentation objects
        self.segments = list(segments.values())
        return self.segments

    def get_segments_for_recomms(self, recomms):
        recomms_segments = []
        for segment in self.segments:
            segment_items = []
            for item in recomms:
                if item in segment:
                    segment_items.append(item)
            if len(segment_items) > 0:
                recomms_segments.append(Segment(segment.segment_id, segment.segmentation_property, *segment_items))
        return recomms_segments


if __name__ == "__main__":
    # Usage Example
    dataset_info_path = '../data/movielens/dataset_info.json'
    items_file_path = '../data/movielens/items.csv'

    extractor = SegmentationExtractor(dataset_info_path, items_file_path)
    segments = extractor.extract_segments('genres')

    for segment in segments:
        print(segment)
