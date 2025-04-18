# segmentation.py

import csv
import json

from tqdm import tqdm

from src.settings import Settings


class Segment(set):
    def __init__(self, segment_id, segmentation_property, *args):
        super().__init__(args)
        self.segment_id = segment_id
        self.segmentation_property = segmentation_property

    def __repr__(self):
        return (f"Segment[segment id='{self.segment_id}', "
                f"segmentation property='{self.segmentation_property}', "
                f"num elements={len(self)}]")

    @property
    def id(self):
        return self.segment_id

    @property
    def property(self):
        return self.segmentation_property


class SegmentationExtractor:
    def __init__(self, settings: Settings):
        self.segments = None

        # Load dataset information
        with open(settings.dataset["info_file"], 'r') as f:
            self.dataset_info = json.load(f)

        self.item_properties = self.dataset_info.get('item_properties', [])
        self.items = self._load_items(settings.items_file)
        self.item_id2idx, self.item_idx2id = self._load_item_mapping(settings.item_mapping_file)

    def _load_item_mapping(self, item_mapping_file_path):
        with open(item_mapping_file_path, 'r') as f:
            item_mapping = json.load(f)
        return item_mapping['item_id2idx'], item_mapping['item_idx2id']

    def _load_items(self, items_file_path):
        items = []
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
                        try:
                            row[prop] = eval(row[prop])  # Convert string representation of list to actual list
                        except Exception as e:
                            row[prop] = row[prop].replace('[', '').replace(']', '')
                items.append(row)
        return items

    # put item indexes into corresponding segments
    def extract_segments(self, segmentation_property):
        if segmentation_property not in self.item_properties:
            raise ValueError(f"Property '{segmentation_property}' is not a valid item property.")

        # Create a dictionary to hold segments
        segments = {}
        for item in tqdm(self.items, desc=f"[SegmentationExtractor] Extracting segments for '{segmentation_property}'", unit="item"):
            item_id = item['item_id']
            item_idx = self.item_id2idx.get(str(item_id))
            property_value = item.get(segmentation_property) if item.get(segmentation_property) is not None else 'NULL'

            if isinstance(property_value, list):
                # Add item to each segment corresponding to the list elements
                for value in property_value:
                    if value not in segments:
                        segments[value] = Segment(segment_id=value, segmentation_property=segmentation_property)
                    segments[value].add(item_idx)
            else:
                # Add item to the segment for the single value
                if property_value not in segments:
                    segments[property_value] = Segment(segment_id=property_value,
                                                       segmentation_property=segmentation_property)
                segments[property_value].add(item_idx)

        self.segments = segments

    def get_segments(self):
        """
        Get the segments extracted from the dataset.
        :return: dict<seg_id, Segment>
        """
        return self.segments

    def get_segments_for_recomms(self, recomms):
        recomms_segments = []
        for segment in self.segments:
            segment_items = []
            for item in recomms:
                if item in segment:
                    segment_items.append(item)
            if len(segment_items) > 0:
                # create new segment with only the items that are in the recommendations and using item rating matrix index instead of item id
                recomms_segments.append(Segment(segment.segment_id, segment.segmentation_property, *segment_items))
        return recomms_segments


def test_segmentation_extractor():
    # Usage Example
    settings = Settings()

    extractor = SegmentationExtractor(settings)
    segments = extractor.extract_segments('genres')

    for segment in segments:
        print(segment)

    # test get_segments_for_recomms
    print("Testing get_segments_for_recomms")
    recomms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    recomms_segments = extractor.get_segments_for_recomms(recomms)
    for segment in recomms_segments:
        print(segment)
        print(list(segment))
    assert len(recomms_segments) > 0


if __name__ == "__main__":
    test_segmentation_extractor()
