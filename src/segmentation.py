# segmentation.py

class Segmentation(set):
    def __init__(self, segment_id, segmentation_property, *args):
        super().__init__(args)
        self.segment_id = segment_id
        self.segmentation_property = segmentation_property

    def __repr__(self):
        return (f"Segmentation(segment id={self.segment_id}, "
                f"segmentation property={self.segmentation_property}, "
                f"elements={list(self)})")

    @property
    def id(self):
        return self.segment_id

    @property
    def property(self):
        return self.segmentation_property
