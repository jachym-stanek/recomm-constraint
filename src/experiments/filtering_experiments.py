import random
import matplotlib.pyplot as plt
import seaborn as sns

from src.algorithms.Preprocessor import ItemPreprocessor
from src.segmentation import Segment
from src.constraints import *


def filtered_items_per_number_of_segments():
    M = 200
    N = 5
    segmentation_property = 'test-prop'
    items = {f'item-{i}': random.uniform(0, 1) for i in range(1, M + 1)}
    preprocessor = ItemPreprocessor(verbose=False)
    results = []

    for S in [5, 10, 15, 20, 25, 30, 35, 40]:
        print(f"Running test for S={S}")
        segments = {f'segment{i}-{segmentation_property}': Segment(f'segment{i}', segmentation_property,
                                           *list(items.keys())[i * (M // S):(i + 1) * (M // S)]) for i in range(S)}
        constraints = [
            GlobalMaxItemsPerSegmentConstraint(segmentation_property, 1, 5),
            MinSegmentsConstraint(segmentation_property, 2, 5)
        ]
        filtered_items = preprocessor.preprocess_items(items, segments, constraints, N)
        results.append((S, M - len(filtered_items)))

    print(results)
    # plot results
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    sns.set_context("notebook", font_scale=1.5)

    sns.barplot(x=[result[0] for result in results], y=[result[1] for result in results])
    # plt.title("Number of Filtered Items for Increasing Number of Segments in Candidate Items \n"
    #           "Using N=20, M=200, C={GlobalMaxItems, MinSegments}")
    plt.xlabel("Number of Segments in Candidate Items (|S|)")
    plt.ylabel("Number of Filtered Items")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filtered_items_per_number_of_segments()
