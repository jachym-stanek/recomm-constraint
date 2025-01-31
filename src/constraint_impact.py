import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from src.algorithms.ILP import ILP
from src.data_split import DataSplitter
from src.segmentation import SegmentationExtractor
from src.settings import Settings
from src.constraints.constraint import GlobalMaxItemsPerSegmentConstraint
from src.evaluator import Evaluator
from src.models import ALSModel


def measure_constraint_impact():
    N = 15
    M = 500
    K = 5
    num_hidden = 50
    window_sizes = range(1, N + 1)
    results = []
    dataset = 'movielens'
    segmentation_property = 'year'
    factors = 256
    regularization = 100
    iterations = 3
    settings = Settings()
    settings.set_dataset_in_use(dataset)
    data_splitter = DataSplitter(settings)
    data_splitter.load_data(dataset)
    data_splitter.split_data()
    train_dataset = data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()
    segmentation_extractor = SegmentationExtractor(settings)
    segmentation_extractor.extract_segments(segmentation_property)

    # train ALS model
    model = ALSModel(num_factors=factors, num_iterations=iterations, regularization=regularization, alpha=1.0, use_gpu=False, nearest_neighbors=5)
    model.train(train_dataset)

    precomputed_similarities = model.item_knn.compute_similarities(model.item_factors, K)
    solver = ILP(verbose=False)

    for W in window_sizes:
        constraints = [GlobalMaxItemsPerSegmentConstraint(segmentation_property=segmentation_property, max_items=1, window_size=W, weight=1.0, verbose=False)]
        print(f"Trying Window size: {W}")
        total_recall = 0.0
        total_recall_constrained = 0.0
        user_count = 0
        total_items_recommended = set()
        total_items_recommended_constrained = set()
        skipped_users = 0

        for user in range(len(test_dataset)):
            user_interaction_vector = test_dataset.matrix[user].nonzero()[1]
            user_relevant_items = np.where(test_dataset.matrix[user].toarray() > 0)[1]

            # if user has too few relevant items, skip
            if len(user_relevant_items) < 3:
                skipped_users += 1
                continue

            recalls = []
            recalls_constrained = []
            for i in range(num_hidden):
                if i == len(user_relevant_items):
                    break
                hidden_item = user_relevant_items[i]
                observed_items = set(user_interaction_vector)
                observed_items.remove(hidden_item)

                user_observation = csr_matrix(test_dataset.matrix[user, list(observed_items)])

                # Generate M candidate items
                inner_recomms, scores = model.recommend(user, user_observation, list(observed_items), N=M, K=K,
                                                        precomputed_similarities=precomputed_similarities,
                                                        test_user=True)

                # select 10 items with the highest scores
                recomms_no_constraints = inner_recomms[:N]

                canditates = {item: score for item, score in zip(inner_recomms, scores)}
                recomm_segments = segmentation_extractor.get_segments_for_recomms(inner_recomms)

                recomms_constrained = solver.solve(canditates, recomm_segments, constraints, N)
                if recomms_constrained is None: # this causes a but that makes the unconstrained recalls to not be counted
                    print(f"[Evaluator] No solution found for user {user}, Hidden item {hidden_item}")
                    continue
                recomms_constrained_items = list(recomms_constrained.values())

                # Compute recall
                hit_count = 1 if hidden_item in recomms_no_constraints else 0
                recalls.append(hit_count)
                total_items_recommended.update(recomms_no_constraints)

                hit_count_constrained = 1 if hidden_item in recomms_constrained_items else 0
                recalls_constrained.append(hit_count_constrained)
                total_items_recommended_constrained.update(recomms_constrained_items)

                # print(f"[Evaluator] User {user}, Hidden item {hidden_item}, Recommendations: {recomms_no_constraints}, Constrained: {recomms_constrained}")

            if recalls:
                user_recall = np.mean(recalls)
                total_recall += user_recall
                user_count += 1

                user_recall_constrained = np.mean(recalls_constrained)
                total_recall_constrained += user_recall_constrained

            if user_count % 10 == 0:
                print(
                    f"[Evaluator] Processed {user_count + skipped_users}/{len(test_dataset)} users. Average Recall@{N}: {total_recall / user_count:.4f} "
                    f"Average catalog coverage: {len(total_items_recommended) / train_dataset.num_items:.4f} "
                    f"Avg Recall@{N} constrained: {total_recall_constrained / user_count:.4f} "
                    f"Catalog coverage constrained: {len(total_items_recommended_constrained) / train_dataset.num_items:.4f}")

        average_recall = total_recall / user_count if user_count > 0 else 0
        average_recall_constrained = total_recall_constrained / user_count if user_count > 0 else 0
        print(
            f"[Evaluator] Average Recall@{N}: {average_recall:.4f} Average Recall@{N} constrained: {average_recall_constrained:.4f}")

        metrics = {'average_recall': average_recall,
                'catalog_coverage': len(total_items_recommended) / train_dataset.num_items,
                'average_recall_constrained': average_recall_constrained,
                'catalog_coverage_constrained': len(total_items_recommended_constrained) / train_dataset.num_items}
        print(f"Metrics: {metrics}")
        results.append((W, metrics))

    print("Results:")
    for result in results:
        print(result)

    # save results to file
    with open("constraint_impact_results3.json", "w") as f:
        json.dump(results, f)


def plot_constraint_impact(results_file):
    # load data
    with open(results_file, "r") as f:
        results = json.load(f)

    # plot results on a recall x catalog coverage graph for constrained metrics for each window size
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    recalls = []
    catalog_coverages = []

    for result in results:
        recalls.append(result[1]['average_recall_constrained'])
        catalog_coverages.append(result[1]['catalog_coverage_constrained'])

    ax.plot(recalls, catalog_coverages, marker='o', label='SegmentationMaxDiversity')
    for i, W in enumerate(range(1, len(recalls) + 1)):
        ax.annotate(f'{W}', (recalls[i], catalog_coverages[i]))

    ax.set_xlabel(f'Average Recall@N')
    ax.set_ylabel('Catalog Coverage')
    ax.set_title('Impact of SegmentationMaxDiversity constraint on recommendation quality')
    plt.show()




if __name__ == "__main__":
    # measure_constraint_impact()
    plot_constraint_impact("constraint_impact_results3.json")
