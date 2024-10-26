# evaluator.py
from traceback import print_tb

import numpy as np
from scipy.sparse import csr_matrix

from src.dataset import Dataset


class Evaluator:
    def __init__(self, log_every=100, num_hidden=10):
        self.log_every = log_every
        self.num_hidden = num_hidden # how many hidden items to evaluate recall@N per user

    def evaluate_recall_at_n(self, train_dataset: Dataset, test_dataset: Dataset, model, N=10):
        print("[Evaluator] Evaluating Recall@N...")
        total_recall = 0.0
        user_count = 0
        total_items_recommended = set()

        for test_user_local_idx, test_user_id in enumerate(test_dataset.user_ids):
            user_interaction_vector = test_dataset.matrix[test_user_local_idx].nonzero()[1]
            if len(user_interaction_vector) < 2:
                continue  # Skip users with less than 2 interactions

            recalls = []
            not_yet_hidden = set(user_interaction_vector)
            for i in range(self.num_hidden):
                if len(not_yet_hidden) == 0:
                    break
                hidden_item_idx = np.random.choice(list(not_yet_hidden))
                not_yet_hidden.remove(hidden_item_idx)
                observed_items = set(user_interaction_vector)
                observed_items.remove(hidden_item_idx)
                # create crs matrix as users row from test dataset with hidden item removed
                user_observation = csr_matrix(test_dataset.matrix[test_user_local_idx])
                user_observation[0, hidden_item_idx] = 0

                # Generate recommendations
                recommended_item_ids = model.recommend(test_user_id, user_observation, observed_items, N, train_dataset, test_dataset)

                # Compute recall
                hit_count = 1 if hidden_item_idx in recommended_item_ids else 0
                recalls.append(hit_count)
                total_items_recommended.update(recommended_item_ids)

            if recalls:
                user_recall = np.mean(recalls)
                total_recall += user_recall
                user_count += 1

            if user_count % self.log_every == 0:
                print(f"[Evaluator] Processed {user_count}/{len(test_dataset)} users. Average Recall@{N}: {total_recall / user_count:.4f} Average catalog coverage: {len(total_items_recommended) / train_dataset.num_items:.4f}")

        average_recall = total_recall / user_count if user_count > 0 else 0
        print(f"[Evaluator] Average Recall@{N}: {average_recall:.4f}")

        return {'average_recall': average_recall, 'catalog_coverage': len(total_items_recommended) / train_dataset.num_items}
