# evaluator.py
from traceback import print_tb

import numpy as np

from src.dataset import Dataset


class Evaluator:
    def __init__(self, log_every=100, num_hidden=10):
        self.log_every = log_every
        self.num_hidden = num_hidden # how many hidden items to evaluate recall@N per user

    def evaluate_recall_at_n(self, train_dataset: Dataset, test_dataset: Dataset, model, N=10):
        print("[Evaluator] Evaluating Recall@N...")
        total_recall = 0.0
        user_count = 0

        for test_user_local_idx, test_user_id in enumerate(test_dataset.user_ids):
            user_interactions = test_dataset.matrix[test_user_local_idx].nonzero()[1]
            if len(user_interactions) < 2:
                continue  # Skip users with less than 2 interactions

            recalls = []
            print(f"User {test_user_id} has {len(user_interactions)} interactions")
            for hidden_item_idx in user_interactions:
                observed_item_indices = set(user_interactions)
                observed_item_indices.remove(hidden_item_idx)
                hidden_item_idx_set = {hidden_item_idx}

                observed_item_ids = [test_dataset.item_idx2id[idx] for idx in observed_item_indices]
                hidden_item_id = test_dataset.item_idx2id[hidden_item_idx]

                # Generate recommendations
                recommended_item_ids = model.recommend(test_user_id, observed_item_ids, N, train_dataset, test_dataset)

                # Compute recall
                hit_count = 1 if hidden_item_id in recommended_item_ids else 0
                recall = hit_count / len(hidden_item_idx_set)
                recalls.append(recall)

            if recalls:
                user_recall = np.mean(recalls)
                print(f"User {test_user_id} Recall@{N}: {user_recall:.4f}")
                total_recall += user_recall
                user_count += 1

            if user_count % self.log_every == 0:
                print(f"[Evaluator] Processed {user_count} users. Average Recall@{N}: {total_recall / user_count:.4f}")

        average_recall = total_recall / user_count if user_count > 0 else 0
        print(f"[Evaluator] Average Recall@{N}: {average_recall:.4f}")

        return {'average_recall': average_recall}

