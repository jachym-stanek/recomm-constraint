# evaluator.py
from traceback import print_tb

import numpy as np
from scipy.sparse import csr_matrix

from src.dataset import Dataset
from src.settings import Settings


class Evaluator:
    def __init__(self, settings: Settings):
        self.log_every = settings.log_every
        self.num_hidden = settings.recommendations['num_hidden'] # how many hidden items to evaluate recall@N per user

    def evaluate_recall_at_n(self, train_dataset: Dataset, test_dataset: Dataset, model, N=10):
        print(f"[Evaluator] Evaluating Recall@{N}, log_every: {self.log_every}, num_hidden: {self.num_hidden}, using model: {model}")
        total_recall = 0.0
        user_count = 0
        total_items_recommended = set()

        for user in range(len(test_dataset)):
            user_interaction_vector = test_dataset.matrix[user].nonzero()[1]
            user_relevant_items = np.where(test_dataset.matrix[user].toarray() > 0)[1]

            recalls = []
            not_yet_hidden = set(user_relevant_items)
            for i in range(self.num_hidden):
                if len(not_yet_hidden) == 0:
                    break
                hidden_item = np.random.choice(list(not_yet_hidden))
                not_yet_hidden.remove(hidden_item)
                observed_items = set(user_interaction_vector)
                observed_items.remove(hidden_item)
                # create crs matrix as users row from test dataset with hidden item removed
                user_observation = csr_matrix(test_dataset.matrix[user, list(observed_items)])

                # Generate recommendations
                # print(f"obsrvation vector: {user_observation}")
                # print(f"[Evaluator] User {user}, Dims of user obs: {user_observation.shape}, Observations: {observed_items}")
                recomms, scores = model.recommend(user, user_observation, list(observed_items), N, cold_start=True)
                # print(f"[Evaluator] User {user}, Hidden item {hidden_item}, Recommendations: {recomms}")
                # print values of recommeded items in the user observation
                # print(f"recommended items: {user_observation[0, recomms].toarray()}")
                # print()

                # Compute recall
                hit_count = 1 if hidden_item in recomms else 0
                recalls.append(hit_count)
                total_items_recommended.update(recomms)

            if recalls:
                user_recall = np.mean(recalls)
                total_recall += user_recall
                user_count += 1

            if user_count % self.log_every == 0:
                print(f"[Evaluator] Processed {user_count}/{len(test_dataset)} users. Average Recall@{N}: {total_recall / user_count:.4f} Average catalog coverage: {len(total_items_recommended) / train_dataset.num_items:.4f}")

        average_recall = total_recall / user_count if user_count > 0 else 0
        print(f"[Evaluator] Average Recall@{N}: {average_recall:.4f}")

        return {'average_recall': average_recall, 'catalog_coverage': len(total_items_recommended) / train_dataset.num_items}

    def evaluate_recall_at_n_batch(self, train_dataset: Dataset, test_dataset: Dataset, model, N=10):
        user_groups = self.separate_test_users_by_interactions(test_dataset)

        for num_relevant_items, users in user_groups.items():
            print(f"[Evaluator] Evaluating Recall@N for users with {num_relevant_items} relevant items...")
            users_interaction_matrix = test_dataset.matrix[users]
            users_relevant_items = np.array([np.where(test_dataset.matrix[user].toarray() > 0)[1] for user in users])
            print(f"[Evaluator] User relevant items shape: {users_relevant_items}")

            for hidden_item_idx in range(num_relevant_items):
                hidden_items = users_relevant_items[hidden_item_idx]
                observed_items = np.delete(users_relevant_items, hidden_item_idx)
                users_observations = csr_matrix(users_interaction_matrix[:, observed_items])

                recomms, scores = model.recommend_batch(users, users_observations, observed_items, N)
                print(f"[Evaluator] Hidden item {hidden_items}, Recommendations: {recomms}")


    # separate test users by number of relevant items (ratings > 0)
    def separate_test_users_by_interactions(self, test_dataset: Dataset):
        print("[Evaluator] Separating test users by number of relevant items...")
        user_groups = {}

        for user in range(len(test_dataset)):
            user_relevant_items = np.where(test_dataset.matrix[user].toarray() > 0)[1]
            num_relevant_items = len(user_relevant_items)
            if num_relevant_items not in user_groups:
                user_groups[num_relevant_items] = []
            user_groups[num_relevant_items].append(user)

        return user_groups
