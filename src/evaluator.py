# evaluator.py
from typing import List
import numpy as np
from scipy.sparse import csr_matrix

from src.dataset import Dataset
from src.settings import Settings
from src.segmentation import SegmentationExtractor
from src.algorithms.ILP import IlpSolver
from src.constraints import Constraint
from src.models import ALSModel, AnnoyALSModel
from src.algorithms.Preprocessor import ItemPreprocessor


class Evaluator:
    def __init__(self, settings: Settings):
        self.log_every = settings.log_every
        self.num_hidden = settings.recommendations['num_hidden'] # how many hidden items to evaluate recall@N per user

    def evaluate_recall_at_n(self, train_dataset: Dataset, test_dataset: Dataset, model, N=10, take_random_hidden=False, min_relevant_items=3):
        print(f"[Evaluator] Evaluating Recall@{N}, log_every: {self.log_every}, num_hidden: {self.num_hidden}, using model: {model}")
        total_recall = 0.0
        user_count = 0
        total_items_recommended = set()
        skipped_users = 0

        if isinstance(model, ALSModel):
            # precomputed_neighborhoods = model.item_knn.compute_neighborhoods(model.item_factors)
            precomputed_neighborhoods = model.item_knn.compute_neighborhoods_chunked(model.item_factors)
        else:
            precomputed_neighborhoods = None

        # for user in test_dataset.users:
        for user in range(len(test_dataset)):
            user_relevant_items = test_dataset.matrix[user].indices

            # if user has too few relevant items, skip
            if len(user_relevant_items) < min_relevant_items:
                skipped_users += 1
                continue

            recalls = []
            not_yet_hidden = set(user_relevant_items)
            for i in range(self.num_hidden):
                if len(not_yet_hidden) == 0:
                    break
                if take_random_hidden:
                    hidden_item = np.random.choice(list(not_yet_hidden))
                else:
                    hidden_item = user_relevant_items[i]
                not_yet_hidden.remove(hidden_item)
                observed_items = set(user_relevant_items)
                observed_items.remove(hidden_item)

                user_observation = csr_matrix(test_dataset.matrix[user, list(observed_items)])

                # Generate recommendations
                # print(f"obsrvation vector: {user_observation}")
                # print(f"[Evaluator] User {user}, Dims of user obs: {user_observation.shape}, Observations: {observed_items}")
                recomms, scores = model.recommend(test_dataset.matrix, user, user_observation, list(observed_items), N=N, precomputed_similarities=precomputed_neighborhoods, test_user=True)
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
            else: # in case model failed to recommend any items
                skipped_users += 1

            if user_count % self.log_every == 0:
                print(f"[Evaluator] Processed total {user_count+skipped_users}/{len(test_dataset)} users ({skipped_users} skipped). Average Recall@{N}: {total_recall / user_count:.4f} Average catalog coverage: {len(total_items_recommended) / train_dataset.num_items:.4f}")

        average_recall = total_recall / user_count if user_count > 0 else 0
        print(f"[Evaluator] Average Recall@{N}: {average_recall:.4f}")

        return {'average_recall': average_recall, 'catalog_coverage': len(total_items_recommended) / train_dataset.num_items}

    def evaluate_recall_at_n_batch(self, train_dataset: Dataset, test_dataset: Dataset, model, N=10):
        """
        Deprecated
        """
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

    def evaluate_constrained_model(self, train_dataset: Dataset, test_dataset: Dataset, segmentation_extractor: SegmentationExtractor,
                                   constraints: List[Constraint], model, N=10, M=100, take_random_hidden=False, method='ilp'):

        if method not in ['ilp', 'filtering', 'slicing']:
            raise ValueError(f"[Evaluator] Method {method} not supported. Supported methods: ilp, filtering, slicing")

        print(
            f"[Evaluator] Evaluating Recall@{N}, log_every: {self.log_every}, num_hidden: {self.num_hidden}, using model: {model}")
        total_recall= 0.0
        total_recall_constrained = 0.0
        user_count = 0
        total_items_recommended = set()
        total_items_recommended_constrained = set()
        skipped_users = 0

        precomputed_similarities = model.item_knn.compute_similarities(model.item_factors)
        solver = IlpSolver(verbose=False)
        filterer = ItemPreprocessor

        for user in range(len(test_dataset)):
            user_interaction_vector = test_dataset.matrix[user].nonzero()[1]
            user_relevant_items = np.where(test_dataset.matrix[user].toarray() > 0)[1]

            # if user has too few relevant items, skip
            if len(user_relevant_items) < 3:
                skipped_users += 1
                continue

            recalls = []
            recalls_constrained = []
            not_yet_hidden = set(user_relevant_items)
            for i in range(self.num_hidden):
                if len(not_yet_hidden) == 0:
                    break
                if take_random_hidden:
                    hidden_item = np.random.choice(list(not_yet_hidden))
                else:
                    hidden_item = user_relevant_items[i]
                not_yet_hidden.remove(hidden_item)
                observed_items = set(user_interaction_vector)
                observed_items.remove(hidden_item)

                user_observation = csr_matrix(test_dataset.matrix[user, list(observed_items)])

                recomms_no_constraints, recomms_constrained = self._solve_ilp(user, model, solver, filterer, segmentation_extractor, method,
                                                                              user_observation, observed_items, precomputed_similarities, N, M, constraints)

                if recomms_constrained is None:
                    print(f"[Evaluator] No solution found for user {user}, Hidden item {hidden_item}")
                    continue
                recomms_constrained_items = list(recomms_constrained.values())

                # print constrainted recommendations
                # print(f"[Evaluator] Recomms constrained:")
                # for position, item in recomms_constrained.items():
                #     score = canditates[item]
                #     item_segments = [segment for segment in recomm_segments if item in segment]
                #     print(f"Position: {position}, Item: {item}, Score: {score}, Segments: {item_segments}")

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

            if user_count % self.log_every == 0 and user_count > 0:
                print(
                    f"[Evaluator] Processed {user_count + skipped_users}/{len(test_dataset)} users. Average Recall@{N}: {total_recall / user_count:.4f} "
                    f"Average catalog coverage: {len(total_items_recommended) / train_dataset.num_items:.4f} "
                    f"Avg Recall@{N} constrained: {total_recall_constrained / user_count:.4f} "
                    f"Catalog coverage constrained: {len(total_items_recommended_constrained) / train_dataset.num_items:.4f}")

        average_recall = total_recall / user_count if user_count > 0 else 0
        average_recall_constrained = total_recall_constrained / user_count if user_count > 0 else 0
        print(f"[Evaluator] Average Recall@{N}: {average_recall:.4f} Average Recall@{N} constrained: {average_recall_constrained:.4f}")

        return {'average_recall': average_recall,
                'catalog_coverage': len(total_items_recommended) / train_dataset.num_items,
                'average_recall_constrained': average_recall_constrained,
                'catalog_coverage_constrained': len(total_items_recommended_constrained) / train_dataset.num_items}


    def _solve_ilp(self, user, model, solver, filterer, segmentation_extractor, method, user_observation, observed_items, precomputed_neighborhoods, N, M, constraints):
        inner_recomms, scores = model.recommend(user, user_observation, list(observed_items), N=M,
                                                precomputed_similarities=precomputed_neighborhoods, test_user=True)

        # select 10 items with the highest scores
        recomms_no_constraints = inner_recomms[:N]

        candidates = {item: score for item, score in zip(inner_recomms, scores)}
        recomm_segments = segmentation_extractor.get_segments_for_recomms(candidates)

        match method:
            case 'ilp':
                # ILP solver
                recomms_constrained = solver.solve(candidates, recomm_segments, constraints, N)
            case 'filtering':
                # Filtering method
                filtered_items = filterer.preprocess_items(candidates, recomm_segments, constraints, N)
                recomms_constrained = solver.solve(candidates, recomm_segments, constraints, N)
            case 'slicing':
                pass

        return recomms_no_constraints, recomms_constrained

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
