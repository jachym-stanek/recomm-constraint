# models.py

import numpy as np
from scipy.stats import alpha
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from implicit.als import AlternatingLeastSquares

from src.dataset import Dataset
from src.algorithms.ItemKnn import ItemKnn


class BaseModel:
    def __init__(self):
        pass

    def train(self, rating_matrix):
        pass

    def get_similar_users(self, observed_items, k=5):
        # Find users in the training set who have interacted with similar items
        # For simplicity, we'll return top-k users who have the most items in common

        user_similarities = []

        num_users = self.train_rating_matrix.shape[0]
        for user_idx in range(num_users):
            user_items = set(self.train_rating_matrix[user_idx].nonzero()[1])
            intersection = observed_items.intersection(user_items)
            similarity = len(intersection)
            user_id = self.user_idx_to_id[user_idx]
            user_similarities.append((user_id, similarity))

        # Sort users by similarity
        user_similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k similar users
        similar_users = [user_id for user_id, sim in user_similarities[:k] if sim > 0]

        return similar_users


class ALSModel(BaseModel):
    def __init__(self, num_factors=20, num_iterations=10, regularization=0.1, alpha=1.0, use_gpu=False, nearest_neighbors=5):
        super().__init__()
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.alpha = alpha  # Confidence scaling factor
        self.use_gpu = use_gpu

        self.item_knn = ItemKnn(K=nearest_neighbors)

    def train(self, train_dataset: Dataset):
        print("[ALSModel] Training ALS model using implicit library...")

        rating_matrix = train_dataset.matrix.tocsr()

        # Initialize the model
        self.model = AlternatingLeastSquares(
            factors=self.num_factors,
            regularization=self.regularization,
            iterations=self.num_iterations,
            calculate_training_loss=True,
            use_gpu=self.use_gpu,
            alpha=self.alpha
        )

        print("[ALSModel] Using model:", type(self.model))

        # Train the model
        self.model.fit(rating_matrix * self.alpha)

    def recommend(self, user: int, user_observation: csr_matrix, observed_items: list, N: int, K: int, test_user: bool = True,
                  cold_start: bool = False, precomputed_similarities=None):
        if test_user:
            # Test user: Find similar items using item-based k-NN
            if precomputed_similarities is not None:
                recommended = self.item_knn.nearest_neighbors_precomputed(observed_items, precomputed_similarities, N, K)
            else:
                recommended = self.item_knn.nearest_neighbors(observed_items, self.model.item_factors, N, K)
        elif cold_start:
            # Cold-start user: Recalculate user factors based on observed items
            # Generate recommendations using the recalculated user
            recommended = self.model.recommend(
                userid=-1, # Dummy user ID
                user_items=user_observation,
                N=N,
                filter_items=observed_items,
                recalculate_user=True
            )
        else:
            # Known user: Use the precomputed user factors
            recommended = self.model.recommend(
                userid=user,
                user_items=None,  # Not needed since user factors are precomputed
                N=N,
                filter_already_liked_items=True,
                recalculate_user=False
            )

        return recommended

    @property
    def item_factors(self):
        return self.model.item_factors
