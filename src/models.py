# models.py

import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from implicit.als import AlternatingLeastSquares

from src.dataset import Dataset


class BaseModel:
    def __init__(self):
        self.user_factors = None
        self.item_factors = None
        self.train_rating_matrix = None

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
    def __init__(self, num_factors=20, num_iterations=10, regularization=0.1, alpha=1.0):
        super().__init__()
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.alpha = alpha  # Confidence scaling factor

    def train(self, train_dataset: Dataset):
        print("[ALSModel] Training ALS model using implicit library...")

        rating_matrix = train_dataset.matrix.tocsr()

        # Initialize the model
        self.model = AlternatingLeastSquares(
            factors=self.num_factors,
            regularization=self.regularization,
            iterations=self.num_iterations,
            calculate_training_loss=True,
            use_gpu=False  # Set to True if you have a compatible GPU and CuPy installed
        )

        # Train the model
        self.model.fit(rating_matrix * self.alpha)

    def recommend(self, user: int, user_observation: csr_matrix, observed_items: list, N: int, cold_start: bool = True):
        if cold_start:
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

    # only implemented for cold start
    def recommend_batch(self, users: list, user_observation: csr_matrix, observed_items: np.array, N: int):
        return self.model.recommend(
            userid=users,  # passed to keep the number of returned items consistent
            user_items=user_observation,
            N=N,
            filter_items=observed_items,
            recalculate_user=True
        )


class SGDModel(BaseModel):
    def __init__(self, num_factors=20, num_epochs=10, learning_rate=0.01, regularization=0.1):
        super().__init__()
        self.num_factors = num_factors
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

    def train(self, rating_matrix, user_id_mapping, item_id_mapping):
        print("[SGDModel] Training SGD model...")
        self.train_rating_matrix = rating_matrix
        self.user_id_mapping = user_id_mapping
        self.item_id_mapping = item_id_mapping
        self.user_idx_to_id = {v: k for k, v in user_id_mapping.items()}
        self.item_idx_to_id = {v: k for k, v in item_id_mapping.items()}

        num_users, num_items = rating_matrix.shape

        # Initialize user and item factors randomly
        self.user_factors = np.random.normal(scale=0.1, size=(num_users, self.num_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(num_items, self.num_factors))

        # Get non-zero entries
        user_indices, item_indices = rating_matrix.nonzero()
        ratings = rating_matrix[user_indices, item_indices].A1

        for epoch in range(self.num_epochs):
            print(f"[SGDModel] Epoch {epoch+1}/{self.num_epochs}...")
            # Shuffle the data
            indices = np.arange(len(ratings))
            np.random.shuffle(indices)
            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r_ui = ratings[idx]
                # Compute prediction and error
                pred = self.user_factors[u, :].dot(self.item_factors[i, :].T)
                error = r_ui - pred
                # Update factors
                self.user_factors[u, :] += self.learning_rate * (error * self.item_factors[i, :] - self.regularization * self.user_factors[u, :])
                self.item_factors[i, :] += self.learning_rate * (error * self.user_factors[u, :] - self.regularization * self.item_factors[i, :])

class NMFModel(BaseModel):
    def __init__(self, num_factors=20, num_iterations=50):
        super().__init__()
        self.num_factors = num_factors
        self.num_iterations = num_iterations

    def train(self, rating_matrix, user_id_mapping, item_id_mapping):
        print("[NMFModel] Training NMF model...")

        self.train_rating_matrix = rating_matrix
        self.user_id_mapping = user_id_mapping
        self.item_id_mapping = item_id_mapping
        self.user_idx_to_id = {v: k for k, v in user_id_mapping.items()}
        self.item_idx_to_id = {v: k for k, v in item_id_mapping.items()}

        model = NMF(n_components=self.num_factors, init='random', random_state=0, max_iter=self.num_iterations)
        W = model.fit_transform(rating_matrix)
        H = model.components_
        self.user_factors = W
        self.item_factors = H.T
