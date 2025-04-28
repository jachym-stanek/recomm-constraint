# models.py

import numpy as np
from scipy.stats import alpha
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import AnnoyAlternatingLeastSquares
from src.dataset import Dataset
from src.algorithms.ItemKnn import ItemKnn


"""
Models serve for generating recommendations for a user based on some input data.
"""
class BaseModel:
    def __init__(self):
        pass

    def train(self, rating_matrix):
        raise NotImplementedError("Train method not implemented.")

    def recommend(self, R: csr_matrix, user: int, user_observation: csr_matrix, observed_items: list, N: int, test_user: bool = True,
                    cold_start: bool = False, precomputed_similarities=None):
            raise NotImplementedError("Recommend method not implemented.")

class ALSModel(BaseModel):
    def __init__(self, num_factors=20, num_iterations=10, regularization=0.1, alpha=1.0, use_gpu=False, nearest_neighbors=10):
        super().__init__()
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.alpha = alpha  # Confidence scaling factor
        self.use_gpu = use_gpu

        self.item_knn = ItemKnn(K=nearest_neighbors)

        self.model = None

    def train(self, train_dataset: Dataset):
        print(f"[ALSModel] Training ALS model with params: {self.__dict__}...")

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
        self.model.fit(rating_matrix)

    def recommend(self, R: csr_matrix, user: int, user_observation: csr_matrix, observed_items: list, N: int, test_user: bool = True,
                  cold_start: bool = False, precomputed_similarities=None):
        if test_user:
            # Test user: Find similar items using item-based k-NN
            if precomputed_similarities is not None:
                recommended = self.item_knn.nearest_neighbors_precomputed(R, user, observed_items, precomputed_similarities, N)
            else:
                recommended = self.item_knn.nearest_neighbors(observed_items, self.model.item_factors, N)
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
        return np.asarray(self.model.item_factors, dtype=np.float32)


class AnnoyALSModel(BaseModel):
    def __init__(self, num_factors=20, num_iterations=10, regularization=0.1, alpha=1.0, use_gpu=False,  num_trees=50):
        super().__init__()
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.alpha = alpha  # Confidence scaling factor
        self.use_gpu = use_gpu
        self.num_trees = num_trees

        self.model = None

    def train(self, train_dataset: Dataset):
        print("[ALSModel] Training Annoy ALS model...")

        rating_matrix = train_dataset.matrix.tocsr()

        # Initialize the model
        self.model = AnnoyAlternatingLeastSquares(
            factors=self.num_factors,
            regularization=self.regularization,
            iterations=self.num_iterations,
            calculate_training_loss=True,
            use_gpu=self.use_gpu,
            alpha=self.alpha,
            n_trees=self.num_trees
        )

        print("[ALSModel] Using model:", type(self.model))

        # Train the model
        self.model.fit(rating_matrix)

    def recommend(self, R: csr_matrix, user: int, user_observation: csr_matrix, observed_items: list, N: int, test_user: bool = True,
                  cold_start: bool = False, precomputed_similarities=None):
        if test_user:
            recommended = self.model.recommend(
                userid=user,
                user_items=user_observation,
                N=N,
                filter_already_liked_items=True,
                recalculate_user=False
            )
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


class BestsellerModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.bestsellers = None

    def train(self, train_dataset: Dataset, max_interaction_age: int = 0):
        self.bestsellers = np.array(train_dataset.matrix.sum(axis=0)).flatten() # Sum of interactions per item
        self.bestsellers = np.argsort(self.bestsellers)[::-1] # Sort in descending order

    def recommend(self, user: int, observed_items: list, N: int, user_observation: csr_matrix = None,  K: int = None,
                  test_user: bool = True, cold_start: bool = False, precomputed_similarities=None):
        # Recommend the top N bestsellers that are not already observed
        bestseller_indices = []
        for item in self.bestsellers:
            if item not in observed_items:
                bestseller_indices.append(item)
            if len(bestseller_indices) == N:
                break
