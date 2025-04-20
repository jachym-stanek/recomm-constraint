# data_splitter.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from implicit.nearest_neighbours import bm25_weight
import os
import json
import random

from src.dataset import Dataset
from src.settings import Settings


class DataSplitter:
    def __init__(self, settings: Settings):
        self.test_users = None
        self.train_users = None
        self.settings = settings

        self.train_rating_matrix = None
        self.test_rating_matrix = None
        self.rating_matrix = None

    def load_data(self, dataset_name):
        self.settings.set_dataset_in_use(dataset_name)
        print(f"[DataSplitter] Loading data for dataset '{dataset_name}'...")

        dataset_dir = self.settings.dataset.get('transformed_data_dir')

        # Paths to data files
        rating_matrix_file = os.path.join(dataset_dir, f"{dataset_name}_rating_matrix.npz")

        # Load rating matrix
        self.rating_matrix = load_npz(rating_matrix_file)

        print(f"[DataSplitter] Data loading complete.")

    def split_data(self, bmB=None):
        if self.rating_matrix is None:
            raise ValueError("[DataSplitter] Rating matrix is not loaded. Load data before splitting.")

        train_ratio = self.settings.split.get('train_ratio')
        random_state = self.settings.split.get('random_state')

        print(f"[DataSplitter] Splitting users into training and testing sets. Train ratio: {train_ratio}, Random state: {random_state}")

        num_users, num_items = self.rating_matrix.shape
        user_ids = list(range(num_users))
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(user_ids)

        split_index = int(len(user_ids) * train_ratio)
        self.train_users = user_ids[:split_index]
        self.test_users = user_ids[split_index:]
        self.train_users.sort()
        self.test_users.sort()

        print(f"[DataSplitter] Number of training users: {len(self.train_users)}")
        print(f"[DataSplitter] Number of testing users: {len(self.test_users)}")

        data = self.rating_matrix

        # Apply BM25 weighting if specified
        if bmB is not None:
            print(f"[DataSplitter] Applying BM25 weighting to the rating matrix with B = {bmB}")
            data = (bm25_weight(data, B=bmB)).tocsr()

        self._create_train_test_matrices(data)

    def _create_train_test_matrices(self, data):
        print("[DataSplitter] Creating training and testing rating matrices...")

        # Create training and testing rating matrix
        self.train_rating_matrix = data[self.train_users, :]
        self.test_rating_matrix = data[self.test_users, :]

        print(f"[DataSplitter] Training rating matrix shape: {self.train_rating_matrix.shape}")
        print(f"[DataSplitter] Testing rating matrix shape: {self.test_rating_matrix.shape}")

    def get_train_data(self):
        return Dataset(self.train_rating_matrix)

    def get_test_data(self):
        return Dataset(self.test_rating_matrix)


if __name__ == "__main__":
    settings = Settings()  # Load default settings
    data_splitter = DataSplitter(settings)
    data_splitter.load_data('movielens')
    data_splitter.split_data()
    train_data = data_splitter.get_train_data()
    test_data = data_splitter.get_test_data()

    # print samples from train and test data
    print(train_data.matrix[:5])
    print(test_data.matrix[:5])
