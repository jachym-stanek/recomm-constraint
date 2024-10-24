# data_splitter.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import os
import json
import random

from src.dataset import Dataset
from src.settings import Settings


class DataSplitter:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.user_id2idx = None
        self.item_id2idx = None
        self.user_idx2id = None
        self.item_idx2id = None

        self.users = None
        self.items = None
        self.interactions = None
        self.train_users = None
        self.test_users = None
        self.train_rating_matrix = None
        self.test_rating_matrix = None
        self.rating_matrix = None

        self.train_mapping = None
        self.test_mapping = None

    def load_data(self, dataset_name):
        print(f"[DataSplitter] Loading data for dataset '{dataset_name}'...")

        dataset_dir = self.settings.dataset.get('transformed_data_dir')

        # Paths to data files
        rating_matrix_file = os.path.join(dataset_dir, f"{dataset_name}_rating_matrix.npz")
        mappings_file = os.path.join(dataset_dir, f"{dataset_name}_id_mappings.json")
        users_file = os.path.join(dataset_dir, 'users.csv')
        items_file = os.path.join(dataset_dir, 'items.csv')
        interactions_file = os.path.join(dataset_dir, 'interactions.csv')

        # Load rating matrix
        self.rating_matrix = load_npz(rating_matrix_file)

        # Load ID mappings
        with open(mappings_file, 'r') as f:
            id_mappings = json.load(f)
        self.user_id2idx = {int(k): int(v) for k, v in id_mappings['user_id_mapping'].items()}
        self.item_id2idx = {int(k): int(v) for k, v in id_mappings['item_id_mapping'].items()}
        self.user_idx2id = {int(v): int(k) for k, v in id_mappings['user_id_mapping'].items()}
        self.item_idx2id = {int(v): int(k) for k, v in id_mappings['item_id_mapping'].items()}

        # Load users, items, and interactions
        self.users = pd.read_csv(users_file)
        self.items = pd.read_csv(items_file)
        self.interactions = pd.read_csv(interactions_file)

        print(f"[DataSplitter] Data loading complete.")

    def split_data(self):
        train_ratio = self.settings.split.get('train_ratio')
        random_state = self.settings.split.get('random_state')

        print(f"[DataSplitter] Splitting users into training and testing sets. Train ratio: {train_ratio}, Random state: {random_state}")

        user_ids = list(self.users['user_id'])
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(user_ids)

        split_index = int(len(user_ids) * train_ratio)
        self.train_users = set(user_ids[:split_index])
        self.test_users = set(user_ids[split_index:])

        print(f"[DataSplitter] Number of training users: {len(self.train_users)}")
        print(f"[DataSplitter] Number of testing users: {len(self.test_users)}")

        self._create_train_test_matrices()

    def _create_train_test_matrices(self):
        print("[DataSplitter] Creating training and testing rating matrices...")

        # Map user IDs to indices
        train_user_indices = [self.user_id2idx[user_id] for user_id in self.train_users]
        test_user_indices = [self.user_id2idx[user_id] for user_id in self.test_users]

        # Create training rating matrix
        train_user_indices.sort()
        train_user_id_idx = {}
        for train_idx, general_idx in enumerate(train_user_indices):
            train_user_id_idx[self.user_idx2id[general_idx]] = train_idx
        self.train_rating_matrix = self.rating_matrix[train_user_indices, :]
        self.train_mapping = train_user_id_idx

        # Create testing rating matrix
        test_user_indices.sort()
        test_user_id_idx = {}
        for test_idx, general_idx in enumerate(test_user_indices):
            test_user_id_idx[self.user_idx2id[general_idx]] = test_idx
        self.test_rating_matrix = self.rating_matrix[test_user_indices, :]
        self.test_mapping = test_user_id_idx

        print(f"[DataSplitter] Training rating matrix shape: {self.train_rating_matrix.shape}")
        print(f"[DataSplitter] Testing rating matrix shape: {self.test_rating_matrix.shape}")

    def get_train_data(self):
        return Dataset(self.train_rating_matrix, self.train_mapping, self.item_id2idx, self.train_users)

    def get_test_data(self):
        return Dataset(self.test_rating_matrix, self.test_mapping, self.item_id2idx, self.test_users)


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
