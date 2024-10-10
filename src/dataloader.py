# data_loader.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import os
import json
import random

class DataLoader:
    def __init__(self, rating_matrix_dir):
        self.rating_matrix_dir = rating_matrix_dir
        self.users = None
        self.items = None
        self.interactions = None
        self.user_id_mapping = {}
        self.item_id_mapping = {}
        self.train_users = None
        self.test_users = None
        self.train_rating_matrix = None
        self.test_interactions = None  # For candidate generation and evaluation

    def load_data(self, dataset_name, train_ratio=0.8, random_state=None):
        print(f"[DataLoader] Loading data for dataset '{dataset_name}'...")

        rating_matrix_file = os.path.join(self.rating_matrix_dir, f"{dataset_name}_rating_matrix.npz")
        mappings_file = os.path.join(self.rating_matrix_dir, f"{dataset_name}_id_mappings.json")
        users_file = os.path.join(self.rating_matrix_dir, f"{dataset_name}_users.csv")
        items_file = os.path.join(self.rating_matrix_dir, f"{dataset_name}_items.csv")
        interactions_file = os.path.join(self.rating_matrix_dir, f"{dataset_name}_interactions.csv")

        # Load rating matrix
        rating_matrix = load_npz(rating_matrix_file)

        # Load ID mappings
        with open(mappings_file, 'r') as f:
            id_mappings = json.load(f)
        self.user_id_mapping = id_mappings['user_id_mapping']
        self.item_id_mapping = id_mappings['item_id_mapping']

        # Load users, items, and interactions
        self.users = pd.read_csv(users_file)
        self.items = pd.read_csv(items_file)
        self.interactions = pd.read_csv(interactions_file)

        # Split users into training and testing sets
        self._split_users(train_ratio, random_state)

        # Create training rating matrix
        self.train_rating_matrix = self._create_train_rating_matrix(rating_matrix)

        # Prepare testing interactions for candidate generation and evaluation
        self.test_interactions = self._get_test_interactions()

        print(f"[DataLoader] Data loading and splitting complete.")

        return self.train_rating_matrix, self.test_interactions, self.user_id_mapping, self.item_id_mapping

    def _split_users(self, train_ratio, random_state):
        print("[DataLoader] Splitting users into training and testing sets...")

        user_ids = list(self.user_id_mapping.keys())
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(user_ids)

        split_index = int(len(user_ids) * train_ratio)
        self.train_users = set(user_ids[:split_index])
        self.test_users = set(user_ids[split_index:])

        print(f"[DataLoader] Number of training users: {len(self.train_users)}")
        print(f"[DataLoader] Number of testing users: {len(self.test_users)}")

    def _create_train_rating_matrix(self, rating_matrix):
        print("[DataLoader] Creating training rating matrix...")

        # Get the indices of the training users
        train_user_indices = [self.user_id_mapping[user_id] for user_id in self.train_users]
        train_user_indices.sort()  # Ensure indices are sorted

        # Extract the rows corresponding to the training users
        train_rating_matrix = rating_matrix[train_user_indices, :]

        print(f"[DataLoader] Training rating matrix shape: {train_rating_matrix.shape}")
        print(f"[DataLoader] Number of non-zero entries: {train_rating_matrix.nnz}")

        return train_rating_matrix

    def _get_test_interactions(self):
        print("[DataLoader] Preparing testing interactions...")

        # Filter interactions to include only those from testing users
        test_interactions = self.interactions[self.interactions['user_id'].isin(self.test_users)]

        print(f"[DataLoader] Number of testing interactions: {test_interactions.shape[0]}")

        return test_interactions

    def get_user_indices(self, user_ids):
        return [self.user_id_mapping[user_id] for user_id in user_ids if user_id in self.user_id_mapping]

    def get_item_indices(self, item_ids):
        return [self.item_id_mapping[item_id] for item_id in item_ids if item_id in self.item_id_mapping]
