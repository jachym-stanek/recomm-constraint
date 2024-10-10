# data_aggregator.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os
import json
import time

from src.settings import Settings


class DataAggregator:
    def __init__(self, settings: Settings):
        self.user_id_mapping = None
        self.item_id_mapping = None

        self.users = None
        self.items = None
        self.interactions = None

        self.settings = settings

    def aggregate(self, dataset_name):
        start = time.time()

        # Ensure dataset is configured
        self.settings.set_dataset_in_use(dataset_name)

        print(f"[DataAggregator] Aggregating data for dataset '{dataset_name}'...")

        self.load_data(dataset_name)
        print(f"[DataAggregator] Using aggregation setting: '{self.settings.aggregation_settings}'")
        print(f"[DataAggregator] Interaction weights: {self.settings.interaction_weights}")

        self.apply_interaction_weights(self.settings.interaction_weights, self.settings.aggregation_settings)
        self.create_rating_matrix()
        self.save_rating_matrix(dataset_name)

        end = time.time()
        print(f"[DataAggregator] Data aggregation completed in {end - start:.2f} seconds.")

    def load_data(self, dataset_name):
        print(f"[DataAggregator] Loading data for dataset '{dataset_name}'...")

        dataset_dir = self.settings.dataset.get('transformed_data_dir')
        users_file = os.path.join(dataset_dir, 'users.csv')
        items_file = os.path.join(dataset_dir, 'items.csv')
        interactions_file = os.path.join(dataset_dir, 'interactions.csv')

        # Load data
        self.users = pd.read_csv(users_file)
        self.items = pd.read_csv(items_file)
        self.interactions = pd.read_csv(interactions_file)

        # Create ID mappings
        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(self.users['user_id'])}
        self.item_id_mapping = {item_id: idx for idx, item_id in enumerate(self.items['item_id'])}

    def apply_interaction_weights(self, interaction_weights, aggregation_setting):
        print("[DataAggregator] Applying interaction weights...")

        # Filter interactions based on source
        if aggregation_setting == 'explicit_only':
            interactions = self.interactions[self.interactions['source'] == 'explicit']
        elif aggregation_setting == 'implicit_only':
            interactions = self.interactions[self.interactions['source'] == 'implicit']
        elif aggregation_setting == 'explicit_and_implicit':
            interactions = self.interactions.copy()
        else:
            raise ValueError(f"[DataAggregator] Invalid aggregation setting: '{aggregation_setting}'")

        self.interactions = interactions

        # Map interaction types to weights
        self.interactions['weight'] = self.interactions['interaction_type'].map(interaction_weights)
        self.interactions['weight'].fillna(0.0, inplace=True)

        # Calculate weighted interaction values
        self.interactions['weighted_value'] = self.interactions['interaction_value'] * self.interactions['weight']

    def create_rating_matrix(self):
        print("[DataAggregator] Creating rating matrix...")

        # Map user and item IDs to indices
        self.interactions['user_idx'] = self.interactions['user_id'].map(self.user_id_mapping)
        self.interactions['item_idx'] = self.interactions['item_id'].map(self.item_id_mapping)

        # Aggregate interactions (sum weighted values for each user-item pair)
        aggregated = self.interactions.groupby(['user_idx', 'item_idx'])['weighted_value'].sum().reset_index()

        # Create sparse rating matrix
        num_users = len(self.user_id_mapping)
        num_items = len(self.item_id_mapping)
        row_indices = aggregated['user_idx'].values
        col_indices = aggregated['item_idx'].values
        data = aggregated['weighted_value'].values

        self.rating_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_users, num_items))

        # Log rating matrix info
        num_nonzero = self.rating_matrix.nnz
        print(f"[DataAggregator] Rating matrix size: {num_users} users x {num_items} items")
        print(f"[DataAggregator] Number of non-zero entries: {num_nonzero}")
        print(f"[DataAggregator] Data format: Compressed Sparse Row (CSR)")

    def save_rating_matrix(self, dataset_name):
        print("[DataAggregator] Saving rating matrix...")

        os.makedirs(self.settings.dataset.get('transformed_data_dir'), exist_ok=True)
        matrix_file = os.path.join(self.settings.dataset.get('transformed_data_dir'), f"{dataset_name}_rating_matrix.npz")
        save_npz(matrix_file, self.rating_matrix)

        # Save mappings
        mappings_file = os.path.join(self.settings.dataset.get('transformed_data_dir'), f"{dataset_name}_id_mappings.json")
        id_mappings = {
            'user_id_mapping': self.user_id_mapping,
            'item_id_mapping': self.item_id_mapping
        }
        with open(mappings_file, 'w') as f:
            json.dump(id_mappings, f)

        print(f"[DataAggregator] Rating matrix and ID mappings saved to '{self.settings.dataset.get('transformed_data_dir')}'.")


if __name__ == "__main__":
    settings = Settings()
    data_aggregator = DataAggregator(settings)
    data_aggregator.aggregate('movielens')
