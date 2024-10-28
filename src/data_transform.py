# dataset_transformer.py
from xml.etree.ElementInclude import include

import pandas as pd
import os
import json

from raven.utils.serializer import transform

from src.settings import Settings


class DatasetTransformer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.supported_datasets = ['movielens']  # Extend this list as more datasets are supported

    def transform(self, datasets_to_transform: list):
        for dataset_name in datasets_to_transform:
            if dataset_name not in self.supported_datasets:
                raise ValueError(f"[DatasetTransformer] Dataset '{dataset_name}' is not supported.")
            print(f"[DatasetTransformer] Transforming dataset '{dataset_name}' into unified format...")
            # Call the appropriate transformation method
            transform_method = getattr(self, f"_transform_{dataset_name}", None)
            if transform_method is None:
                raise NotImplementedError(f"[DatasetTransformer] Transformation method for dataset '{dataset_name}' is not implemented.")
            transform_method()

    def _transform_movielens(self):
        # Ensure movielens dataset is configured
        settings.set_dataset_in_use('movielens')

        # Paths to raw data files
        movies_file = self.settings.dataset.get('movies_file')
        ratings_file = self.settings.dataset.get('ratings_file')
        tags_file = self.settings.dataset.get('tags_file')
        genome_scores_file = self.settings.dataset.get('genome_scores_file')
        genome_tags_file = self.settings.dataset.get('genome_tags_file')
        include_tags = self.settings.dataset.get('include_tags')
        transformed_data_dir = self.settings.dataset.get('transformed_data_dir')

        # Check if files exist
        required_files = [movies_file, ratings_file]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"[DatasetTransformer] Required file '{file}' not found for MovieLens dataset.")

        # Load raw data
        movies = pd.read_csv(movies_file)
        ratings = pd.read_csv(ratings_file)
        tags = pd.read_csv(tags_file) if tags_file and os.path.exists(tags_file) else pd.DataFrame()
        # Scores not needed for now
        # genome_scores = pd.read_csv(genome_scores_file) if genome_scores_file and os.path.exists(genome_scores_file) else pd.DataFrame()
        # genome_tags = pd.read_csv(genome_tags_file) if genome_tags_file and os.path.exists(genome_tags_file) else pd.DataFrame()

        # Transform users data
        print("[DatasetTransformer] Extracting users data...")
        users = ratings[['userId']].drop_duplicates()
        users.reset_index(drop=True, inplace=True)
        users['user_id'] = users['userId']
        users.drop(columns=['userId'], inplace=True)
        user_properties = []  # No additional properties in MovieLens
        users_file = os.path.join(transformed_data_dir, 'users.csv')
        users.to_csv(users_file, index=False)
        print(f"[DatasetTransformer] Extracted {users.shape[0]} users with columns: {', '.join(users.columns)}")

        # Transform items data
        print("[DatasetTransformer] Extracting items data...")
        items = movies.rename(columns={'movieId': 'item_id'})
        items['genres'] = items['genres'].apply(lambda x: x.split('|') if pd.notnull(x) else [])
        item_properties = ['title', 'genres']
        items_file = os.path.join(transformed_data_dir, 'items.csv')
        items.to_csv(items_file, index=False)
        print(f"[DatasetTransformer] Extracted {items.shape[0]} items with columns: {', '.join(items.columns)}")

        # Transform interactions data
        print("[DatasetTransformer] Extracting interactions data...")
        interactions = ratings.rename(columns={
            'userId': 'user_id',
            'movieId': 'item_id',
            'rating': 'interaction_value',
            'timestamp': 'timestamp'
        })
        interactions['interaction_type'] = 'rating'
        interactions['source'] = 'explicit'  # All interactions are explicit for MovieLens

        # subtract mean
        interactions['interaction_value'] = interactions['interaction_value'] - interactions['interaction_value'].mean()

        # remove duplicates
        interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'])
        interactions_file = os.path.join(transformed_data_dir, 'interactions.csv')
        interactions.to_csv(interactions_file, index=False)

        # Optionally include tags as interactions
        if not tags.empty and include_tags:
            print("[DatasetTransformer] Including tags as interactions...")
            tags = tags.rename(columns={
                'userId': 'user_id',
                'movieId': 'item_id',
                'tag': 'interaction_value',
                'timestamp': 'timestamp'
            })
            tags['interaction_type'] = 'tag'
            tags_interactions_file = os.path.join(transformed_data_dir, 'tags_interactions.csv')
            tags.to_csv(tags_interactions_file, index=False)

            # Combine ratings and tags interactions
            interactions = pd.concat([interactions, tags], ignore_index=True)
            interactions_file = os.path.join(transformed_data_dir, 'interactions.csv')
            interactions.to_csv(interactions_file, index=False)

        # Create dataset_info.json
        print("[DatasetTransformer] Creating dataset_info.json...")
        dataset_info = {
            'num_users': users.shape[0],
            'num_items': items.shape[0],
            'num_interactions': interactions.shape[0],
            'user_properties': user_properties,
            'item_properties': item_properties,
            'interaction_types': interactions['interaction_type'].unique().tolist(),
        }
        dataset_info_file = os.path.join(transformed_data_dir, 'dataset_info.json')
        with open(dataset_info_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)

        print(f"[DatasetTransformer] Transformation complete. Transformed data saved in '{transformed_data_dir}'.")

if __name__ == "__main__":
    settings = Settings() # Load default settings
    transformer = DatasetTransformer(settings)
    transformer.transform(['movielens'])
