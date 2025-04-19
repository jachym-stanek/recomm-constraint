# dataset_transformer.py

import pandas as pd
import os
import json

from src.settings import Settings


class DatasetTransformer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.supported_datasets = ['movielens', 'industrial_dataset1']

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
        self.settings.set_dataset_in_use('movielens')

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
        items['year'] = items['title'].str[-5:-1] # extract year from title (last 6 characters is '(year)')
        item_properties = ['title', 'genres', 'year']
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

        # leave only positive interactions (ratings > 3)
        interactions = interactions[interactions['interaction_value'] > 3]
        # set all interactions to 1
        # interactions['interaction_value'] = 1.0

        # print average amount of interactions per user
        print(f"[DatasetTransformer] Average amount of interactions per user: {interactions.groupby('user_id').size().mean()}")

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

    def _transform_bookcrossing(self):
        """
        Deprecated
        """
        # Ensure bookcrossing dataset is configured
        self.settings.set_dataset_in_use('bookcrossing')

        # Paths to raw data files
        users_file = self.settings.dataset.get('users_file')
        books_file = self.settings.dataset.get('books_file')
        ratings_file = self.settings.dataset.get('ratings_file')
        transformed_data_dir = self.settings.dataset.get('transformed_data_dir')

        # Check if files exist
        required_files = [users_file, books_file, ratings_file]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"[DatasetTransformer] Required file '{file}' not found for BookCrossing dataset.")

        # Load raw data
        users = pd.read_csv(users_file, sep=';', encoding='latin1')
        books = pd.read_csv(books_file, sep=';', encoding='latin1')
        ratings = pd.read_csv(ratings_file, sep=';', encoding='latin1')

        # Transform users data
        print("[DatasetTransformer] Extracting users data...")
        users = users.rename(columns={
            'User-ID': 'user_id',
            'Location': 'location',
            'Age': 'age'
        })
        user_properties = ['location', 'age']

        # Transform items data
        print("[DatasetTransformer] Extracting items data...")
        items = books.rename(columns={
            'ISBN': 'item_id',
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Year-Of-Publication': 'year_of_publication',
            'Publisher': 'publisher'
        })
        item_properties = ['title', 'author', 'year_of_publication', 'publisher']
        items_file = os.path.join(transformed_data_dir, 'items.csv')
        items.to_csv(items_file, index=False)
        print(f"[DatasetTransformer] Extracted {items.shape[0]} items with columns: {', '.join(items.columns)}")

        # Transform interactions data
        print("[DatasetTransformer] Extracting interactions data...")
        interactions = ratings.rename(columns={
            'User-ID': 'user_id',
            'ISBN': 'item_id',
            'Book-Rating': 'interaction_value'
        })
        interactions['interaction_type'] = 'rating'
        interactions['source'] = 'explicit'

        # subtract median rating
        print(f"[DatasetTransformer] Subtracting median rating (5.0) from interactions...")
        interactions['interaction_value'] = interactions['interaction_value'] - 5.0

        # # leave only positive interactions (ratings > 0)
        # interactions = interactions[interactions['interaction_value'] > 0]
        #
        # # set all interactions to 1
        # interactions['interaction_value'] = 1.0

        # remove duplicates
        interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'])
        interactions_file = os.path.join(transformed_data_dir, 'interactions.csv')
        interactions.to_csv(interactions_file, index=False)

        # remove users with no non-zero interactions
        user_interactions = interactions.groupby('user_id')['interaction_value'].sum()
        users = users[users['user_id'].isin(user_interactions[user_interactions != 0].index)]
        users_file = os.path.join(transformed_data_dir, 'users.csv')
        users.to_csv(users_file, index=False)
        print(f"[DatasetTransformer] Extracted {users.shape[0]} users with columns: {', '.join(users.columns)}")

        # Create dataset_info.json
        print("[DatasetTransformer] Creating dataset_info.json...")
        dataset_info = {
            'num_users': users.shape[0],
            'num_items': items.shape[0],
            'num_interactions': interactions.shape[0],
            'user_properties': user_properties,
            'item_properties': item_properties,
            'interaction_types': ['rating'],
        }

        dataset_info_file = os.path.join(transformed_data_dir, 'dataset_info.json')
        with open(dataset_info_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)

        print(f"[DatasetTransformer] Transformation complete. Transformed data saved in '{transformed_data_dir}'.")

    def _transform_industrial_dataset1(self):
        # all the data is in correct format, we just need to combine the interactions into a single file so that the
        # standardized data aggregator can use it
        self.settings.set_dataset_in_use('industrial_dataset1')

        # Paths to raw data files
        users_file = self.settings.dataset.get('users_file')
        items_file = self.settings.dataset.get('items_file')
        bookmarks_file = self.settings.dataset.get('bookmarks_file')
        detail_views_file = self.settings.dataset.get('detail_views_file')
        purchases_file = self.settings.dataset.get('purchases_file')
        transformed_data_dir = self.settings.dataset.get('transformed_data_dir')

        # Check if files exist
        required_files = [bookmarks_file, detail_views_file, purchases_file]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"[DatasetTransformer] Required file '{file}' not found for Industrial Dataset 1.")

        # Load raw data
        bookmarks = pd.read_csv(bookmarks_file)
        detail_views = pd.read_csv(detail_views_file)
        purchases = pd.read_csv(purchases_file)

        # drop duplicates (where user_id and item_id are the same)
        bookmarks = bookmarks.drop_duplicates(subset=['user_id', 'item_id'])
        detail_views = detail_views.drop_duplicates(subset=['user_id', 'item_id'])
        purchases = purchases.drop_duplicates(subset=['user_id', 'item_id'])

        # combine interactions into a single dataframe
        bookmarks['interaction_type'] = 'bookmark'
        bookmarks['interaction_value'] = 1.0
        bookmarks['source'] = 'implicit'
        bookmarks = bookmarks[['user_id', 'item_id', 'interaction_value', 'timestamp', 'interaction_type', 'source']]
        detail_views['interaction_type'] = 'detail_view'
        detail_views['interaction_value'] = 1.0
        detail_views['source'] = 'implicit'
        detail_views = detail_views[['user_id', 'item_id', 'interaction_value', 'timestamp', 'interaction_type', 'source']]
        purchases['interaction_type'] = 'purchase'
        purchases['interaction_value'] = 1.0
        purchases['source'] = 'implicit'
        purchases = purchases[['user_id', 'item_id', 'interaction_value', 'timestamp', 'interaction_type', 'source']]
        interactions = pd.concat([bookmarks, detail_views, purchases], ignore_index=True)

        # save interactions to file
        interactions_file = os.path.join(transformed_data_dir, 'interactions.csv')
        interactions.to_csv(interactions_file, index=False)

        # load data about users and items
        users = pd.read_csv(users_file)
        items = pd.read_csv(items_file)

        # list user properties
        user_properties = users.columns.tolist()
        user_properties.remove('user_id')

        # list item properties
        item_properties = items.columns.tolist()
        item_properties.remove('item_id')

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
    # transformer.transform(['movielens'])
    transformer.transform(['industrial_dataset1'])
