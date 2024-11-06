# settings.py

import yaml
import os

class Settings:
    def __init__(self, config_file=None):
        self._config = {}
        self.dataset_in_use = 'movielens' # default dataset
        self._load_defaults()
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        else:
            print("[Settings] Config file not provided or not found. Using default settings.")

    def set_dataset_in_use(self, dataset_name):
        self.dataset_in_use = dataset_name

    def _load_defaults(self):
        # Default settings
        self._config = {
            'datasets': {
                'movielens': {
                    'movies_file': '../data/movielens_raw/movie.csv',
                    'ratings_file': '../data/movielens_raw/rating.csv',
                    'tags_file': '../data/movielens_raw/tag.csv',
                    'genome_scores_file': '../data/movielens_raw/genome_scores.csv',
                    'genome_tags_file': '../data/movielens_raw/genome_tags.csv',
                    'include_tags': False,
                    'transformed_data_dir': '../data/movielens',
                    'rating_matrix_file': '../data/movielens/rating_matrix.npz',
                    'id_mappings_file': '../data/movielens/movielens_id_mappings.json',
                },
                'bookcrossing': {
                    'users_file': '../data/bookcrossing_raw/BX-Users.csv',
                    'books_file': '../data/bookcrossing_raw/BX-Books.csv',
                    'ratings_file': '../data/bookcrossing_raw/BX-Book-Ratings.csv',
                    'transformed_data_dir': '../data/bookcrossing',
                    'rating_matrix_file': '../data/bookcrossing/rating_matrix.npz',
                }

            },
            'aggregation_settings': {
                'movielens': 'explicit_only',  # Options: explicit_only, implicit_only, explicit_and_implicit
                'bookcrossing': 'explicit_and_implicit',
            },
            'interaction_weights': {
                'movielens': {
                    'rating': 1.0,
                },
                'bookcrossing': {
                    'rating': 1.0,
                }
            },
            'segmentation': {
                'movielens': {
                    'properties': ['genre'],
                }
            },
            'split': {
                'train_ratio': 0.992,
                'random_state': 10,
            },
            'candidates': {
                'top_n': 1000,
            },
            'constraints': {
                'segment_property': 'genre',
            },
            'recommendations': {
                'top_n': 10,
                'num_hidden': 20,
            },
            'use_gpu': True,
            'logging': {
                'log_every': 10,
            },
        }

    def _load_config_file(self, config_file):
        print(f"[Settings] Loading configuration from {config_file}...")
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
            self._deep_update(self._config, file_config)

    def _deep_update(self, source, overrides):
        # Recursively update the source dict with overrides
        for key, value in overrides.items():
            if isinstance(value, dict) and key in source:
                self._deep_update(source[key], value)
            else:
                source[key] = value

    # Properties to access settings

    @property
    def dataset(self):
        return self._config['datasets'][self.dataset_in_use]

    @property
    def aggregation_settings(self):
        return self._config['aggregation_settings'][self.dataset_in_use]

    @property
    def interaction_weights(self):
        return self._config['interaction_weights'][self.dataset_in_use]

    @property
    def segmentation(self):
        return self._config['segmentation'][self.dataset_in_use]

    @property
    def split(self):
        return self._config['split']

    @property
    def model(self):
        return self._config['model']

    @property
    def candidates(self):
        return self._config['candidates']

    @property
    def constraints(self):
        return self._config['constraints'][self.dataset_in_use]

    @property
    def recommendations(self):
        return self._config['recommendations']

    @property
    def use_gpu(self):
        return self._config['use_gpu']

    @property
    def log_every(self):
        return self._config['logging']['log_every']
