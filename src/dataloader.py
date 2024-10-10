import random
from collections import defaultdict

class DataLoader:
    def __init__(self, num_users=5, num_items=50, num_segments=5):
        self.num_users = num_users
        self.num_items = num_items
        self.num_segments = num_segments
        self.users = []
        self.items = []
        self.item_properties = {}
        self.segmentations = {}
        self.item_scores = defaultdict(dict)
        self.I_cand = {}
        self.load_data()

    def load_data(self):
        self.generate_users()
        self.generate_items()
        self.extract_segmentations()
        self.generate_scores()
        self.generate_candidate_items()

    def generate_users(self):
        self.users = [f'u_{i+1}' for i in range(self.num_users)]

    def generate_items(self):
        self.items = [f'i_{i+1}' for i in range(self.num_items)]
        # Assign properties to items
        for item in self.items:
            props = {
                'title': f'Title of {item}',
                'genres': random.sample(range(1, self.num_segments+1), random.randint(1, 2)),
                'paid': random.choice([True, False])
            }
            self.item_properties[item] = props

    def extract_segmentations(self):
        # Segmentation 1: Genres
        genres_segmentation = defaultdict(list)
        for item, props in self.item_properties.items():
            for genre in props['genres']:
                genres_segmentation[f'Genre_{genre}'].append(item)
        # Segmentation 2: Paid/Free
        paid_segmentation = {
            'Paid': [item for item, props in self.item_properties.items() if props['paid']],
            'Free': [item for item, props in self.item_properties.items() if not props['paid']]
        }
        self.segmentations = {
            'Genres': genres_segmentation,
            'Paid/Free': paid_segmentation
        }

    def generate_scores(self):
        # For simplicity, generate random scores
        for user in self.users:
            for item in self.items:
                self.item_scores[user][item] = random.uniform(0, 1)

    def generate_candidate_items(self, top_k=20):
        # For each user, select top_k items based on scores
        for user in self.users:
            scores = self.item_scores[user]
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self.I_cand[user] = sorted_items[:top_k]

    def get_data_for_user(self, user):
        return {
            'candidate_items': self.I_cand[user],
            'segmentations': self.segmentations
        }
