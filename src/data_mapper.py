# id_mapping.py

import json

class MatrixMapper:
    def __init__(self, user_id_mapping=None, item_id_mapping=None):
        self.__user_id_to_idx = user_id_mapping if user_id_mapping else {}
        self.__item_id_to_idx = item_id_mapping if item_id_mapping else {}
        self.__user_idx_to_id = {idx: user_id for user_id, idx in self.__user_id_to_idx.items()}
        self.__item_idx_to_id = {idx: item_id for item_id, idx in self.__item_id_to_idx.items()}

    @classmethod
    def load_from_files(cls, mappings_file):
        with open(mappings_file, 'r') as f:
            id_mappings = json.load(f)
        user_id_mapping = {int(k): int(v) for k, v in id_mappings['user_id_mapping'].items()}
        item_id_mapping = {int(k): int(v) for k, v in id_mappings['item_id_mapping'].items()}
        return cls(user_id_mapping, item_id_mapping)

    def user_id2idx(self, user_id):
        return self.__user_id_to_idx.get(user_id)

    def user_idx2id(self, user_idx):
        return self.__user_idx_to_id.get(user_idx)

    def item_id2idx(self, item_id):
        return self.__item_id_to_idx.get(item_id)

    def item_idx2id(self, item_idx):
        return self.__item_idx_to_id.get(item_idx)
