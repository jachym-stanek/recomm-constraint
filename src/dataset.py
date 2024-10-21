class Dataset:
    def __init__(self, matrix, user_mapping, item_mapping, user_ids):
        self.matrix = matrix
        self.user_id2idx = user_mapping
        self.item_id2idx = item_mapping
        self.user_idx2id = {idx: user_id for user_id, idx in user_mapping.items()}
        self.item_idx2id = {idx: item_id for item_id, idx in item_mapping.items()}
        self.num_users, self.num_items = matrix.shape
        self.user_ids = user_ids
