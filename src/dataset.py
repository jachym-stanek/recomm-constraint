class Dataset:
    def __init__(self, matrix):
        self.matrix = matrix
        self.num_users, self.num_items = matrix.shape

    def __len__(self):
        return self.num_users
