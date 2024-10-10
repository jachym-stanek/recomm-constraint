from scipy.sparse import csr_matrix, load_npz
import numpy as np

rating_matrix_file  = '../data/movielens/movielens_rating_matrix.npz'
rating_matrix = load_npz(rating_matrix_file)

print(f"Rating matrix shape: {rating_matrix.shape}")
print(f"Rating matrix density: {rating_matrix.nnz / (rating_matrix.shape[0] * rating_matrix.shape[1]):.5f}")

# find numbers of ratings per rating value
rating_values, rating_counts = np.unique(rating_matrix.data, return_counts=True)
rating_dist = dict(zip(rating_values, rating_counts))
print(f"Rating distribution: {rating_dist}")


