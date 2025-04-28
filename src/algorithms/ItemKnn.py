import time
import numpy as np
from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_chunked

from src.algorithms.algorithm import Algorithm

class ItemKnn(Algorithm):
    def __init__(self, name="ItemKnn", description="Item-based k-Nearest Neighbors", verbose=False, K=5):
        super().__init__(name, description, verbose)
        self.K = K

    """
    items: set of item indices that were interacted with by the user
    K: number of similar items to find
    embeddings: ALS item embeddings
    
    For each item in the set of interacted items, measure similarity to all other items in the embeddings matrix.
    Sum up the calculated similarities across the interacted items set.
    Select the top K items with the highest sum of similarities.
    Similarity is measured using cosine similarity.
    """
    def nearest_neighbors(self, items: list, embeddings, N: int):
        num_items, embedding_dim = embeddings.shape
        accumulated_similarities = np.zeros(num_items)

        for item in items:
            emb_item = embeddings[item]
            dot_products = embeddings @ emb_item  # Shape: (num_items,)

            # Compute norms
            emb_item_norm = np.linalg.norm(emb_item)
            embeddings_norms = np.linalg.norm(embeddings, axis=1)

            # Compute cosine similarities
            denom = embeddings_norms * emb_item_norm
            epsilon = 1e-10  # Small value to prevent division by zero
            denom = np.where(denom == 0, epsilon, denom)
            cosine_similarities = dot_products / denom

            # Accumulate similarities
            accumulated_similarities += cosine_similarities

        # Exclude items already interacted with by setting their similarities to negative infinity
        accumulated_similarities[items] = -np.inf

        # Get the indices of the top K items with highest accumulated similarities
        top_K_indices = np.argpartition(accumulated_similarities, -N)[-N:]
        # Sort the top K indices in descending order of similarity
        top_K_indices = top_K_indices[np.argsort(accumulated_similarities[top_K_indices])[::-1]]
        scores = accumulated_similarities[top_K_indices]

        return top_K_indices, scores

    def nearest_neighbors_precomputed(self, R, user, user_items: list, neighborhoods, N: int):
        recommended_items = {}
        for item in user_items:
            similar_items = neighborhoods[item]  # Get similar items to user's interacted items
            Rui = R[user, item]  # Rating of the user for the item
            for sim_item in similar_items:
                if sim_item not in user_items and sim_item not in recommended_items:
                    recommended_items[sim_item] = Rui * neighborhoods[item][sim_item]
                elif sim_item in recommended_items:
                    recommended_items[sim_item] += Rui * neighborhoods[item][sim_item]
        # Sort recommended items by similarity score
        recommendations = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)
        if len(recommendations) == 0:
            print("[ItemKnn] WARNING: No recommendations found.")
            return [], []

        items, scores = zip(*recommendations)
        return items[:N], scores[:N]

    def compute_neighborhoods(self, item_embeddings):
        print("[ItemKnn] Computing item embedding neighborhoods...")
        start = time.time()

        similarities = cosine_similarity(item_embeddings)

        top_k_neighbors = {}
        for item_id in range(similarities.shape[0]):
            # Get similarity scores for the item
            sim_scores = similarities[item_id]
            # Exclude the item itself by setting its similarity to -inf
            sim_scores[item_id] = -np.inf
            # Get indices of top K similar items
            neighbors = np.argpartition(-sim_scores, self.K)[:self.K]
            # Sort neighbors by similarity score
            neighbors = neighbors[np.argsort(-sim_scores[neighbors])]
            neighbors_scores = sim_scores[neighbors]
            top_k_neighbors[item_id] = {}
            for neighbor, score in zip(neighbors, neighbors_scores):
                top_k_neighbors[item_id][neighbor] = score

        print(f"[ItemKnn] Computed item neighborhoods in {time.time() - start:.2f} seconds.")
        return top_k_neighbors

    def compute_neighborhoods_chunked(self, emb):
        print("[ItemKnn] Computing item embedding neighborhoods...")
        start_time = time.time()
        top_k = {i: {} for i in range(len(emb))}

        def accumulate(chunk, start):
            sim = 1.0 - chunk  # convert cosine distance -> similarity
            np.fill_diagonal(sim, -np.inf)  # self-sim on this sub-block
            for i, row in enumerate(sim):
                idx = np.argpartition(-row, self.K)[:self.K]
                idx = idx[np.argsort(-row[idx])]
                top_k[start + i] = {j: row[j] for j in idx}

        gen = pairwise_distances_chunked(
            emb,
            metric='cosine',
            working_memory=256,  # MB
            reduce_func=lambda ch, s: accumulate(ch, s),
            n_jobs=-1  # all CPUs
        )
        for _ in gen:  # consume the generator
            pass

        print(f"[ItemKnn] Computed item neighborhoods in {time.time() - start_time:.2f} seconds.")

        return top_k

    def __repr__(self):
        return f"ItemKnn(K={self.K})"
