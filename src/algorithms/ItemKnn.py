import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    def nearest_neighbors(self, items: list, embeddings, N: int, K: int):
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

    def nearest_neighbors_precomputed(self, items: list, similarities, N: int, K: int):
        # num_items = len(similarities)
        # accumulated_similarities = np.zeros(num_items)
        #
        # for item in items:
        #     # Accumulate similarities
        #     accumulated_similarities += similarities[item]
        #
        # # Exclude items already interacted with by setting their similarities to negative infinity
        # accumulated_similarities[items] = -np.inf
        #
        # # Get the indices of the top K items with highest accumulated similarities
        # top_K_indices = np.argpartition(accumulated_similarities, -N)[-N:]
        # # Sort the top K indices in descending order of similarity
        # top_K_indices = top_K_indices[np.argsort(accumulated_similarities[top_K_indices])[::-1]]
        # scores = accumulated_similarities[top_K_indices]
        #
        # return top_K_indices, scores

        recommended_items = {}
        for item in items:
            similar_items = similarities[item]  # Get top K similar items
            for sim_item in similar_items:
                if sim_item not in items and sim_item not in recommended_items:
                    recommended_items[sim_item] = similarities[item][sim_item]
                elif sim_item in recommended_items:
                    recommended_items[sim_item] += similarities[item][sim_item]
        # Sort recommended items by similarity score
        recommendations = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)
        items, scores = zip(*recommendations)
        return items[:N], scores[:N]

    def compute_similarities(self, item_embeddings, K):
        print("[ItemKnn] Computing item similarities...")
        start = time.time()

        # num_items, embedding_dim = embeddings.shape
        # similarities = np.zeros((num_items, num_items))
        #
        # for i in range(num_items):
        #     emb_i = embeddings[i]
        #     dot_products = embeddings @ emb_i
        #
        #     # Compute norms
        #     emb_i_norm = np.linalg.norm(emb_i)
        #     embeddings_norms = np.linalg.norm(embeddings, axis=1)
        #
        #     # Compute cosine similarities
        #     denom = embeddings_norms * emb_i_norm
        #     epsilon = 1e-10
        #     denom = np.where(denom == 0, epsilon, denom)
        #     cosine_similarities = dot_products / denom
        #
        #     similarities[i] = cosine_similarities

        similarities = cosine_similarity(item_embeddings)

        top_k_neighbors = {}
        for item_id in range(similarities.shape[0]):
            # Get similarity scores for the item
            sim_scores = similarities[item_id]
            # Exclude the item itself by setting its similarity to -inf
            sim_scores[item_id] = -np.inf
            # Get indices of top K similar items
            neighbors = np.argpartition(-sim_scores, K)[:K]
            # Sort neighbors by similarity score
            neighbors = neighbors[np.argsort(-sim_scores[neighbors])]
            neighbors_scores = sim_scores[neighbors]
            top_k_neighbors[item_id] = {}
            for neighbor, score in zip(neighbors, neighbors_scores):
                top_k_neighbors[item_id][neighbor] = score

        print(f"[ItemKnn] Computed item similarities in {time.time() - start:.2f} seconds.")
        return top_k_neighbors
