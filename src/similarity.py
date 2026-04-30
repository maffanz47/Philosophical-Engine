import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar(query_vec, all_vecs, metadata, top_k=5):
    """Find top-k most similar texts to a query vector"""
    sims = cosine_similarity([query_vec], all_vecs)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [
        {"source": metadata[i]["label"],
         "score":  round(float(sims[i]), 3),
         "preview": metadata[i]["text"][:100] + "..."}
        for i in top_indices
    ]