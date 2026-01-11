
import faiss
import numpy as np

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index using Inner Product (equivalent to Cosine Similarity if normalized).
    """
    d = embeddings.shape[1]
    # IndexFlatIP is exact search. Fast enough for <1M vectors on GPU.
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index
