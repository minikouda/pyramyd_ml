from sentence_transformers import SentenceTransformer
import numpy as np

def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")

