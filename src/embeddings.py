
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from src.config import MODEL_CACHE_DIR
except ImportError:
    MODEL_CACHE_DIR = None

def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """
    Generates embeddings for a list of texts using SentenceTransformers.
    Auto-detects GPU and normalizes embeddings for cosine similarity.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {model_name} on {device}...")
    
    # Load model (using cache if available)
    model = SentenceTransformer(model_name, device=device, cache_folder=MODEL_CACHE_DIR)
    
    print(f"Embedding {len(texts)} documents (batch_size={batch_size})...")
    
    # Encode
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True, # Vital for dot-product/cosine retrieval
        convert_to_numpy=True
    )
    
    return embeddings
