
import numpy as np
from src.embeddings import embed_texts
from src.config import EMBED_MODEL

def search(query: str, index, docs: list, top_k: int = 10, filters: dict = None) -> list:
    """
    Semantic search with optional metadata filtering.
    """
    # 1. Embed query
    # Note: embed_texts returns (N, D), we need (1, D)
    q_emb = embed_texts([query], EMBED_MODEL, batch_size=1)[0].reshape(1, -1)
    
    # 2. Search FAISS (Fetch 5x top_k to allow for filtering attrition)
    fetch_k = top_k * 5
    scores, indices = index.search(q_emb, fetch_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(docs): 
            continue
            
        doc = docs[idx]
        meta = doc.get("meta", {})
        
        # 3. Apply Filters
        if filters:
            match = True
            for key, val in filters.items():
                # Example: filter by "locations" (list check)
                if key == "location" and isinstance(meta.get("locations"), list):
                    if val not in meta["locations"]:
                        match = False; break
                # Example: filter by "min_rating"
                elif key == "min_rating":
                    if meta.get("rating", 0) < val:
                        match = False; break
            if not match:
                continue
        
        results.append({
            "id": str(idx),
            "score": float(score),
            "text": doc["text"],
            "meta": meta
        })
        
        if len(results) >= top_k:
            break
            
    return results
