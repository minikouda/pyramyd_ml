
from __future__ import annotations

from typing import Any

import numpy as np

from src.config import EMBED_MODEL
from src.embeddings import embed_query


def _matches_filters(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True

    for key, val in filters.items():
        if key == "location":
            if not isinstance(val, str) or not val.strip():
                continue
            target = val.strip().lower()
            locs = meta.get("locations")
            if isinstance(locs, list):
                if not any(str(x).strip().lower() == target for x in locs):
                    return False
            elif isinstance(locs, str):
                if target not in locs.lower():
                    return False
            else:
                return False

        elif key == "min_rating":
            try:
                thr = float(val)
            except Exception:
                continue
            rating = meta.get("rating")
            try:
                if float(rating) < thr:
                    return False
            except Exception:
                return False

        elif key == "min_salary":
            try:
                thr = float(val)
            except Exception:
                continue
            salary = meta.get("salary_median")
            try:
                if float(salary) < thr:
                    return False
            except Exception:
                return False

        elif key == "max_salary":
            try:
                thr = float(val)
            except Exception:
                continue
            salary = meta.get("salary_median")
            try:
                if float(salary) > thr:
                    return False
            except Exception:
                return False

        else:
            # Exact-match fallback
            if meta.get(key) != val:
                return False

    return True


def search(
    query: str,
    index,
    docs: list[dict[str, Any]],
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search with optional metadata filtering.
    """
    if top_k <= 0:
        return []

    # 1) Embed query
    q_emb = embed_query(query, EMBED_MODEL).astype(np.float32, copy=False)
    
    # 2) Search FAISS (fetch extra to allow for filter attrition)
    fetch_k = max(int(top_k) * 5, int(top_k))
    scores, indices = index.search(q_emb, fetch_k)
    
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(docs): 
            continue
            
        doc = docs[idx]
        meta = doc.get("meta", {}) or {}

        # 3) Apply filters
        if filters and not _matches_filters(meta, filters):
            continue
        
        results.append({
            "id": str(doc.get("id", idx)),
            "score": float(score),
            "text": doc.get("text", ""),
            "meta": meta
        })
        
        if len(results) >= top_k:
            break
            
    return results
