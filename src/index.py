from __future__ import annotations

from typing import Any

import numpy as np


def build_index(embeddings: np.ndarray) -> Any:
    """Build a FAISS index for inner-product similarity.

    If embeddings are L2-normalized, inner-product == cosine similarity.
    """

    try:
        import faiss  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'faiss'. Install 'faiss-cpu' (or 'faiss-gpu' on Colab) to build the index."
        ) from e

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    d = int(embeddings.shape[1])

    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32, copy=False))
    return index
