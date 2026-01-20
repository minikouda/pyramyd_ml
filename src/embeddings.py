from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np


def _get_cache_dir() -> Optional[str]:
    try:
        from src.config import MODEL_CACHE_DIR

        return MODEL_CACHE_DIR
    except Exception:
        return None


def _pick_device() -> str:
    """Pick the best available device for embeddings."""

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str):
    """Load and cache a SentenceTransformer model.

    We cache to avoid re-downloading / reloading the model on every query.
    """

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'sentence-transformers'. Install it to generate embeddings."
        ) from e

    cache_dir = _get_cache_dir()
    return SentenceTransformer(model_name, device=device, cache_folder=cache_dir)


def embed_texts(
    texts: list[str],
    model_name: str,
    *,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Embed texts as a 2D numpy array (N, D).

    Defaults to normalized embeddings so dot-product == cosine similarity.
    """

    device = _pick_device()
    model = _load_sentence_transformer(model_name, device)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Convenience helper: embed a single query into shape (1, D)."""

    v = embed_texts([query], model_name, batch_size=1, show_progress_bar=False)
    return v.reshape(1, -1)
