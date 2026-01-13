
from __future__ import annotations

import os
from pathlib import Path

# Repo root (../ from this file: src/config.py -> repo_root)
REPO_ROOT = Path(__file__).resolve().parents[1]

# LLM (Using cached 7B model)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Embeddings: BAAI/bge-large-en-v1.5 is SOTA and fits easily on L4
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# Persist artifacts & caches under the repo by default.
# Override in Colab (e.g. Drive) via env var for persistence across runtimes.
ARTIFACT_DIR = os.environ.get("PYRAMYD_ARTIFACT_DIR", str(REPO_ROOT / "artifacts"))
MODEL_CACHE_DIR = os.environ.get("PYRAMYD_MODEL_CACHE_DIR", str(REPO_ROOT / "model_cache"))

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
