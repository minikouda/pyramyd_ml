
import os

# LLM (Using cached 7B model)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Embeddings: BAAI/bge-large-en-v1.5 is SOTA and fits easily on L4
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

ARTIFACT_DIR = "artifacts"

# Persistent cache directory
MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
