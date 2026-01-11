
import os

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ARTIFACT_DIR = "artifacts"

# Persistent cache directory (creates 'model_cache' in the current persistent repo folder)
MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
