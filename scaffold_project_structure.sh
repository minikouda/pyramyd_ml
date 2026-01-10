#!/usr/bin/env bash
set -e

echo "📦 Scaffolding AI Product Discovery RAG project..."

# ========== Create directories ==========
mkdir -p notebooks src scripts artifacts data
touch src/__init__.py

# ========== .gitignore ==========
if [ ! -f .gitignore ]; then
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
.env

# Data & model artifacts
data/
artifacts/
*.npy
*.pkl
*.index
*.zip

# OS
.DS_Store
EOF
echo "✔ .gitignore created"
else
echo "ℹ .gitignore already exists — skipping"
fi

# ========== requirements.txt ==========
if [ ! -f requirements.txt ]; then
cat > requirements.txt <<'EOF'
pandas
numpy
matplotlib
scikit-learn
tqdm
sentence-transformers
faiss-cpu
transformers
accelerate
bitsandbytes
gradio
torch
EOF
echo "✔ requirements.txt created"
else
echo "ℹ requirements.txt already exists — skipping"
fi

# ========== README.md ==========
if [ ! -f README.md ]; then
cat > README.md <<'EOF'
# AI Product Discovery with RAG

Semantic search + hybrid ranking + grounded LLM comparisons over company reviews.

## Structure
notebooks/    # Jupyter / Colab notebooks
src/          # Core pipeline logic
scripts/      # Entry points
data/         # Raw CSV (gitignored)
artifacts/    # Embeddings, FAISS index (gitignored)

## Quickstart
Open notebooks/AI_Product_Discovery_RAG.ipynb
EOF
echo "✔ README.md created"
else
echo "ℹ README.md already exists — skipping"
fi

# ========== src modules ==========
create_file () {
  FILE=$1
  CONTENT=$2

  if [ ! -f "$FILE" ]; then
    printf "%s\n" "$CONTENT" > "$FILE"
    echo "✔ Created $FILE"
  else
    echo "ℹ $FILE already exists — skipping"
  fi
}

create_file src/config.py \
'# Global configuration
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "google/gemma-2b-it"
ARTIFACT_DIR = "artifacts"
'

create_file src/data.py \
'import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)
'

create_file src/embeddings.py \
'from sentence_transformers import SentenceTransformer
import numpy as np

def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")
'

create_file src/index.py \
'import faiss

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index
'

create_file src/ranker.py \
'def hybrid_score(sim, structured, w_sem=0.6):
    return w_sem * sim + (1 - w_sem) * structured
'

create_file src/rag.py \
'def build_prompt(query, context):
    return f"""Answer the question using only the context.

Query:
{query}

Context:
{context}
"""
'

create_file src/ui.py \
'# Gradio UI entry point (to be implemented)
'

# ========== scripts ==========
if [ ! -f scripts/run_ui.py ]; then
cat > scripts/run_ui.py <<'EOF'
from src.ui import *

if __name__ == "__main__":
    print("Run Gradio UI here")
EOF
echo "✔ scripts/run_ui.py created"
else
echo "ℹ scripts/run_ui.py already exists — skipping"
fi

echo "✅ Project structure ready."
