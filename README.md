# Pyramyd: Decision‑Focused RAG for Product/Company Discovery

**Semantic retrieval · deterministic hybrid ranking · grounded explanations**

Disclaimer: see [DISCLAIMER.md](DISCLAIMER.md)

------------------------------------------------------------------------

## Demo

![Search interface](demo/interface.png)

![Results for the data science query](demo/data_science.png)

![End-to-end demo](demo/demo.gif)

------------------------------------------------------------------------

## What It Does

**Input** - A natural-language query (e.g., “high-paying entry-level roles with strong work-life balance”) - Optional hard constraints (location, salary bounds, remote eligibility) - Optional preference weights (salary vs culture vs growth)

**Output** - A ranked Top‑K list (commonly Top 3) - Per‑item “why it matches” rationale - Global “why this ordering” rationale - Evidence citations (pros and keywords) tied to retrieved snippets - Side-by-side comparison view

------------------------------------------------------------------------

## How It Works (High Level)

1.  **Retrieve candidates** via dense embeddings (FAISS) and optional sparse retrieval (BM25).
2.  **Fuse and/or rerank** candidates deterministically (e.g., RRF fusion; hybrid scoring).
3.  **Apply preferences** via a structured utility model controlled by weights and $\alpha$.
4.  **Generate explanations and side-by-side comparisons** using an LLM constrained to cite evidence.
5.  **Evaluate** retrieval/ranking and run lightweight grounding checks.

The core scoring idea is: 
$$
    \text{score}(i) = \alpha\,\text{semantic}(i) + (1-\alpha)\,\text{utility}(i; \mathbf{w})
$$

------------------------------------------------------------------------

## What’s Implemented

-   **Dense retrieval** with SentenceTransformers embeddings + FAISS index
-   **Sparse retrieval** with BM25 (dependency‑light)
-   **Hybrid retrieval** with RRF fusion (dense + sparse)
-   **Deterministic ranking** that combines semantic match + structured utility features
-   **Optional learning-to-rank** baseline reranker (LogisticRegression; optional LightGBM/XGBoost hooks)
-   **Grounding checks** (citation coverage + simple claim–evidence overlap heuristics)
-   **Tests + CI** via `pytest` and GitHub Actions

------------------------------------------------------------------------

## Repo At A Glance

``` text
.
├── src/                  # core pipeline (retrieval, ranking, rag, eval)
├── notebooks/            # end-to-end notebook demo
├── data/                 # raw datasets (gitignored)
├── artifacts/            # embeddings/index/outputs (gitignored)
├── requirements.txt
└── environment.yml
```

------------------------------------------------------------------------

## Getting Started (Minimal)

**Run the notebook** - Open [notebooks/AI_Product_Discovery_RAG_Structure.ipynb](notebooks/AI_Product_Discovery_RAG_Structure.ipynb) and run top‑to‑bottom.(Make sure you have the required data/artifacts.)

**Run tests**

``` bash
pytest
```

Notes: - `data/` and `artifacts/` are intentionally gitignored. - Cache/output directories can be overridden via env vars (see `src/config.py`).