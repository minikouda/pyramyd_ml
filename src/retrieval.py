from __future__ import annotations

import math
import re
from dataclasses import dataclass
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


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Index:
    """A lightweight BM25 index.

    This is intentionally dependency-free (no rank_bm25) and designed for
    small-to-medium corpora used in this repo.
    """

    doc_ids: list[str]
    doc_len: np.ndarray  # shape (N,)
    avgdl: float
    idf: dict[str, float]
    postings: dict[str, list[tuple[int, int]]]  # term -> [(doc_idx, tf), ...]
    k1: float = 1.5
    b: float = 0.75


def build_bm25_index(
    docs: list[dict[str, Any]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    max_terms_per_doc: int = 50_000,
) -> BM25Index:
    """Build a BM25 index over `docs[*]['text']`.

    The BM25 scores are computed with Robertson/Sparck Jones IDF:
    $idf(t) = log(1 + (N - df + 0.5)/(df + 0.5))$.
    """

    n = len(docs)
    doc_ids = [str(d.get("id", i)) for i, d in enumerate(docs)]

    postings: dict[str, list[tuple[int, int]]] = {}
    df: dict[str, int] = {}
    doc_len = np.zeros((n,), dtype=np.int32)

    for i, d in enumerate(docs):
        tokens = _tokenize(str(d.get("text", "")))
        if max_terms_per_doc and len(tokens) > max_terms_per_doc:
            tokens = tokens[:max_terms_per_doc]
        doc_len[i] = len(tokens)

        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        for t, c in tf.items():
            postings.setdefault(t, []).append((i, c))
        for t in tf.keys():
            df[t] = df.get(t, 0) + 1

    avgdl = float(doc_len.mean()) if n > 0 else 0.0

    idf: dict[str, float] = {}
    for t, dfi in df.items():
        # Always positive, stable for df=0 edge cases.
        idf[t] = math.log(1.0 + (n - float(dfi) + 0.5) / (float(dfi) + 0.5))

    return BM25Index(
        doc_ids=doc_ids,
        doc_len=doc_len,
        avgdl=avgdl,
        idf=idf,
        postings=postings,
        k1=float(k1),
        b=float(b),
    )


def sparse_search(
    query: str,
    bm25: BM25Index,
    docs: list[dict[str, Any]],
    *,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Sparse retrieval using BM25 over `docs`.

    Returns the same result schema as dense `search()`, but score is BM25.
    """

    if top_k <= 0:
        return []
    if not docs:
        return []

    terms = _tokenize(query)
    if not terms:
        return []

    n = len(docs)
    scores = np.zeros((n,), dtype=np.float32)

    k1 = float(bm25.k1)
    b = float(bm25.b)
    avgdl = float(bm25.avgdl) if bm25.avgdl > 0 else 1.0
    dl = bm25.doc_len.astype(np.float32, copy=False)
    denom_base = k1 * (1.0 - b + b * (dl / avgdl))

    # Accumulate BM25 scores for each query term.
    for t in terms:
        idf = bm25.idf.get(t)
        if idf is None:
            continue
        for doc_idx, tf in bm25.postings.get(t, []):
            tf_f = float(tf)
            scores[doc_idx] += float(idf) * (tf_f * (k1 + 1.0)) / (tf_f + denom_base[doc_idx])

    # Get top candidates by score, then apply filters.
    k = min(int(top_k) * 5, n)
    if k <= 0:
        return []

    # Argpartition for speed; then sort those candidates.
    cand_idx = np.argpartition(-scores, kth=max(0, k - 1))[:k]
    cand_idx = cand_idx[np.argsort(-scores[cand_idx])]

    out: list[dict[str, Any]] = []
    for idx in cand_idx.tolist():
        doc = docs[idx]
        meta = doc.get("meta", {}) or {}
        if filters and not _matches_filters(meta, filters):
            continue

        out.append(
            {
                "id": str(doc.get("id", idx)),
                "score": float(scores[idx]),
                "sparse_score": float(scores[idx]),
                "text": doc.get("text", ""),
                "meta": meta,
            }
        )
        if len(out) >= top_k:
            break
    return out


def rrf_fuse(
    result_sets: dict[str, list[dict[str, Any]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion (RRF).

    For each system s and document d with rank r_s(d) (1-indexed):
    $score(d) += w_s / (k + r_s(d))$.

    Returns a list of fused rows with diagnostic fields:
    - id
    - rrf_score
    - ranks: per-system rank
    - raw_scores: per-system original score (if present)
    """

    if not result_sets:
        return []

    k = int(k)
    if k <= 0:
        k = 60

    w = {name: float(v) for name, v in (weights or {}).items()}

    fused: dict[str, dict[str, Any]] = {}
    for name, rows in result_sets.items():
        if not rows:
            continue
        weight = float(w.get(name, 1.0))
        for rank, r in enumerate(rows, start=1):
            rid = str(r.get("id", ""))
            if not rid:
                continue

            entry = fused.get(rid)
            if entry is None:
                entry = {
                    "id": rid,
                    "rrf_score": 0.0,
                    "ranks": {},
                    "raw_scores": {},
                }
                fused[rid] = entry

            entry["rrf_score"] = float(entry["rrf_score"]) + weight * (1.0 / float(k + rank))
            entry["ranks"][name] = int(rank)
            if "score" in r:
                try:
                    entry["raw_scores"][name] = float(r.get("score"))
                except Exception:
                    pass

    out = list(fused.values())
    out.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)
    return out


def hybrid_search(
    query: str,
    *,
    index: Any | None,
    docs: list[dict[str, Any]],
    bm25: BM25Index | None = None,
    top_k: int = 10,
    dense_top_k: int = 50,
    sparse_top_k: int = 50,
    filters: dict[str, Any] | None = None,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval: dense + sparse fused via RRF.

    - Runs dense retrieval if `index` is provided.
    - Runs sparse retrieval if `bm25` is provided.
    - Fuses via RRF and returns Top-K unified results.

    Output rows include:
    - score: fused score (rrf_score)
    - dense_score / sparse_score when available
    - ranks/raw_scores diagnostics
    """

    if top_k <= 0:
        return []

    result_sets: dict[str, list[dict[str, Any]]] = {}

    dense_results: list[dict[str, Any]] = []
    if index is not None:
        dense_results = search(query, index, docs, top_k=int(dense_top_k), filters=filters)
        result_sets["dense"] = dense_results

    sparse_results: list[dict[str, Any]] = []
    if bm25 is not None:
        sparse_results = sparse_search(query, bm25, docs, top_k=int(sparse_top_k), filters=filters)
        result_sets["sparse"] = sparse_results

    fused = rrf_fuse(result_sets, k=int(rrf_k), weights=weights)

    # Build quick lookup for text/meta.
    by_id: dict[str, dict[str, Any]] = {str(d.get("id", i)): d for i, d in enumerate(docs)}
    dense_by_id = {str(r.get("id", "")): r for r in dense_results}
    sparse_by_id = {str(r.get("id", "")): r for r in sparse_results}

    out: list[dict[str, Any]] = []
    for row in fused[: int(top_k)]:
        rid = str(row.get("id", ""))
        doc = by_id.get(rid, {})

        base = dense_by_id.get(rid) or sparse_by_id.get(rid) or {}
        meta = (base.get("meta") or doc.get("meta") or {})
        text = base.get("text") if "text" in base else doc.get("text", "")

        out.append(
            {
                "id": rid,
                "score": float(row.get("rrf_score", 0.0)),
                "rrf_score": float(row.get("rrf_score", 0.0)),
                "text": text,
                "meta": meta,
                "ranks": row.get("ranks", {}),
                "raw_scores": row.get("raw_scores", {}),
                "dense_score": (dense_by_id.get(rid) or {}).get("score"),
                "sparse_score": (sparse_by_id.get(rid) or {}).get("score"),
            }
        )

    return out


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
