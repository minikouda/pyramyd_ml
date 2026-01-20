from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

from src.retrieval import search


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class EvalRow:
    query: str
    k: int
    recall_at_k: float
    latency_ms: float
    mrr_at_k: float = 0.0
    ndcg_at_k: float = 0.0


def mrr_at_k(retrieved_ids: Iterable[str], relevant_ids: set[str] | frozenset[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, rid in enumerate(list(retrieved_ids)[:k], start=1):
        if rid in relevant:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(retrieved_ids: Iterable[str], relevant_ids: set[str] | frozenset[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    topk = list(retrieved_ids)[:k]
    dcg = 0.0
    for i, rid in enumerate(topk, start=1):
        rel = 1.0 if rid in relevant else 0.0
        if rel > 0:
            dcg += rel / (math.log2(i + 1))

    # Ideal DCG is all relevant items first (binary relevance)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / (math.log2(i + 1)) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: set[str] | frozenset[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    topk = list(retrieved_ids)[:k]
    hits = sum(1 for rid in topk if rid in relevant)
    return hits / float(len(relevant))


def evaluate_retrieval(
    *,
    index,
    docs: list[dict[str, Any]],
    queries: list[EvalQuery],
    k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for q in queries:
        start = time.perf_counter()
        results = search(q.query, index, docs, top_k=max(k, 1), filters=filters or None)
        latency_ms = (time.perf_counter() - start) * 1000.0
        retrieved_ids = [r.get("id", "") for r in results]
        rows.append(
            EvalRow(
                query=q.query,
                k=k,
                recall_at_k=recall_at_k(retrieved_ids, q.relevant_ids, k=k),
                mrr_at_k=mrr_at_k(retrieved_ids, q.relevant_ids, k=k),
                ndcg_at_k=ndcg_at_k(retrieved_ids, q.relevant_ids, k=k),
                latency_ms=latency_ms,
            )
        )
    return rows


def evaluate_retrieval_fn(
    *,
    search_fn,
    queries: list[EvalQuery],
    k: int = 10,
) -> list[EvalRow]:
    """Evaluate an arbitrary retrieval function.

    `search_fn` should be a callable: (query: str, top_k: int) -> list[dict]
    where each dict has an `id`.
    """

    rows: list[EvalRow] = []
    for q in queries:
        start = time.perf_counter()
        results = search_fn(q.query, top_k=max(k, 1))
        latency_ms = (time.perf_counter() - start) * 1000.0
        retrieved_ids = [str(r.get("id", "")) for r in (results or [])]
        rows.append(
            EvalRow(
                query=q.query,
                k=k,
                recall_at_k=recall_at_k(retrieved_ids, q.relevant_ids, k=k),
                mrr_at_k=mrr_at_k(retrieved_ids, q.relevant_ids, k=k),
                ndcg_at_k=ndcg_at_k(retrieved_ids, q.relevant_ids, k=k),
                latency_ms=latency_ms,
            )
        )
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, float]:
    if not rows:
        return {"avg_recall_at_k": 0.0, "avg_mrr_at_k": 0.0, "avg_ndcg_at_k": 0.0, "avg_latency_ms": 0.0}
    return {
        "avg_recall_at_k": sum(r.recall_at_k for r in rows) / len(rows),
        "avg_mrr_at_k": sum(r.mrr_at_k for r in rows) / len(rows),
        "avg_ndcg_at_k": sum(r.ndcg_at_k for r in rows) / len(rows),
        "avg_latency_ms": sum(r.latency_ms for r in rows) / len(rows),
    }
