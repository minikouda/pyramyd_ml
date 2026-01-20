from __future__ import annotations

from src.retrieval import build_bm25_index, hybrid_search, sparse_search


def test_bm25_build_and_search(toy_docs: list[dict]) -> None:
    bm25 = build_bm25_index(toy_docs)
    assert bm25.avgdl > 0
    assert len(bm25.idf) > 0

    res = sparse_search("work life balance", bm25, toy_docs, top_k=3)
    assert len(res) > 0
    assert all("id" in r and "score" in r for r in res)


def test_hybrid_search_sparse_only(toy_docs: list[dict]) -> None:
    bm25 = build_bm25_index(toy_docs)
    fused = hybrid_search("high pay", index=None, docs=toy_docs, bm25=bm25, top_k=2)
    assert len(fused) == 2
    assert all("rrf_score" in r for r in fused)
