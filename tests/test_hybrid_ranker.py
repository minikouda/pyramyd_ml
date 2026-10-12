from __future__ import annotations

from src.ranker import hybrid_score


def test_hybrid_score_adds_fields_and_sorts(toy_results: list[dict]) -> None:
    ranked = hybrid_score(toy_results, {"salary": 0.5, "rating": 0.1}, alpha=0.7)

    assert len(ranked) == len(toy_results)
    assert all("hybrid_score" in r for r in ranked)
    assert all("semantic_score_norm" in r for r in ranked)
    assert all("utility_score" in r for r in ranked)

    # Must be sorted descending by hybrid_score
    scores = [r["hybrid_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
