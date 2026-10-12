from __future__ import annotations

from src.ltr import TrainRow, apply_reranker, train_pointwise_reranker


def test_train_and_apply_reranker(toy_results: list[dict]) -> None:
    # Build a tiny training set with one positive and the rest negative.
    rows = [
        TrainRow(query="company b", result=toy_results[1], label=1),
        TrainRow(query="company b", result=toy_results[0], label=0),
        TrainRow(query="company b", result=toy_results[2], label=0),
        TrainRow(query="work life balance", result=toy_results[0], label=1),
        TrainRow(query="work life balance", result=toy_results[1], label=0),
        TrainRow(query="work life balance", result=toy_results[2], label=1),
    ]

    reranker = train_pointwise_reranker(rows)
    assert reranker.feature_names

    out = apply_reranker(query="work life balance", results=toy_results, reranker=reranker)
    assert len(out) == len(toy_results)
    assert all("ltr_score" in r for r in out)
    assert all("combined_score" in r for r in out)

    combined = [r["combined_score"] for r in out]
    assert combined == sorted(combined, reverse=True)
