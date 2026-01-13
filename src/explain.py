from __future__ import annotations

from typing import Any


def build_explanation_payload(
    *,
    query: str,
    results: list[dict[str, Any]],
    priorities: dict[str, float] | None = None,
    alpha: float | None = None,
    filters: dict[str, Any] | None = None,
    max_snippet_chars: int = 200,
) -> dict[str, Any]:
    """Build a minimal structured explanation payload for UI/debug.

    This is intentionally deterministic and contains *no* LLM-generated content.
    The LLM (if used) should only explain these results, never reorder them.
    """

    payload: dict[str, Any] = {
        "query": query,
        "filters": filters or {},
        "preferences": {
            "alpha": alpha,
            "weights": priorities or {},
        },
        "results": [],
    }

    out_results: list[dict[str, Any]] = []
    for rank, r in enumerate(results or [], start=1):
        meta = (r.get("meta") or {})
        text = (r.get("text") or "")
        snippet = text[:max_snippet_chars].rstrip() if isinstance(text, str) else ""

        out_results.append(
            {
                "rank": rank,
                "id": str(r.get("id", "")),
                "company": meta.get("company", "Unknown"),
                "scores": {
                    "semantic_raw": r.get("semantic_score_raw", r.get("score", 0.0)),
                    "semantic_norm": r.get("semantic_score_norm"),
                    "utility": r.get("utility_score"),
                    "hybrid": r.get("hybrid_score", r.get("score", 0.0)),
                },
                "utility_components": r.get("utility_components", {}),
                "meta": {
                    "row_index": meta.get("row_index"),
                    "rating": meta.get("rating"),
                    "salary_median": meta.get("salary_median"),
                    "locations": meta.get("locations"),
                },
                "snippet": snippet,
            }
        )

    payload["results"] = out_results
    return payload
