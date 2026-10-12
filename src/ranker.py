from __future__ import annotations

import math
from typing import Any


def _min_max_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def hybrid_score(
    results: list[dict[str, Any]],
    priorities: dict[str, float],
    *,
    alpha: float = 0.7,
) -> list[dict[str, Any]]:
    """Deterministic hybrid ranking.

    - Uses semantic score from retrieval (`r['score']`).
    - Uses structured utility from metadata fields (`salary_median`, `rating`, ...).
    - Produces `r['hybrid_score']` and sorts descending.

    The LLM is NOT used here.
    """

    if not results:
        return []

    alpha = float(max(0.0, min(1.0, alpha)))
    weights = {k: float(v) for k, v in (priorities or {}).items() if float(v) > 0}
    w_sum = sum(weights.values())

    # Collect raw features for normalization across the candidate set.
    sem_raw: list[float] = [float(r.get("score", 0.0)) for r in results]
    sem_norm = _min_max_norm(sem_raw)

    rating_raw: list[float] = []
    salary_log_raw: list[float] = []
    for r in results:
        meta = (r.get("meta") or {})

        rating = meta.get("rating")
        try:
            rating_raw.append(float(rating))
        except Exception:
            rating_raw.append(float("nan"))

        salary = meta.get("salary_median")
        try:
            s = float(salary)
            salary_log_raw.append(math.log10(s) if s > 0 else float("nan"))
        except Exception:
            salary_log_raw.append(float("nan"))

    # Replace NaNs with min for stable normalization.
    def _nan_to_min(xs: list[float]) -> list[float]:
        finite = [x for x in xs if math.isfinite(x)]
        if not finite:
            return [0.0 for _ in xs]
        mn = min(finite)
        return [x if math.isfinite(x) else mn for x in xs]

    rating_norm = _min_max_norm(_nan_to_min(rating_raw))
    salary_norm = _min_max_norm(_nan_to_min(salary_log_raw))

    ranked: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        meta = (r.get("meta") or {})

        # Utility in [0,1] computed from normalized structured fields.
        utility = 0.0
        if w_sum > 0:
            if "rating" in weights:
                utility += weights["rating"] * rating_norm[i]
            if "salary" in weights:
                utility += weights["salary"] * salary_norm[i]
            utility /= w_sum

        hybrid = alpha * sem_norm[i] + (1.0 - alpha) * utility

        out = dict(r)
        out["meta"] = meta
        out["semantic_score_raw"] = float(sem_raw[i])
        out["semantic_score_norm"] = float(sem_norm[i])
        out["utility_score"] = float(utility)
        out["utility_components"] = {
            "rating_norm": float(rating_norm[i]),
            "salary_norm": float(salary_norm[i]),
            "weights": dict(weights),
        }
        out["hybrid_score"] = float(hybrid)
        ranked.append(out)

    ranked.sort(key=lambda x: float(x.get("hybrid_score", 0.0)), reverse=True)
    return ranked


def embedding_only_rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baseline: rank purely by embedding similarity score."""

    ranked = [dict(r) for r in (results or [])]
    ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return ranked


def structured_only_rank(
    results: list[dict[str, Any]],
    priorities: dict[str, float],
) -> list[dict[str, Any]]:
    """Baseline: rank purely by structured utility.

    This uses the same normalization as `hybrid_score`, but with alpha=0.
    """

    return hybrid_score(results, priorities, alpha=0.0)
