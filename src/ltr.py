from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _min_max_norm(xs: Iterable[float]) -> list[float]:
    xs_list = [float(x) for x in xs]
    if not xs_list:
        return []
    vmin = min(xs_list)
    vmax = max(xs_list)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return [0.5 for _ in xs_list]
    return [(x - vmin) / (vmax - vmin) for x in xs_list]


@dataclass(frozen=True)
class TrainRow:
    query: str
    result: dict[str, Any]
    label: int


@dataclass(frozen=True)
class TrainedReranker:
    model: Any
    feature_names: list[str]


def featurize(query: str, r: dict[str, Any]) -> dict[str, float]:
    """Turn (query, result) into a numeric feature dict.

    This is intentionally lightweight and deterministic.
    """

    meta = (r.get("meta") or {})
    text = str(r.get("text", ""))

    q_toks = set(_tokenize(query))
    t_toks = set(_tokenize(text))
    overlap = len(q_toks & t_toks)

    # Core retrieval signals (may be missing depending on the stage).
    dense = _safe_float(r.get("semantic_score_norm", r.get("dense_score", r.get("score", 0.0))))
    sparse = _safe_float(r.get("sparse_score", 0.0))

    # Utility-like signals.
    rating = _safe_float(meta.get("rating"), default=0.0)
    salary = _safe_float(meta.get("salary_median"), default=0.0)

    feats: dict[str, float] = {
        "dense_score": dense,
        "sparse_score": sparse,
        "rating": rating,
        "salary_log10": math.log10(salary) if salary > 0 else 0.0,
        "token_overlap": float(overlap),
        "text_len": float(len(text)),
    }

    return feats


def featurize_matrix(query: str, results: list[dict[str, Any]], *, feature_names: list[str] | None = None):
    """Vectorize features into (X, names)."""

    feat_dicts = [featurize(query, r) for r in (results or [])]
    if not feat_dicts:
        return np.zeros((0, 0), dtype=np.float32), []

    names = feature_names or sorted({k for d in feat_dicts for k in d.keys()})
    x = np.zeros((len(feat_dicts), len(names)), dtype=np.float32)
    for i, d in enumerate(feat_dicts):
        for j, name in enumerate(names):
            x[i, j] = float(d.get(name, 0.0))

    # Normalize a few heavy-range features for stability.
    for col in ("salary_log10", "text_len", "token_overlap"):
        if col in names:
            j = names.index(col)
            x[:, j] = np.asarray(_min_max_norm(x[:, j]), dtype=np.float32)

    return x, names


def train_pointwise_reranker(rows: list[TrainRow]) -> TrainedReranker:
    """Train a simple pointwise reranker.

    Uses scikit-learn LogisticRegression (available in this repo) and keeps the
    implementation dependency-light. This is a practical LTR baseline.
    """

    if not rows:
        raise ValueError("No training rows provided")

    from sklearn.linear_model import LogisticRegression

    # Build per-row features.
    feat_dicts = [featurize(r.query, r.result) for r in rows]
    feature_names = sorted({k for d in feat_dicts for k in d.keys()})

    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int32)

    for i, (fd, row) in enumerate(zip(feat_dicts, rows)):
        for j, name in enumerate(feature_names):
            x[i, j] = float(fd.get(name, 0.0))
        y[i] = int(row.label)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x, y)

    return TrainedReranker(model=model, feature_names=feature_names)


def apply_reranker(
    *,
    query: str,
    results: list[dict[str, Any]],
    reranker: TrainedReranker,
    score_key: str = "ltr_score",
    combine_with: str = "hybrid_score",
    ltr_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Apply a trained reranker and optionally combine with an existing score.

    - Adds `score_key` to each result
    - Adds `combined_score`
    - Returns results sorted by `combined_score` desc
    """

    if not results:
        return []

    x, _ = featurize_matrix(query, results, feature_names=reranker.feature_names)
    if x.shape[0] == 0:
        return results

    model = reranker.model
    # Prefer probabilities when available.
    if hasattr(model, "predict_proba"):
        ltr = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        ltr = model.decision_function(x)
    else:
        ltr = model.predict(x)

    ltr_norm = _min_max_norm([float(v) for v in ltr])

    out: list[dict[str, Any]] = []
    for r, ltr_s in zip(results, ltr_norm):
        base = _safe_float(r.get(combine_with, r.get("score", 0.0)))
        combined = (1.0 - float(ltr_weight)) * base + float(ltr_weight) * float(ltr_s)

        rr = dict(r)
        rr[score_key] = float(ltr_s)
        rr["combined_score"] = float(combined)
        out.append(rr)

    out.sort(key=lambda d: float(d.get("combined_score", 0.0)), reverse=True)
    return out
