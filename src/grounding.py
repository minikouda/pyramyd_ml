from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CIT_RE = re.compile(r"\[(\d+)\]")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def extract_citations(text: str) -> list[int]:
    """Extract bracket citations like [1], [2] from text."""

    out: list[int] = []
    for m in _CIT_RE.finditer(text or ""):
        try:
            out.append(int(m.group(1)))
        except Exception:
            continue
    return out


def parse_context_snippets(context: str) -> list[str]:
    """Parse the `src.rag.build_context()` string into a list of snippet texts.

    Each snippet in build_context is separated by blank lines; header line contains [i].
    We return only the text portion (below the header).
    """

    if not context:
        return []

    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]
    snippets: list[str] = []
    for b in blocks:
        lines = b.splitlines()
        if not lines:
            continue
        if len(lines) == 1:
            snippets.append("")
            continue
        snippets.append("\n".join(lines[1:]).strip())
    return snippets


def citation_coverage(answer: str, *, num_snippets: int) -> dict[str, Any]:
    """Simple citation coverage checks.

    This is a heuristic metric (not a full verifier):
    - counts citations
    - checks out-of-range citations
    - estimates per-line coverage: fraction of non-empty lines that contain a citation
    """

    citations = extract_citations(answer)
    unique = sorted(set(citations))

    oob = [c for c in unique if c < 1 or c > int(num_snippets)]

    lines = [ln.strip() for ln in (answer or "").splitlines()]
    non_empty = [ln for ln in lines if ln]
    cited_lines = [ln for ln in non_empty if _CIT_RE.search(ln)]

    return {
        "num_snippets": int(num_snippets),
        "num_citations": int(len(citations)),
        "unique_citations": unique,
        "out_of_range_citations": oob,
        "line_coverage": (len(cited_lines) / len(non_empty)) if non_empty else 0.0,
    }


def claim_evidence_overlap(
    answer: str,
    snippets: list[str],
    *,
    cited_only: bool = True,
) -> dict[str, Any]:
    """Heuristic claim–evidence overlap.

    Computes token Jaccard overlap between the answer and:
    - cited snippets only (default), or
    - all snippets.
    """

    if not snippets:
        return {"avg_jaccard": 0.0, "max_jaccard": 0.0, "used_snippet_indices": []}

    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return {"avg_jaccard": 0.0, "max_jaccard": 0.0, "used_snippet_indices": []}

    used_idx: list[int] = []
    if cited_only:
        used_idx = sorted({c - 1 for c in extract_citations(answer) if c >= 1 and c <= len(snippets)})
    else:
        used_idx = list(range(len(snippets)))

    if not used_idx:
        return {"avg_jaccard": 0.0, "max_jaccard": 0.0, "used_snippet_indices": []}

    scores: list[float] = []
    for i in used_idx:
        s_tokens = _tokenize(snippets[i])
        if not s_tokens:
            scores.append(0.0)
            continue
        inter = len(ans_tokens & s_tokens)
        union = len(ans_tokens | s_tokens)
        scores.append((inter / union) if union else 0.0)

    return {
        "avg_jaccard": sum(scores) / len(scores) if scores else 0.0,
        "max_jaccard": max(scores) if scores else 0.0,
        "used_snippet_indices": used_idx,
    }
