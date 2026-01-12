from __future__ import annotations

from typing import Any


def build_context(results: list[dict[str, Any]], *, max_chars: int = 700) -> str:
    """Format retrieval results into numbered context snippets.

    The numbering ([1], [2], ...) is what the model should cite in its answer.
    """

    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        meta = r.get("meta", {}) or {}
        company = meta.get("company", "Unknown")
        row = meta.get("row_index")
        rating = meta.get("rating")
        salary = meta.get("salary_median")

        header = [f"[{i}] company={company}"]
        if row is not None:
            header.append(f"row={row}")
        if rating is not None:
            header.append(f"rating={rating}")
        if salary:
            try:
                header.append(f"salary_median={float(salary):,.0f}")
            except Exception:
                header.append(f"salary_median={salary}")

        text = (r.get("text") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " …"
        parts.append(" | ".join(header) + "\n" + text)

    return "\n\n".join(parts)


def ask_rag(
    query: str,
    results: list[dict[str, Any]],
    model,
    tokenizer,
    *,
    max_new_tokens: int = 400,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """Grounded RAG answer using only retrieved docs.

    `results` should be output of retrieval/ranking (each item has text/meta/score).
    """

    context_str = build_context(results)

    system = (
        "You are a careful assistant. Use ONLY the provided context snippets. "
        "If the context is insufficient, say you don't know. "
        "Cite sources inline using the bracket numbers like [1], [2]. "
        "Do NOT invent facts or citations."
    )
    user = (
        f"Query: {query}\n\n"
        f"Context snippets:\n{context_str}\n\n"
        "Return:\n"
        "- A short answer\n"
        "- Top 3 recommendations (if applicable)\n"
        "- Bullet reasons with citations"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Lazy import so `build_context()` can be used without heavy deps like torch.
    from src.llm import generate_chat

    return generate_chat(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
