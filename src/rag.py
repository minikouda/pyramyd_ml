from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class RAGDoc:
    """Lightweight retrieval document used for prompt construction."""

    doc_id: str
    text: str
    meta: dict[str, Any] | None = None

    @property
    def citation_label(self) -> str:
        if self.meta and "row_index" in self.meta:
            return f"row={self.meta['row_index']}"
        return f"doc={self.doc_id}"


def _coerce_docs(docs: Iterable[dict[str, Any]] | Iterable[RAGDoc]) -> list[RAGDoc]:
    out: list[RAGDoc] = []
    for d in docs:
        if isinstance(d, RAGDoc):
            out.append(d)
        else:
            out.append(
                RAGDoc(
                    doc_id=str(d.get("id", "")),
                    text=str(d.get("text", "")),
                    meta=d.get("meta") if isinstance(d.get("meta"), dict) else None,
                )
            )
    return out


def build_prompt(query: str, context: str) -> str:
    """Backward-compatible prompt helper.

    The notebook originally calls this with a raw context string.
    For structured RAG, prefer `build_rag_messages()`.
    """

    return f"""Answer the question using only the context.

Query:
{query}

Context:
{context}
"""


def build_rag_messages(
    query: str,
    docs: Iterable[dict[str, Any]] | Iterable[RAGDoc],
    *,
    max_chars_per_doc: int = 1200,
) -> list[dict[str, str]]:
    """Build chat-style messages for an instruct model.

    The model is instructed to only use the provided context and to cite sources.
    """

    rag_docs = _coerce_docs(docs)
    context_blocks: list[str] = []
    for i, d in enumerate(rag_docs, start=1):
        txt = (d.text or "").strip()
        if len(txt) > max_chars_per_doc:
            txt = txt[:max_chars_per_doc].rstrip() + " …"
        company = ""
        if d.meta and "company" in d.meta and d.meta["company"] is not None:
            company = f" company={d.meta['company']}"
        context_blocks.append(
            f"[{i}] ({d.citation_label}{company})\n{txt}"
        )
    context_str = "\n\n".join(context_blocks) if context_blocks else "(no context provided)"

    system = (
        "You are a careful assistant. Answer using ONLY the provided context snippets. "
        "If the context is insufficient, say you don't know. "
        "Cite sources inline using bracketed numbers like [1], [2]. "
        "Do NOT invent facts or citations."
    )
    user = (
        f"Question: {query}\n\n"
        f"Context snippets:\n{context_str}\n\n"
        "Return:\n"
        "1) A short answer\n"
        "2) Top 3 recommendations (if applicable)\n"
        "3) Bullet reasons with citations"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_llm(
    model_name: str,
    *,
    use_4bit: bool = True,
    trust_remote_code: bool = False,
):
    """Load a HuggingFace Transformers causal LM + tokenizer.

    Designed for Colab: uses `device_map='auto'` and optional 4-bit quantization.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )
        except Exception:
            quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def generate_answer(
    query: str,
    docs: Iterable[dict[str, Any]] | Iterable[RAGDoc],
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 350,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """Generate a grounded answer using provided docs as context."""

    import torch

    messages = build_rag_messages(query, docs)

    # Prefer chat templates when available (Qwen instruct models support this)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback to a simple prompt
        prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\n\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
        )

    # Strip the prompt tokens
    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def format_citations(docs: Iterable[dict[str, Any]] | Iterable[RAGDoc]) -> str:
    """Human-readable citation legend matching the [1], [2] numbering."""

    rag_docs = _coerce_docs(docs)
    lines: list[str] = []
    for i, d in enumerate(rag_docs, start=1):
        company = ""
        if d.meta and d.meta.get("company") is not None:
            company = f" company={d.meta['company']}"
        lines.append(f"[{i}] {d.citation_label}{company}")
    return "\n".join(lines)

