from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineArtifacts:
    docs: list[dict[str, Any]]
    index: Any


def _default_doc_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("description", "reviews", "review", "summary"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    if parts:
        return "\n".join(parts)
    return "\n".join([f"{k}: {v}" for k, v in row.items() if isinstance(v, (str, int, float))])


def build_docs_from_dataframe(df) -> list[dict[str, Any]]:
    """Create retrieval docs from a pandas DataFrame.

    This intentionally keeps parsing minimal. The notebook can do richer cleaning.
    """

    cols = [c.strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = cols

    name_col = "name" if "name" in df.columns else df.columns[0]

    docs: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        row_dict = dict(row)
        meta: dict[str, Any] = {"row_index": int(i), "company": str(row_dict.get(name_col, i))}
        for c in (
            "rating",
            "happiness",
            "ceo_approval",
            "salary_median",
            "salary_min",
            "salary_max",
            "roles",
            "locations",
        ):
            if c in row_dict:
                meta[c] = row_dict.get(c)

        text = row_dict.get("doc_text")
        if not isinstance(text, str) or not text.strip():
            text = _default_doc_text(row_dict)

        docs.append({"id": str(i), "text": str(text), "meta": meta})
    return docs


def load_or_build_artifacts(
    *,
    csv_path: str,
    artifact_dir: str,
    force_rebuild: bool = False,
) -> PipelineArtifacts:
    from src.data import load_data
    from src.embeddings import embed_texts
    from src.index import build_index
    from src.config import EMBED_MODEL

    import numpy as np

    try:
        import faiss  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FAISS is required for the UI. Install faiss-cpu (or faiss-gpu on Colab)."
        ) from e

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    docs_path = artifact_path / "docs.json"
    emb_path = artifact_path / "embeddings.npy"
    index_path = artifact_path / "faiss.index"

    docs: list[dict[str, Any]] | None = None
    if docs_path.exists() and not force_rebuild:
        docs = json.loads(docs_path.read_text())

    if docs is None:
        df = load_data(csv_path)
        docs = build_docs_from_dataframe(df)
        docs_path.write_text(json.dumps(docs, ensure_ascii=False))

    if emb_path.exists() and not force_rebuild:
        emb = np.load(str(emb_path))
    else:
        texts = [d.get("text", "") for d in docs]
        emb = embed_texts(texts, EMBED_MODEL)
        np.save(str(emb_path), emb)

    if index_path.exists() and not force_rebuild:
        index = faiss.read_index(str(index_path))
    else:
        index = build_index(emb)
        faiss.write_index(index, str(index_path))

    return PipelineArtifacts(docs=docs, index=index)


def build_app(
    *,
    csv_path: str = "data/company_reviews.csv",
    artifact_dir: str | None = None,
):
    import gradio as gr

    from src.config import ARTIFACT_DIR, LLM_MODEL
    from src.explain import build_explanation_payload
    from src.ranker import hybrid_score
    from src.rag import ask_rag
    from src.retrieval import search
    from src.llm import load_qwen_model

    artifact_dir = artifact_dir or ARTIFACT_DIR

    artifacts_state = gr.State(value=None)
    model_state = gr.State(value=None)
    tokenizer_state = gr.State(value=None)

    def _debug_print_exception(context: str, exc: BaseException) -> None:
        exc_type = type(exc)
        print(f"[UI ERROR] {context}: {exc_type.__module__}.{exc_type.__name__}: {exc}")
        print(traceback.format_exc())

    def _ensure_artifacts(force_rebuild: bool = False):
        return load_or_build_artifacts(
            csv_path=csv_path,
            artifact_dir=artifact_dir,
            force_rebuild=force_rebuild,
        )

    def prepare(force_rebuild: bool):
        try:
            arts = _ensure_artifacts(force_rebuild=force_rebuild)
            return arts, f"Ready: {len(arts.docs)} docs | artifacts in {artifact_dir}"
        except Exception as e:
            _debug_print_exception("prepare", e)
            return None, f"Error ({type(e).__name__}): {e}"

    def load_model():
        try:
            model, tokenizer = load_qwen_model(LLM_MODEL)
            return model, tokenizer, f"Loaded model: {LLM_MODEL}"
        except Exception as e:
            _debug_print_exception("load_model", e)
            return None, None, f"Error ({type(e).__name__}): {e}"

    def run(
        query: str,
        top_k: int,
        salary_w: float,
        rating_w: float,
        min_rating: float,
        location: str,
        do_rag: bool,
        arts: PipelineArtifacts | None,
        model,
        tokenizer,
    ):
        try:
            if arts is None:
                arts = _ensure_artifacts(force_rebuild=False)

            filters: dict[str, Any] = {}
            if min_rating and min_rating > 0:
                filters["min_rating"] = float(min_rating)
            if location and location.strip():
                filters["location"] = location.strip()

            results = search(query, arts.index, arts.docs, top_k=int(top_k), filters=filters or None)
            priorities = {"salary": float(salary_w), "rating": float(rating_w)}
            ranked = hybrid_score(results, priorities)

            table = []
            for r in ranked:
                meta = r.get("meta") or {}
                table.append(
                    {
                        "company": meta.get("company", "Unknown"),
                        "score": round(float(r.get("score", 0.0)), 4),
                        "hybrid_score": round(float(r.get("hybrid_score", r.get("score", 0.0))), 4),
                        "rating": meta.get("rating"),
                        "salary_median": meta.get("salary_median"),
                        "snippet": (r.get("text") or "")[:200],
                    }
                )

            answer = ""
            if do_rag:
                if model is None or tokenizer is None:
                    answer = "LLM not loaded. Click 'Load Qwen model' first."
                else:
                    answer = ask_rag(query, ranked[:3], model, tokenizer)

            payload = build_explanation_payload(
                query=query,
                results=ranked,
                priorities=priorities,
                alpha=0.7,
                filters=filters or {},
            )
            explain_json = json.dumps(payload, ensure_ascii=False, indent=2)
            return table, answer, explain_json, arts
        except Exception as e:
            _debug_print_exception("run", e)
            err = {"error_type": type(e).__name__, "error_message": str(e)}
            return [], f"Error ({type(e).__name__}): {e}", json.dumps(err, ensure_ascii=False, indent=2), arts

    with gr.Blocks() as demo:
        gr.Markdown("# Pyramyd: AI Product Discovery (RAG)")

        with gr.Row():
            prep_btn = gr.Button("Prepare data/index")
            rebuild_chk = gr.Checkbox(label="Force rebuild artifacts", value=False)
            status = gr.Textbox(label="Status", value="Not prepared", interactive=False)

        with gr.Row():
            load_btn = gr.Button("Load Qwen model")
            model_status = gr.Textbox(label="Model", value="Not loaded", interactive=False)

        query = gr.Textbox(label="Query", placeholder="e.g., sharp eyes, good culture, high pay")

        with gr.Row():
            top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
            min_rating = gr.Slider(0, 5, value=0, step=0.5, label="Min rating")
            location = gr.Textbox(label="Location filter (optional)")

        with gr.Row():
            salary_w = gr.Slider(0, 1, value=0.5, step=0.05, label="Salary weight")
            rating_w = gr.Slider(0, 1, value=0.1, step=0.05, label="Rating weight")
            do_rag = gr.Checkbox(label="Generate grounded answer (RAG)", value=False)

        run_btn = gr.Button("Search")
        results_table = gr.Dataframe(label="Results", interactive=False)
        answer = gr.Textbox(label="Answer", lines=10)
        explain_json = gr.Textbox(label="Explanation JSON (deterministic)", lines=12)

        prep_btn.click(prepare, inputs=[rebuild_chk], outputs=[artifacts_state, status])
        load_btn.click(load_model, inputs=[], outputs=[model_state, tokenizer_state, model_status])
        run_btn.click(
            run,
            inputs=[
                query,
                top_k,
                salary_w,
                rating_w,
                min_rating,
                location,
                do_rag,
                artifacts_state,
                model_state,
                tokenizer_state,
            ],
            outputs=[results_table, answer, explain_json, artifacts_state],
        )

    return demo


