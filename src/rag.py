
from src.llm import generate_response

def build_context(results: list) -> str:
    context_parts = []
    for i, r in enumerate(results):
        meta = r["meta"]
        snippet = f"Document [{i+1}]
Company: {meta.get('company', 'Unknown')}"
        if "salary_median" in meta and meta["salary_median"]:
            snippet += f"
Median Salary: ${meta['salary_median']:,.0f}"
        if "rating" in meta:
            snippet += f"
Rating: {meta['rating']}/5.0"
        snippet += f"
Description: {r['text'][:500]}..." # Truncate to save tokens
        context_parts.append(snippet)
    return "

".join(context_parts)

def ask_rag(query: str, results: list, model, tokenizer) -> str:
    context_str = build_context(results)
    
    prompt = f"""You are an expert AI Job Recruiter. Use the retrieved documents below to answer the candidate's query.
    
Query: "{query}"

Retrieved Documents:
{context_str}

Instructions:
1. Recommend the best options from the documents.
2. Explicitly cite the Document number [x] for every claim.
3. If the documents don't help, admit it.
4. Be concise and professional.

Answer:"""

    return generate_response(model, tokenizer, prompt)
