def build_prompt(query, context):
    return f"""Answer the question using only the context.

Query:
{query}

Context:
{context}
"""

