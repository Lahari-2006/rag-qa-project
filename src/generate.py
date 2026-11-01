"""
Stage 6: Generate an answer using retrieved chunks as context (RAG).
Uses Groq's OpenAI-compatible chat completions API.

Run with: python src/generate.py
"""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from retrieve import retrieve_relevant_chunks

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def build_prompt(query: str, chunks: list[str]) -> str:
    """
    Combine retrieved chunks + the user's question into one prompt.
    This is the 'grounding' step that keeps the LLM honest.
    """
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant answering questions using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Do not make up information that isn't in the context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    return prompt


def answer_question(query: str, n_results: int = 3, source: str | None = None) -> tuple[str, list[str]]:
    query = query.strip()
    if not query:
        raise ValueError("Question cannot be empty.")

    # 1. Retrieve relevant chunks (Stage 5), optionally scoped to one document
    results = retrieve_relevant_chunks(query, n_results=n_results, source=source)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not chunks:
        return (
            "I don't have any indexed documents to search yet. "
            "Please upload a PDF first.",
            []
        )

    # 2. Build the grounded prompt
    prompt = build_prompt(query, chunks)

    # 3. Call Groq's OpenAI-compatible API to generate the answer
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY must be set to generate answers.")

    request = Request(
        API_URL,
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            completion = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Groq API returned HTTP {error.code}: {detail}"
        ) from error

    answer = completion["choices"][0]["message"]["content"]

    # 4. Return answer + sources so the user can verify
    sources = [f"{m['source']} (chunk {m['chunk_id']})" for m in metadatas]
    return answer, sources


if __name__ == "__main__":
    query = input("Ask a question about your document: ")

    answer, sources = answer_question(query)

    print(f"\nQuestion: {query}\n")
    print(f"Answer:\n{answer}\n")
    print("Sources used:")
    for s in sources:
        print(f"  - {s}")