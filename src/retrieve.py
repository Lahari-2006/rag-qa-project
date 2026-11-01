"""
Stage 5: Retrieve the most relevant chunks for a given question.

Run with: python src/retrieve.py
"""

import chromadb
import os
from embed_chunks import get_embedding

# Resolve chroma_db path relative to THIS file's location, not the current
# working directory - so it works whether you run this from project root,
# from src/, or via uvicorn.
CHROMA_PATH = os.environ.get(
    "CHROMA_PATH", os.path.join(os.path.dirname(__file__), "..", "chroma_db")
)


def retrieve_relevant_chunks(query: str, n_results: int = 3, source: str | None = None):
    # 1. Connect to the same persistent Chroma database we built in Stage 4
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(name="pdf_docs")
    except Exception:
        # No documents have been uploaded/indexed yet - return an empty
        # result in the same shape Chroma normally returns, instead of
        # letting a raw "collection not found" error bubble up.
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # 2. Embed the query using the SAME model used for the document chunks
    query_vector = get_embedding(query)

    # 3. Search the collection for the most similar chunks.
    # If `source` is given, restrict the search to chunks from that one
    # document only - this is what "per-document filtering" means in
    # practice: the `where` clause is Chroma's metadata filter.
    query_kwargs = {"query_embeddings": [query_vector], "n_results": n_results}
    if source:
        query_kwargs["where"] = {"source": source}

    results = collection.query(**query_kwargs)

    return results


def list_available_sources() -> list[str]:
    """Return the distinct document filenames currently indexed."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="pdf_docs")
    all_items = collection.get()
    sources = {m["source"] for m in all_items["metadatas"]}
    return sorted(sources)


if __name__ == "__main__":
    query = "What is this document about?"  # try changing this to test different questions
    results = retrieve_relevant_chunks(query)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"Query: {query}\n")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"--- Result {i+1} (distance: {dist:.4f}) ---")
        print(f"Source: {meta['source']}, Chunk: {meta['chunk_id']}")
        print(doc[:300])
        print()