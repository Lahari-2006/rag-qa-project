"""
Stage 4 (upgraded): Incrementally store embedded chunks in ChromaDB.

Key difference from the original version: instead of wiping and rebuilding
the whole collection every run, this checks which chunk IDs already exist
and only embeds + adds the NEW ones. This means adding a second, third,
tenth PDF to data/ never re-processes documents you already indexed.

Run with: python src/vector_store.py
"""

import os
import chromadb
from load_and_chunk import load_and_chunk_all, load_pdf_text, chunk_text
from embed_chunks import get_embeddings_batch

CHROMA_PATH = os.environ.get(
    "CHROMA_PATH", os.path.join(os.path.dirname(__file__), "..", "chroma_db")
)
COLLECTION_NAME = "pdf_docs"


def get_collection():
    """Open (or create) the persistent collection. Safe to call anytime."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_pdf_to_store(pdf_path: str) -> int:
    """
    Process a SINGLE PDF and add only its NEW chunks to the vector store.
    Returns the number of new chunks actually added (0 if already indexed).
    This is the function the upload endpoint will call.
    """
    filename = os.path.basename(pdf_path)
    text = load_pdf_text(pdf_path)

    if not text.strip():
        print(f"Skipped {filename}: no extractable text")
        return 0

    chunks = chunk_text(text)
    collection = get_collection()

    # Build candidate IDs and check which ones already exist in the store
    candidate_ids = [f"{filename}_{i}" for i in range(len(chunks))]
    existing = collection.get(ids=candidate_ids)
    already_indexed = set(existing["ids"])

    # Keep only chunks that are genuinely new
    new_texts, new_ids, new_metas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        if chunk_id not in already_indexed:
            new_texts.append(chunk)
            new_ids.append(chunk_id)
            new_metas.append({"source": filename, "chunk_id": i})

    if not new_texts:
        print(f"{filename}: already fully indexed, nothing to add")
        return 0

    embeddings = get_embeddings_batch(new_texts)
    collection.add(
        ids=new_ids,
        embeddings=embeddings,
        documents=new_texts,
        metadatas=new_metas
    )

    print(f"{filename}: added {len(new_texts)} new chunks "
          f"(skipped {len(chunks) - len(new_texts)} already indexed)")
    return len(new_texts)


def delete_document(filename: str) -> int:
    """
    Remove all chunks belonging to a given filename from the vector store,
    and delete the underlying PDF file from data/.
    Returns the number of chunks removed.
    """
    collection = get_collection()

    # Find out how many chunks this document actually has before deleting,
    # so we can report a meaningful count back to the caller.
    existing = collection.get(where={"source": filename})
    count = len(existing["ids"])

    if count == 0:
        return 0

    collection.delete(where={"source": filename})

    # Also remove the physical file from data/, if it's still there
    data_dir = os.environ.get(
        "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
    )
    data_path = os.path.join(data_dir, filename)
    if os.path.exists(data_path):
        os.remove(data_path)

    print(f"{filename}: removed {count} chunks from the vector store")
    return count


def build_vector_store_from_folder(data_dir: str = "data"):
    """
    Process every PDF in data_dir incrementally. Safe to re-run any time -
    already-indexed PDFs are skipped automatically, new ones get added.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", data_dir)
    total_added = 0

    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            total_added += add_pdf_to_store(full_path)

    collection = get_collection()
    print(f"\nDone. {total_added} new chunks added this run.")
    print(f"Collection now holds {collection.count()} chunks total.")


if __name__ == "__main__":
    build_vector_store_from_folder("data")