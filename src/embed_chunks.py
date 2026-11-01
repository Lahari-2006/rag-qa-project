"""
Stage 3: Generate embeddings for text chunks using a FREE local model
via fastembed (ONNX Runtime based - no PyTorch, avoids Windows DLL issues).

Run with: python src/embed_chunks.py
"""

from fastembed import TextEmbedding
from load_and_chunk import load_and_chunk_all

# Downloads the model once (~small, under 100MB) and caches it locally.
# Every run after the first will be fast, no internet needed.
print("Loading embedding model (first run may take a minute)...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def get_embedding(text: str) -> list[float]:
    """Get the embedding vector for a single piece of text."""
    # model.embed() returns a generator of numpy arrays
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed many texts at once - fastembed handles batching internally.
    """
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


if __name__ == "__main__":
    chunks = load_and_chunk_all("data")
    print(f"Loaded {len(chunks)} chunks")

    # Test on just the first chunk first
    sample_text = chunks[0]["text"]
    vector = get_embedding(sample_text)
    print(f"\nSingle embedding length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

    # Now test embedding ALL chunks in a batch (this is what Stage 4 will use)
    print(f"\nEmbedding all {len(chunks)} chunks...")
    all_texts = [c["text"] for c in chunks]
    all_vectors = get_embeddings_batch(all_texts)
    print(f"Done. Generated {len(all_vectors)} embeddings.")