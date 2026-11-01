"""
Stage 2: Load PDFs and split them into overlapping text chunks.
Run with: python src/load_and_chunk.py
"""

import os
from pypdf import PdfReader


def load_pdf_text(pdf_path: str) -> str:
    """Extract all text from a single PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # some pages (scanned/blank) may return None
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks.

    chunk_size: number of characters per chunk
    overlap: number of characters shared between consecutive chunks
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += step

    return chunks


def load_and_chunk_all(data_dir: str = "data") -> list[dict]:
    """
    Process every PDF in data_dir.
    Returns a list of dicts: {"source": filename, "chunk_id": int, "text": chunk}
    This structure is what we'll feed into embeddings in Stage 3.
    """
    all_chunks = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(data_dir, filename)
        text = load_pdf_text(path)

        if not text.strip():
            print(f"⚠️  {filename}: no extractable text (might be scanned/image PDF)")
            continue

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source": filename,
                "chunk_id": idx,
                "text": chunk
            })

        print(f"✅ {filename}: {len(chunks)} chunks created")

    return all_chunks


if __name__ == "__main__":
    all_chunks = load_and_chunk_all("data")

    print(f"\nTotal chunks across all PDFs: {len(all_chunks)}")

    if all_chunks:
        print("\n--- Preview of first chunk ---")
        print(f"Source: {all_chunks[0]['source']}")
        print(f"Chunk ID: {all_chunks[0]['chunk_id']}")
        print(all_chunks[0]["text"][:300])