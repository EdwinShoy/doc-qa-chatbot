"""
ingest.py

Reads every .md/.txt file in docs/, splits each into overlapping chunks,
fits a TF-IDF vectorizer over all chunks, and saves the vectorizer + chunk
text + chunk metadata to data/ so retriever.py can load it without
re-reading the documents each time.

Usage:
    python ingest.py
"""

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Each step moves the window forward by (chunk_size - overlap). If that's
    # zero or negative, `start` never advances and the loop below runs forever.
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        raise ValueError(
            f"overlap must be >= 0 and smaller than chunk_size ({chunk_size}), got {overlap}"
        )

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def load_documents() -> list[dict]:
    """Returns a list of {source, chunk_index, text} dicts for every chunk."""
    records = []
    doc_paths = sorted(list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.txt")))

    if not doc_paths:
        raise FileNotFoundError(f"No .md/.txt files found in {DOCS_DIR}. Add some documents first.")

    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            records.append({"source": path.name, "chunk_index": i, "text": chunk})

    return records


def build_index():
    records = load_documents()
    texts = [r["text"] for r in records]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    DATA_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(DATA_DIR / "matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)

    with open(DATA_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Indexed {len(records)} chunks from {len(set(r['source'] for r in records))} document(s).")
    print(f"Saved index to {DATA_DIR}/")


if __name__ == "__main__":
    build_index()
