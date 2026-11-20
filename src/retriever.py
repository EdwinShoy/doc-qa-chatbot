"""
retriever.py

Loads the TF-IDF index built by ingest.py and ranks document chunks by
cosine similarity to a query.
"""

import json
import pickle
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Retriever:
    def __init__(self):
        vectorizer_path = DATA_DIR / "vectorizer.pkl"
        matrix_path = DATA_DIR / "matrix.pkl"
        chunks_path = DATA_DIR / "chunks.json"

        if not (vectorizer_path.exists() and matrix_path.exists() and chunks_path.exists()):
            raise FileNotFoundError(
                "Index not found. Run `python src/ingest.py` first to build it from docs/."
            )

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(matrix_path, "rb") as f:
            self.matrix = pickle.load(f)
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[dict]:
        """Returns up to top_k chunks most similar to the query, each with a score.

        Chunks scoring <= min_score are dropped. With the default of 0.0 that
        removes chunks sharing no terms with the query at all, so the LLM is
        never handed (and never cites) a "source" that's completely unrelated.
        May return fewer than top_k results, or an empty list.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] <= min_score:
                break  # ranked high-to-low, so everything after this is lower too
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results
