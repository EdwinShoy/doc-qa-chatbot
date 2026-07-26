"""
test_retriever.py

Basic sanity test: build an index from the sample doc and confirm a query
about the project's core topic (RAG) retrieves a relevant chunk.

Run manually with: python -m pytest tests/ (from repo root, after `pip install pytest`)
or just: python tests/test_retriever.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import build_index  # noqa: E402
from retriever import Retriever  # noqa: E402


def test_retrieval_finds_relevant_chunk():
    build_index()
    retriever = Retriever()

    results = retriever.retrieve("what is retrieval augmented generation", top_k=1)

    assert len(results) == 1
    assert results[0]["score"] > 0
    print("PASS: retrieval returned a relevant, non-zero-score chunk")


if __name__ == "__main__":
    test_retrieval_finds_relevant_chunk()
