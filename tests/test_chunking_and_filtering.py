"""
test_chunking_and_filtering.py

Tests for chunk_text() parameter validation and for Retriever dropping
chunks that have nothing in common with the query.

Run with: python -m pytest tests/ (from repo root)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import build_index, chunk_text  # noqa: E402
from retriever import Retriever  # noqa: E402


# --- chunk_text -----------------------------------------------------------

def test_chunks_overlap_by_the_requested_amount():
    text = "abcdefghij" * 3  # 30 chars
    chunks = chunk_text(text, chunk_size=10, overlap=3)

    # Window advances 7 chars each time: starts at 0, 7, 14, 21, 28
    assert chunks[0] == text[0:10]
    assert chunks[1] == text[7:17]
    assert len(chunks) == 5


@pytest.mark.parametrize("overlap", [10, 15])
def test_overlap_not_smaller_than_chunk_size_is_rejected(overlap):
    # Before the fix, these made chunk_text loop forever.
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=overlap)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=-1)


def test_non_positive_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)


# --- Retriever min_score filtering ---------------------------------------

@pytest.fixture(scope="module")
def retriever():
    build_index()
    return Retriever()


def test_unrelated_query_returns_no_chunks(retriever):
    # Nonsense words that appear in no document: every chunk scores 0.
    assert retriever.retrieve("zxqv blorptastic wumbleflax", top_k=3) == []


def test_results_never_include_zero_scores(retriever):
    results = retriever.retrieve("retrieval augmented generation", top_k=50)

    assert results, "expected at least one relevant chunk"
    assert all(r["score"] > 0 for r in results)


def test_results_are_sorted_best_first(retriever):
    results = retriever.retrieve("retrieval augmented generation", top_k=5)
    scores = [r["score"] for r in results]

    assert scores == sorted(scores, reverse=True)
