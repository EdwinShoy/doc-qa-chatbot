# Architecture

## Pipeline

1. `ingest.py` reads all `.md`/`.txt` files in `docs/`, splits each into
   ~800-character overlapping chunks (150-char overlap so context isn't
   lost at chunk boundaries), fits a TF-IDF vectorizer over all chunks, and
   saves the vectorizer + matrix + chunk metadata to `data/`.
2. `retriever.py` loads that saved index and, given a query, vectorizes it
   with the same fitted TF-IDF vectorizer and ranks all chunks by cosine
   similarity.
3. `llm_client.py` takes the top-k chunks plus the question, builds a
   prompt that instructs the model to answer only from the given context,
   and calls whichever LLM API has a key configured.
4. `chatbot.py` wires the above into an interactive CLI loop.

## Why TF-IDF instead of neural embeddings

Neural embeddings (sentence-transformers, OpenAI embeddings, etc.) capture
semantic meaning better — "cost" and "price" would be recognized as related.
TF-IDF is purely keyword/frequency-based. It was chosen here to keep the
scaffold dependency-light and runnable anywhere without a GPU or extra API
calls for indexing. See open Issues for upgrading retrieval quality.

## Design decisions / open questions

- [ ] Swap TF-IDF for real embeddings (sentence-transformers locally, or an
      embeddings API) for better semantic matching.
- [ ] Add a minimal web UI (Flask/FastAPI + a simple HTML page) instead of
      CLI-only.
- [ ] Build a small evaluation set (question -> expected source doc) to
      measure retrieval quality objectively instead of eyeballing it.
- [ ] Handle documents larger than memory / a real document count (current
      approach loads everything into memory, fine for a portfolio demo,
      not for a large corpus).
