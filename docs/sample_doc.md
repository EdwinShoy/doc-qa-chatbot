# About This Project

This project is a retrieval-augmented generation (RAG) chatbot. It answers
questions using your own documents as grounding context, rather than relying
purely on what a language model learned during training.

## Why RAG matters

Large language models are trained on a fixed snapshot of data and don't know
about your private documents, recent changes, or internal knowledge. RAG
solves this by retrieving relevant passages at query time and feeding them
to the model as context, so answers are grounded in real source material
instead of guesses.

## How retrieval works here

Documents are split into overlapping chunks. Each chunk is converted into a
TF-IDF vector — a representation based on how important each word is to that
chunk relative to the whole document set. When a question comes in, it's
converted to a TF-IDF vector the same way, and compared against every chunk
using cosine similarity. The top-scoring chunks are passed to the LLM as
context.

## Limitations of TF-IDF retrieval

TF-IDF is a keyword-based method — it doesn't understand synonyms or deeper
semantic meaning the way neural embeddings do. A question like "cost" won't
strongly match a document that only says "price" unless both words appear.
Swapping in sentence-transformer embeddings or an embeddings API would
improve retrieval quality at the cost of extra dependencies and (for API
based embeddings) a per-call cost.
