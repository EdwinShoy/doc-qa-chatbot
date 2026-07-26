# Doc Q&A Chatbot

A small retrieval-augmented generation (RAG) chatbot: ask questions in plain
English, it retrieves the most relevant chunks from a local set of documents
using TF-IDF similarity, then passes them as context to an LLM to generate a
grounded answer — instead of the model just hallucinating from training data.

Built as a portfolio project to demonstrate practical LLM application
engineering: retrieval, prompt construction, and API integration.

## How it works

```
question --> embed query (TF-IDF) --> rank doc chunks by cosine similarity
         --> take top-k chunks --> build prompt with context
         --> call LLM API --> grounded answer, with source chunks shown
```

No vector database or heavy ML dependencies required — TF-IDF + cosine
similarity via scikit-learn is enough for a small local document set and
keeps the project easy to run anywhere. Swapping in real embeddings
(sentence-transformers, OpenAI embeddings, etc.) is a natural next step —
see `docs/ARCHITECTURE.md`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
```

Supports either Anthropic (`ANTHROPIC_API_KEY`) or OpenAI (`OPENAI_API_KEY`)
— set whichever one you have and the client picks it automatically.

## Usage

Drop `.md` or `.txt` files into `docs/`, then:

```bash
python src/ingest.py       # builds the TF-IDF index from docs/
python src/chatbot.py      # interactive Q&A loop
```

Example:
```
> what is this project about?
Answer: This project is a retrieval-augmented Q&A chatbot that answers
questions using your own documents as grounding context...

Sources: docs/sample_doc.md (chunk 1)
```

## Project layout

```
doc-qa-chatbot/
├── docs/               # your source documents (markdown/text) go here
├── src/
│   ├── ingest.py       # chunks docs, builds TF-IDF index, saves to data/
│   ├── retriever.py    # loads index, ranks chunks by similarity to a query
│   ├── llm_client.py   # thin wrapper around Anthropic/OpenAI chat completion
│   └── chatbot.py       # ties retriever + llm_client together, CLI loop
├── data/               # generated index (gitignored)
└── tests/
    └── test_retriever.py
```

## Status

Scaffold — retrieval and CLI chat loop implemented. See open Issues for
next steps (real embeddings, simple web UI, evaluation set).
