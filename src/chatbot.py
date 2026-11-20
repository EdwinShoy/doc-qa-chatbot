"""
chatbot.py

Interactive CLI: takes a question, retrieves relevant chunks from the local
TF-IDF index, and asks the configured LLM to answer using that context.

Usage:
    python src/chatbot.py
"""

from llm_client import ask
from retriever import Retriever


def main():
    try:
        retriever = Retriever()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Doc Q&A Chatbot — ask a question about the documents in docs/. Ctrl+C to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue

        chunks = retriever.retrieve(question, top_k=3)

        if not chunks:
            print("Answer: I couldn't find anything relevant in the documents for that.\n")
            continue

        try:
            answer = ask(question, chunks)
        except RuntimeError as e:
            print(f"Error: {e}")
            continue

        sources = ", ".join(f"{c['source']} (chunk {c['chunk_index']})" for c in chunks)
        print(f"\nAnswer: {answer}\n\nSources: {sources}\n")


if __name__ == "__main__":
    main()
