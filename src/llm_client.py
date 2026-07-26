"""
llm_client.py

Thin wrapper around the Anthropic or OpenAI chat completion APIs. Picks
whichever provider has an API key set in the environment (ANTHROPIC_API_KEY
takes priority over OPENAI_API_KEY if both are present).
"""

import os

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer isn't in the context, say you don't "
    "know rather than guessing. Be concise."
)


def _build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']} chunk {c['chunk_index']}]\n{c['text']}" for c in context_chunks
    )
    return f"Context:\n{context_text}\n\nQuestion: {question}"


def _ask_anthropic(question: str, context_chunks: list[dict], api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(question, context_chunks)}],
    )
    return response.content[0].text


def _ask_openai(question: str, context_chunks: list[dict], api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, context_chunks)},
        ],
    )
    return response.choices[0].message.content


def ask(question: str, context_chunks: list[dict]) -> str:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        return _ask_anthropic(question, context_chunks, anthropic_key)
    elif openai_key:
        return _ask_openai(question, context_chunks, openai_key)
    else:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file "
            "(copy .env.example to .env first)."
        )
