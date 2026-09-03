"""Phase 3 — Embed: turn text into vectors with an OpenAI embedding model.

Nearest-neighbor search needs numbers, not strings. An embedding model maps
each chunk (and later the question) into the same vector space so similar
meanings land near each other.

If you skip embeddings, Chroma has nothing to compare. If you embed documents
with one model and the question with another, distances are meaningless.
"""

from __future__ import annotations

import os

from openai import OpenAI

MISSING_KEY_MESSAGE = (
    "OPENAI_API_KEY is missing. Copy .env.example to .env and add your key."
)


def require_api_key() -> str:
    """Return OPENAI_API_KEY or exit with a how-to.

    Args:
        none — reads the process environment.

    Returns:
        The API key string.

    Pipeline role: both ingest (embed chunks) and ask (embed the question)
    need a key before any OpenAI call. cli.py loads .env; this function
    only checks what is already in the environment.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit(MISSING_KEY_MESSAGE)
    return key


def get_openai_client() -> OpenAI:
    """Build the official OpenAI client using the env key.

    Args:
        none.

    Returns:
        An OpenAI client.

    Pipeline role: production ingest/ask call this; tests pass a mock instead
    so CI never hits the network.
    """
    return OpenAI(api_key=require_api_key())


def embed_texts(texts: list[str], *, model: str, client=None) -> list[list[float]]:
    """Embed a batch of strings with one API call.

    Args:
        texts: Chunk strings (ingest) or a one-item list with the question (ask).
        model: Must match the model used when the collection was built.
        client: Optional OpenAI-like client; created from the env key if omitted.

    Returns:
        One embedding vector per input string, in the same order.

    Pipeline role: ingest stores these vectors in Chroma; ask embeds the
    question with the same model so cosine search can rank chunks.
    """
    if not texts:
        return []
    if client is None:
        client = get_openai_client()
    # One batched call is cheaper and keeps chunk order aligned with vectors.
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
