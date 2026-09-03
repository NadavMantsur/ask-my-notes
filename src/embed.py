"""Phase 3 — Embed: turn text into vectors with a local ONNX model.

Nearest-neighbor search needs numbers, not strings. An embedding model maps
each chunk (and later the question) into the same vector space so similar
meanings land near each other.

If you skip embeddings, Chroma has nothing to compare. If you embed documents
with one model and the question with another, distances are meaningless.
"""

from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MISSING_MESSAGE = (
    "Ollama is not running. Install it from https://ollama.com, then run: "
    "ollama pull llama3.2:1b"
)

_onnx_model = None


def _local_embed(texts: list[str]) -> list[list[float]]:
    """Embed strings with Chroma's bundled all-MiniLM-L6-v2 ONNX model.

    Args:
        texts: Chunk strings (ingest) or a one-item list with the question (ask).

    Returns:
        One embedding vector per input string, in the same order.
    """
    global _onnx_model
    if _onnx_model is None:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )

        _onnx_model = ONNXMiniLM_L6_V2()
    vectors = _onnx_model(texts)
    return [list(map(float, vec)) for vec in vectors]


def get_openai_client() -> OpenAI:
    """Build an OpenAI-compatible client pointed at local Ollama.

    Args:
        none.

    Returns:
        An OpenAI client whose base_url is Ollama.

    Pipeline role: production ask() uses this for chat. Tests pass a mock
    instead so CI never needs Ollama running. Ollama ignores the API key
    but the SDK requires a non-empty string.
    """
    url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
    if not url:
        url = DEFAULT_OLLAMA_BASE_URL
    key = os.getenv("OPENAI_API_KEY", "").strip() or "ollama"
    return OpenAI(api_key=key, base_url=url)


def embed_texts(texts: list[str], *, model: str, client=None) -> list[list[float]]:
    """Embed a batch of strings locally (or via an injected OpenAI-like client).

    Args:
        texts: Chunk strings (ingest) or a one-item list with the question (ask).
        model: Must match the model used when the collection was built.
        client: Optional OpenAI-like client; local ONNX is used if omitted.

    Returns:
        One embedding vector per input string, in the same order.

    Pipeline role: ingest stores these vectors in Chroma; ask embeds the
    question with the same model so cosine search can rank chunks.
    """
    if not texts:
        return []
    if client is not None:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
    return _local_embed(texts)
