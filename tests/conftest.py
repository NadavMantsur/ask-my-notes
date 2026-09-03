"""Shared fixtures: data dir and a tiny bag-of-words embedder for tests.

Tests must never call OpenAI. This fake embedder is lexical (term counts),
which is good enough to rank a Wi-Fi password question near the orchid-42
chunk without downloading a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def bag_of_words_embedder(texts: Sequence[str], *, vocab: list[str] | None = None):
    """Return a function that embeds by normalized term frequency over a vocab.

    Built once from the texts you plan to index plus any query strings you
    will search with, so the vector width stays fixed.
    """

    tokenized = [_tokens(text) for text in texts]
    if vocab is None:
        vocab = sorted({token for row in tokenized for token in row})

    def embed_fn(batch: list[str], *, model: str = "fake", client=None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in batch:
            counts = {token: 0.0 for token in vocab}
            for token in _tokens(text):
                if token in counts:
                    counts[token] += 1.0
            raw = [counts[token] for token in vocab]
            norm = sum(value * value for value in raw) ** 0.5
            vectors.append([value / norm for value in raw] if norm else raw)
        return vectors

    embed_fn.vocab = vocab  # type: ignore[attr-defined]
    return embed_fn


def _tokens(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR
