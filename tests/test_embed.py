"""Tests for the OpenAI embeddings wrapper (mocked — no network)."""

from unittest.mock import MagicMock

import pytest

from src.embed import embed_texts, require_api_key


def test_require_api_key_exits_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is missing"):
        require_api_key()


def test_embed_texts_calls_openai_with_model_and_input():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]

    vectors = embed_texts(
        ["hello", "world"], model="text-embedding-3-small", client=mock_client
    )

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["hello", "world"]
    )
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
