"""Tests for local embeddings and the Ollama-compatible chat client (no network)."""

from unittest.mock import MagicMock

from src.embed import embed_texts, get_openai_client


def test_embed_texts_uses_local_encoder_when_client_omitted(monkeypatch):
    fake = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr("src.embed._local_embed", fake, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    vectors = embed_texts(["hello", "world"], model="all-MiniLM-L6-v2")

    fake.assert_called_once_with(["hello", "world"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_texts_calls_openai_compatible_client_when_provided():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]

    vectors = embed_texts(
        ["hello", "world"], model="all-MiniLM-L6-v2", client=mock_client
    )

    mock_client.embeddings.create.assert_called_once_with(
        model="all-MiniLM-L6-v2", input=["hello", "world"]
    )
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_get_openai_client_targets_ollama_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = get_openai_client()
    assert "11434" in str(client.base_url)
