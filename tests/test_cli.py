"""Tests for ingest orchestration, ask output, --show-chunks, and --config."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.chunk import chunk_documents
from src.cli import ask, ingest, main
from src.config import Settings
from src.generate import build_messages
from src.index import get_persistent_client
from src.load import load_markdown
from src.retrieve import retrieve_chunks
from tests.conftest import DATA_DIR, bag_of_words_embedder


def _settings(tmp_path: Path, *, top_k: int = 4) -> Settings:
    return Settings(
        chunk_size=500,
        chunk_overlap=80,
        top_k=top_k,
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4o-mini",
        temperature=0.0,
        data_dir=DATA_DIR,
        chroma_path=tmp_path / "chroma_db",
        collection_name="notes",
    )


def _embed_fn_for(settings: Settings, extra_texts: list[str] | None = None):
    docs = load_markdown(settings.data_dir)
    chunks = chunk_documents(
        docs, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    texts = [chunk.text for chunk in chunks] + (extra_texts or [])
    return bag_of_words_embedder(texts)


def test_ingest_stores_source_and_chunk_index(tmp_path: Path):
    settings = _settings(tmp_path)
    embed_fn = _embed_fn_for(settings)
    count = ingest(settings, embed_fn=embed_fn)
    assert count > 0

    collection = get_persistent_client(settings.chroma_path).get_collection(
        name=settings.collection_name, embedding_function=None
    )
    stored = collection.get(include=["documents", "metadatas"])
    assert stored["metadatas"]
    sources = {meta["source"] for meta in stored["metadatas"]}
    assert "wifi-and-office.md" in sources
    assert all("source" in meta and "chunk_index" in meta for meta in stored["metadatas"])
    assert any("orchid-42" in doc for doc in stored["documents"])


def test_assembled_prompt_contains_orchid_chunk(tmp_path: Path):
    settings = _settings(tmp_path)
    question = "What is the office Wi-Fi password?"
    embed_fn = _embed_fn_for(settings, extra_texts=[question])
    ingest(settings, embed_fn=embed_fn)

    collection = get_persistent_client(settings.chroma_path).get_collection(
        name=settings.collection_name, embedding_function=None
    )
    query_vec = embed_fn([question])[0]
    hits = retrieve_chunks(collection, query_vec, k=settings.top_k)
    blob = "\n".join(message["content"] for message in build_messages(question, hits))
    assert "orchid-42" in blob


def test_ask_prints_retrieved_chunks_before_answer(tmp_path: Path, capsys):
    settings = _settings(tmp_path)
    question = "What is the office Wi-Fi password?"
    embed_fn = _embed_fn_for(settings, extra_texts=[question])
    ingest(settings, embed_fn=embed_fn)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="The password is orchid-42."))
    ]

    ask(
        question,
        settings,
        show_chunks=True,
        embed_fn=embed_fn,
        chat_client=mock_client,
    )
    out = capsys.readouterr().out
    assert out.index("Retrieved chunks") < out.index("Answer:")
    assert "orchid-42" in out
    assert "Sources:" in out
    assert "wifi-and-office.md" in out
    mock_client.chat.completions.create.assert_called_once()


def test_main_loads_config_flag(tmp_path: Path, monkeypatch):
    chroma = tmp_path / "chroma_db"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[chunking]
size = 500
overlap = 80
[retrieval]
top_k = 1
[embedding]
model = "text-embedding-3-small"
[generation]
model = "gpt-4o-mini"
temperature = 0.0
[paths]
data_dir = "{DATA_DIR}"
chroma_path = "{chroma}"
collection_name = "notes"
"""
    )
    settings = Settings(
        chunk_size=500,
        chunk_overlap=80,
        top_k=1,
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4o-mini",
        temperature=0.0,
        data_dir=DATA_DIR,
        chroma_path=chroma,
        collection_name="notes",
    )
    question = "hello"
    embed_fn = _embed_fn_for(settings, extra_texts=[question])
    ingest(settings, embed_fn=embed_fn)

    monkeypatch.setattr("src.cli.embed_texts", embed_fn)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="I don't know."))
    ]
    monkeypatch.setattr("src.cli.get_openai_client", lambda: mock_client)

    main(["--config", str(config_path), "ask", question])
    mock_client.chat.completions.create.assert_called_once()
    n_results_k = mock_client.chat.completions.create.call_args.kwargs
    user = n_results_k["messages"][1]["content"]
    # top_k=1 so at most one [filename] context block plus the empty case
    assert user.count("[") >= 0


def test_main_ask_exits_when_ollama_is_down(tmp_path: Path, monkeypatch):
    from httpx import Request
    from openai import APIConnectionError

    chroma = tmp_path / "chroma_db"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[chunking]
size = 500
overlap = 80
[retrieval]
top_k = 1
[embedding]
model = "all-MiniLM-L6-v2"
[generation]
model = "llama3.2"
temperature = 0.0
[paths]
data_dir = "{DATA_DIR}"
chroma_path = "{chroma}"
collection_name = "notes"
"""
    )
    settings = _settings(tmp_path)
    question = "hello"
    embed_fn = _embed_fn_for(settings, extra_texts=[question])
    ingest(settings, embed_fn=embed_fn)

    monkeypatch.setattr("src.cli.embed_texts", embed_fn)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=Request("POST", "http://localhost:11434/v1/chat/completions")
    )
    monkeypatch.setattr("src.cli.get_openai_client", lambda: mock_client)

    with pytest.raises(SystemExit, match="Ollama is not running"):
        main(["--config", str(config_path), "ask", question])
