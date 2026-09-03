"""Tests for vector top-k retrieval, including the orchid-42 Wi-Fi case."""

from pathlib import Path

from src.chunk import Chunk, chunk_documents
from src.index import get_persistent_client, reset_collection, upsert_chunks
from src.load import load_markdown
from src.retrieve import retrieve_chunks
from tests.conftest import DATA_DIR, bag_of_words_embedder


def test_k_is_honored_with_hand_chosen_vectors(tmp_path: Path):
    client = get_persistent_client(tmp_path / "chroma")
    collection = reset_collection(client, "notes")
    chunks = [
        Chunk(text="alpha", source="a.md", chunk_index=0),
        Chunk(text="beta", source="b.md", chunk_index=0),
        Chunk(text="gamma", source="c.md", chunk_index=0),
    ]
    upsert_chunks(
        collection,
        chunks,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    one = retrieve_chunks(collection, [1.0, 0.0, 0.0], k=1)
    assert len(one) == 1
    assert one[0].text == "alpha"

    two = retrieve_chunks(collection, [1.0, 0.0, 0.0], k=2)
    assert len(two) == 2
    assert two[0].text == "alpha"


def test_wifi_password_query_returns_orchid_chunk(tmp_path: Path):
    documents = load_markdown(DATA_DIR)
    chunks = chunk_documents(documents, chunk_size=500, overlap=80)
    query = "What is the office Wi-Fi password?"
    embed_fn = bag_of_words_embedder([chunk.text for chunk in chunks] + [query])
    vectors = embed_fn([chunk.text for chunk in chunks])

    client = get_persistent_client(tmp_path / "chroma")
    collection = reset_collection(client, "notes")
    upsert_chunks(collection, chunks, vectors)

    query_vec = embed_fn([query])[0]
    hits = retrieve_chunks(collection, query_vec, k=4)
    assert any("orchid-42" in hit.text for hit in hits)
