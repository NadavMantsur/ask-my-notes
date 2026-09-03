"""Tests for Chroma collection reset and chunk upsert."""

from src.chunk import Chunk
from src.index import get_persistent_client, reset_collection, upsert_chunks


def _two_chunks():
    return [
        Chunk(text="the office Wi-Fi password is orchid-42", source="wifi-and-office.md", chunk_index=0),
        Chunk(text="standup is at 10:00", source="team-rituals.md", chunk_index=0),
    ]


def test_upsert_stores_documents_and_metadata(tmp_path):
    client = get_persistent_client(tmp_path / "chroma")
    collection = reset_collection(client, "notes")
    chunks = _two_chunks()
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    upsert_chunks(collection, chunks, embeddings)

    assert collection.count() == 2
    got = collection.get(ids=["wifi-and-office.md::0"], include=["documents", "metadatas"])
    assert got["documents"][0] == chunks[0].text
    assert got["metadatas"][0]["source"] == "wifi-and-office.md"
    assert got["metadatas"][0]["chunk_index"] == 0


def test_reset_collection_drops_old_ids(tmp_path):
    client = get_persistent_client(tmp_path / "chroma")
    collection = reset_collection(client, "notes")
    upsert_chunks(
        collection,
        [Chunk(text="stale", source="old.md", chunk_index=0)],
        [[1.0, 0.0, 0.0]],
    )
    assert collection.count() == 1

    collection = reset_collection(client, "notes")
    assert collection.count() == 0
