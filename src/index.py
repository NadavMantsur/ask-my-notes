"""Phase 4 — Store: persist chunk text, vectors, and metadata in Chroma.

A vector store is a library that can answer “which stored vectors are closest
to this query vector?” We use Chroma on disk so ingest and ask can be separate
commands. Metadata (source, chunk_index) rides along so retrieval can cite files.

If you skip this phase you would re-embed every note on every question. If you
skip reset on ingest, deleted or edited files would leave stale chunks behind.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from src.chunk import Chunk


def get_persistent_client(chroma_path: Path):
    """Open (or create) a Chroma database directory.

    Args:
        chroma_path: Folder for the persistent DB (config paths.chroma_path).

    Returns:
        A chromadb PersistentClient.

    Pipeline role: ingest and ask share this path so questions search the
    vectors that ingest just wrote.
    """
    chroma_path = Path(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def reset_collection(client, name: str):
    """Delete the collection if it exists, then create an empty one.

    Args:
        client: Chroma client from get_persistent_client.
        name: Collection name (config paths.collection_name).

    Returns:
        A brand-new empty collection.

    Pipeline role: v1 ingest is a full rebuild. Resetting drops chunks from
    files you removed so they cannot be retrieved by accident.
    """
    try:
        client.delete_collection(name)
    except Exception:
        # First ingest — nothing to delete.
        pass
    # We pass our own embeddings (local ONNX MiniLM), so Chroma must not also embed.
    return client.create_collection(name=name, embedding_function=None)


def upsert_chunks(collection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Write chunks and their vectors into the collection.

    Args:
        collection: Empty (or target) Chroma collection.
        chunks: Output of chunk.py, same length as embeddings.
        embeddings: Output of embed.py, one vector per chunk.

    Returns:
        None. Side effect: rows appear in Chroma.

    Pipeline role: this is the last ingest step. Ids include source and
    chunk_index so a re-ingest is deterministic.
    """
    if not chunks:
        return
    ids = [f"{chunk.source}::{chunk.chunk_index}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    # Metadata is how retrieve.py rebuilds a Chunk without re-reading data/.
    metadatas = [
        {"source": chunk.source, "chunk_index": chunk.chunk_index} for chunk in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
