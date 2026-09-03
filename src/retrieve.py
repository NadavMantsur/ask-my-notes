"""Phase 5 — Retrieve: find the top-k chunks whose vectors are nearest the question.

The question is embedded with the *same* model used at ingest. Chroma ranks
stored vectors by similarity and returns the text of the winners. Those texts
are the only notes the chat model is allowed to see.

If you skip retrieval, the model answers from memory and will invent. If k is
1 you may miss a fact split across two windows; if k is huge you stuff the
prompt with noise and the model may ignore the right sentence.
"""

from __future__ import annotations

from src.chunk import Chunk


def retrieve_chunks(collection, query_embedding: list[float], *, k: int) -> list[Chunk]:
    """Return the k nearest chunks to the query embedding.

    Args:
        collection: Chroma collection filled by ingest.
        query_embedding: Vector of the user question (from embed.py).
        k: How many chunks to keep (config retrieval.top_k).

    Returns:
        Chunk objects with text, source, and chunk_index, most similar first.

    Pipeline role: ask() embeds the question, calls this, then hands the
    chunks to generate.py to build the grounded prompt.
    """
    if k < 1:
        return []
    # n_results is top-k: only this many neighbors enter the prompt.
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    chunks: list[Chunk] = []
    for text, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        chunks.append(
            Chunk(
                text=text,
                source=str(metadata.get("source", "unknown")),
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
        )
    return chunks
