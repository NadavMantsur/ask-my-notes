"""Phase 2 — Chunk: split each document into overlapping windows.

Embeddings and retrieval work on *pieces*, not whole files. A 3,000-character
wiki page would become one vector that is "about" too many topics at once.

If you skip chunking, a question about Wi-Fi competes with every other
sentence in the same file. If you skip overlap, a fact that sits on a window
boundary is split in half and may never match the question.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.load import Document


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit: text plus enough metadata to cite it later."""

    text: str
    source: str
    chunk_index: int


def chunk_documents(
    documents: list[Document], *, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Split documents into character windows with overlap.

    Args:
        documents: Loaded markdown files from load.py.
        chunk_size: Max characters per chunk (config chunking.size, default 500).
        overlap: Characters copied from the end of one window into the next
            (config chunking.overlap, default 80).

    Returns:
        Chunks in file order, each tagged with source filename and chunk_index.

    Pipeline role: ingest embeds these strings and stores them in Chroma.
    Retrieval returns chunks, not files, which is why source lives on every row.
    """
    chunks: list[Chunk] = []
    for document in documents:
        # Empty notes produce no vectors — nothing to retrieve.
        if not document.text:
            continue
        start = 0
        index = 0
        length = len(document.text)
        while start < length:
            end = min(start + chunk_size, length)
            window = document.text[start:end]
            # Filename on every chunk: citations after retrieval need it.
            chunks.append(
                Chunk(text=window, source=document.source, chunk_index=index)
            )
            if end == length:
                break
            # Step by (size - overlap) so the next window re-reads the boundary.
            start += chunk_size - overlap
            index += 1
    return chunks
