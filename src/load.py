"""Phase 1 — Load: turn a folder of markdown files into Document objects.

RAG cannot retrieve what it never read. This phase walks data/, keeps the
filename, and stores the raw text. No chunking, no embeddings yet.

If you skip load, later phases have nothing to split or embed. If you drop
the filename here, you cannot cite sources after retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """One markdown file as the rest of the pipeline sees it."""

    source: str  # filename only, e.g. "wifi-and-office.md"
    text: str


def load_markdown(data_dir: Path) -> list[Document]:
    """Read every .md file in data_dir, sorted by filename.

    Args:
        data_dir: Folder that contains the wiki notes.

    Returns:
        One Document per markdown file, source set to the filename (not a path).

    Pipeline role: ingest starts here so chunk.py receives text plus a stable
    source name to copy onto every chunk.
    """
    data_dir = Path(data_dir)
    documents: list[Document] = []
    # Only .md — a stray .txt or .pdf in v1 is ignored so the loader stays obvious.
    for path in sorted(data_dir.glob("*.md")):
        documents.append(
            Document(source=path.name, text=path.read_text(encoding="utf-8"))
        )
    return documents
