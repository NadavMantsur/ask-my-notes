"""Tests for overlapping character-window chunking."""

from src.chunk import chunk_documents
from src.load import Document


def test_short_document_is_one_chunk():
    doc = Document(source="tiny.md", text="a" * 50)
    chunks = chunk_documents([doc], chunk_size=500, overlap=80)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].source == "tiny.md"
    assert chunks[0].text == "a" * 50


def test_long_document_uses_size_and_overlap():
    doc = Document(source="long.md", text="x" * 1200)
    chunks = chunk_documents([doc], chunk_size=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 500 for chunk in chunks)
    full = [c for c in chunks if len(c.text) == 500]
    assert len(full) >= 2
    assert full[0].text[-80:] == full[1].text[:80]


def test_empty_document_yields_no_chunks():
    doc = Document(source="empty.md", text="")
    assert chunk_documents([doc], chunk_size=500, overlap=80) == []


def test_chunk_size_argument_is_honored():
    doc = Document(source="sized.md", text="y" * 250)
    chunks = chunk_documents([doc], chunk_size=100, overlap=20)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert len(chunks) > 1
