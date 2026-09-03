"""Tests for loading markdown notes into Document objects."""

from src.load import load_markdown


def test_loads_markdown_files_and_ignores_other_extensions(tmp_path):
    (tmp_path / "b-second.md").write_text("beta body")
    (tmp_path / "a-first.md").write_text("alpha body")
    (tmp_path / "notes.txt").write_text("should be ignored")

    docs = load_markdown(tmp_path)

    assert [doc.source for doc in docs] == ["a-first.md", "b-second.md"]
    assert docs[0].text == "alpha body"
    assert docs[1].text == "beta body"
    assert all("/" not in doc.source for doc in docs)
