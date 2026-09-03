"""Tests for loading and validating config.toml into Settings."""

from pathlib import Path

import pytest

from src.config import load_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = PROJECT_ROOT / "config.toml"


def test_shipped_config_has_expected_defaults():
    settings = load_settings(SHIPPED_CONFIG)
    assert settings.top_k == 4
    assert settings.temperature == 0.0
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 80
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chat_model == "gpt-4o-mini"


def test_paths_resolve_against_config_parent():
    settings = load_settings(SHIPPED_CONFIG)
    assert settings.data_dir == (SHIPPED_CONFIG.parent / "data").resolve()
    assert settings.chroma_path == (SHIPPED_CONFIG.parent / "chroma_db").resolve()
    assert settings.data_dir.is_absolute()
    assert settings.chroma_path.is_absolute()


def test_missing_file_exits_with_clear_message(tmp_path):
    missing = tmp_path / "nope.toml"
    with pytest.raises(SystemExit, match="config.toml not found"):
        load_settings(missing)


def test_top_k_zero_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[chunking]
size = 500
overlap = 80
[retrieval]
top_k = 0
[embedding]
model = "text-embedding-3-small"
[generation]
model = "gpt-4o-mini"
temperature = 0.0
[paths]
data_dir = "data"
chroma_path = "chroma_db"
collection_name = "notes"
"""
    )
    with pytest.raises(SystemExit, match="top_k"):
        load_settings(path)


def test_overlap_not_smaller_than_size_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[chunking]
size = 80
overlap = 80
[retrieval]
top_k = 4
[embedding]
model = "text-embedding-3-small"
[generation]
model = "gpt-4o-mini"
temperature = 0.0
[paths]
data_dir = "data"
chroma_path = "chroma_db"
collection_name = "notes"
"""
    )
    with pytest.raises(SystemExit, match="overlap"):
        load_settings(path)


def test_temperature_out_of_range_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[chunking]
size = 500
overlap = 80
[retrieval]
top_k = 4
[embedding]
model = "text-embedding-3-small"
[generation]
model = "gpt-4o-mini"
temperature = 3
[paths]
data_dir = "data"
chroma_path = "chroma_db"
collection_name = "notes"
"""
    )
    with pytest.raises(SystemExit, match="temperature"):
        load_settings(path)
