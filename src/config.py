"""Load pipeline knobs from config.toml.

This is not a RAG phase. It is the settings layer that every phase reads
through cli.py so you can change top_k or temperature without editing Python.

Skipping a config file would bury magic numbers inside retrieve.py and
chunk.py. You would not know which knobs need a fresh ingest (chunk size,
embedding model) versus which only affect the next ask (top_k, temperature).

Secrets do not live here. Optional OLLAMA_BASE_URL stays in the environment / .env.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

# Directory that contains src/ — used only as a fallback mental model.
# Real path resolution is against the config file's parent so --config copies work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Typed snapshot of config.toml after validation and path resolution."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str
    chat_model: str
    temperature: float
    data_dir: Path
    chroma_path: Path
    collection_name: str


def load_settings(config_path: Path) -> Settings:
    """Parse and validate a TOML config file.

    Args:
        config_path: Path to config.toml (or a copy passed with --config).

    Returns:
        Settings with data_dir and chroma_path resolved to absolute paths.

    Pipeline role: cli.py calls this once at startup and passes fields into
    each RAG phase so tests can inject literals instead of reading the file.
    """
    config_path = Path(config_path)
    # Fail early with a how-to instead of a FileNotFoundError traceback.
    if not config_path.is_file():
        raise SystemExit(
            "config.toml not found. Copy the project config.toml or pass --config."
        )

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    chunking = raw["chunking"]
    retrieval = raw["retrieval"]
    embedding = raw["embedding"]
    generation = raw["generation"]
    paths = raw["paths"]

    chunk_size = int(chunking["size"])
    overlap = int(chunking["overlap"])
    top_k = int(retrieval["top_k"])
    temperature = float(generation["temperature"])
    embedding_model = str(embedding["model"]).strip()
    chat_model = str(generation["model"]).strip()
    collection_name = str(paths["collection_name"]).strip()

    # top_k=0 would retrieve nothing and the model would always say it doesn't know.
    if top_k < 1:
        raise SystemExit("Invalid config: top_k must be >= 1.")
    # Overlap must be strictly smaller than the window or the splitter never advances.
    if overlap < 0 or chunk_size <= overlap:
        raise SystemExit(
            "Invalid config: chunking.overlap must be >= 0 and smaller than chunking.size."
        )
    # OpenAI accepts 0–2; values outside that are almost always a typo.
    if not 0.0 <= temperature <= 2.0:
        raise SystemExit("Invalid config: temperature must be between 0.0 and 2.0.")
    if not embedding_model:
        raise SystemExit("Invalid config: embedding.model must be non-empty.")
    if not chat_model:
        raise SystemExit("Invalid config: generation.model must be non-empty.")
    if not collection_name:
        raise SystemExit("Invalid config: paths.collection_name must be non-empty.")

    # Resolve against the config file's directory so a copied config.toml
    # still points at its own data/ and chroma_db/ neighbors.
    config_parent = config_path.parent.resolve()
    data_dir = (config_parent / paths["data_dir"]).resolve()
    chroma_path = (config_parent / paths["chroma_path"]).resolve()

    return Settings(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=top_k,
        embedding_model=embedding_model,
        chat_model=chat_model,
        temperature=temperature,
        data_dir=data_dir,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )
