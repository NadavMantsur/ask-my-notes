"""CLI conductor: ingest runs phases 1–4; ask runs phases 5–6.

Read this file as the pipeline index. Every other module is one phase.
cli.py loads config.toml and .env, then calls those modules in order.

If you skip ingest, ask has an empty Chroma folder. If you skip ask's
retrieve step, the model never sees your notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError

from src.chunk import chunk_documents
from src.config import PROJECT_ROOT, Settings, load_settings
from src.embed import OLLAMA_MISSING_MESSAGE, embed_texts, get_openai_client
from src.generate import format_sources, generate_answer
from src.index import get_persistent_client, reset_collection, upsert_chunks
from src.load import load_markdown
from src.retrieve import retrieve_chunks


def ingest(settings: Settings, embed_fn=embed_texts) -> int:
    """Run load → chunk → embed → store and return how many chunks were written.

    Args:
        settings: Validated config (paths, chunk size, embedding model).
        embed_fn: Defaults to local ONNX embeddings; tests pass a fake.

    Returns:
        Number of chunks upserted into Chroma.

    Pipeline role: this is the ingest command. Changing chunk size or the
    embedding model requires running it again.
    """
    documents = load_markdown(settings.data_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    vectors = embed_fn(
        [chunk.text for chunk in chunks], model=settings.embedding_model
    )
    client = get_persistent_client(settings.chroma_path)
    collection = reset_collection(client, settings.collection_name)
    upsert_chunks(collection, chunks, vectors)
    return len(chunks)


def ask(
    question: str,
    settings: Settings,
    *,
    show_chunks: bool = False,
    embed_fn=embed_texts,
    chat_client=None,
) -> None:
    """Retrieve top-k chunks, optionally print them, then generate an answer.

    Args:
        question: Natural-language question.
        settings: Config including top_k, chat model, temperature, chroma path.
        show_chunks: If True, print retrieved text BEFORE the model runs so
            you can tell retrieval quality apart from generation.
        embed_fn: Question embedder; must match the model used at ingest.
        chat_client: Optional OpenAI-like client for the chat call (Ollama).

    Returns:
        None. Prints answer and sources to stdout.

    Pipeline role: this is the ask command. top_k and temperature take
    effect here without a new ingest.
    """
    chroma = get_persistent_client(settings.chroma_path)
    collection = chroma.get_collection(
        name=settings.collection_name, embedding_function=None
    )
    # Same embedding model as ingest — mixed models break nearest-neighbor.
    query_embedding = embed_fn([question], model=settings.embedding_model)[0]
    chunks = retrieve_chunks(collection, query_embedding, k=settings.top_k)

    if show_chunks:
        print("Retrieved chunks (before generation):")
        if not chunks:
            print("(none)")
        for chunk in chunks:
            print(f"--- {chunk.source} #{chunk.chunk_index} ---")
            print(chunk.text)
            print()

    answer = generate_answer(
        question,
        chunks,
        model=settings.chat_model,
        temperature=settings.temperature,
        client=chat_client,
    )
    print("Answer:")
    print(answer)
    print()
    print("Sources:")
    lines = format_sources(chunks)
    if lines:
        print("\n".join(lines))
    else:
        print("(none)")


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args, load config, and run ingest or ask.

    Args:
        argv: Token list for tests; None reads sys.argv.

    Returns:
        None. Exits via argparse or missing-key SystemExit.

    Pipeline role: process entry point (`python -m src.cli ...`).
    """
    # Load .env here, not in embed.py, so unit tests fully control the env.
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ask questions against a local folder of markdown notes."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.toml"),
        help="Path to config.toml (knobs, not secrets).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load, chunk, embed, and store notes.")

    ask_parser = subparsers.add_parser("ask", help="Retrieve notes and answer.")
    ask_parser.add_argument("question", help="Question to answer from the notes.")
    ask_parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print retrieved chunks before calling the chat model.",
    )

    args = parser.parse_args(argv)
    settings = load_settings(Path(args.config))

    if args.command == "ingest":
        count = ingest(settings, embed_fn=embed_texts)
        print(f"Ingested {count} chunks into {settings.chroma_path}.")
        return

    if args.command == "ask":
        # Pass embed_texts by name so tests can monkeypatch src.cli.embed_texts.
        try:
            ask(
                args.question,
                settings,
                show_chunks=args.show_chunks,
                embed_fn=embed_texts,
                chat_client=get_openai_client(),
            )
        except APIConnectionError:
            raise SystemExit(OLLAMA_MISSING_MESSAGE)
        return


if __name__ == "__main__":
    main()
