"""Phase 6 — Prompt + generate: answer only from retrieved chunks.

The chat model has no built-in access to your wiki. We paste the retrieved
chunks into the prompt and forbid it from using anything else. Temperature 0
asks it to copy, not to invent a fluent-sounding extra fact.

If you skip this phase you only have search, not an answer. If you skip the
"only use this context" rule, the model will fill gaps from training data
and you will not know which sentences came from your notes.
"""

from __future__ import annotations

from src.chunk import Chunk
from src.embed import get_openai_client

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the notes "
    "provided in the user message. If the notes do not contain the answer, "
    "say you don't know. Never invent facts. Cite sources by filename."
)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    """Build the chat messages: system rules + labeled context + question.

    Args:
        question: The user's question.
        chunks: Top-k retrieved chunks (may be empty).

    Returns:
        A two-message list suitable for chat.completions.create.

    Pipeline role: this is the grounded prompt. Tests inspect it to prove
    the orchid-42 chunk actually reaches the model, and that empty context
    still tells the model to say it doesn't know.
    """
    if chunks:
        # Filename headers make citations easy and keep sources visible.
        blocks = [f"[{chunk.source}]\n{chunk.text}" for chunk in chunks]
        context = "\n\n".join(blocks)
    else:
        context = "(No notes were retrieved for this question.)"

    user = (
        "Answer the question using ONLY the notes below. "
        "If the notes are missing or irrelevant, say you don't know. "
        "Never invent facts.\n\n"
        f"Notes:\n{context}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_sources(chunks: list[Chunk], snippet_len: int = 120) -> list[str]:
    """Turn retrieved chunks into filename + short snippet lines.

    Args:
        chunks: Chunks that were sent to the model.
        snippet_len: Max characters of chunk text to show.

    Returns:
        Human-readable source lines for the CLI.

    Pipeline role: printed after the answer so you can see which files
    retrieval used, even when you did not pass --show-chunks.
    """
    lines: list[str] = []
    for chunk in chunks:
        snippet = " ".join(chunk.text.split())
        if len(snippet) > snippet_len:
            snippet = snippet[: snippet_len - 3] + "..."
        lines.append(f"- {chunk.source}: \"{snippet}\"")
    return lines


def generate_answer(
    question: str,
    chunks: list[Chunk],
    *,
    model: str,
    temperature: float,
    client=None,
) -> str:
    """Call the chat model with the grounded prompt.

    Args:
        question: The user's question.
        chunks: Retrieved context (empty is allowed).
        model: Chat model name from config (default llama3.2:1b via Ollama).
        temperature: 0 copies from notes; higher values invent more freely.
        client: Optional OpenAI-like client; created for local Ollama if omitted.

    Returns:
        The model's reply text.

    Pipeline role: last step of ask(). We still call the model when chunks
    are empty so the don't-know instruction is what it sees.
    """
    if client is None:
        client = get_openai_client()
    messages = build_messages(question, chunks)
    # temperature=0: RAG should quote the notes, not "be creative".
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content or ""
