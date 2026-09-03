"""Tests for prompt assembly and the mocked chat completion."""

from unittest.mock import MagicMock

from src.chunk import Chunk
from src.generate import build_messages, generate_answer


ORCHID_CHUNK = Chunk(
    text="The office Wi-Fi password is orchid-42 on Lumen-Office.",
    source="wifi-and-office.md",
    chunk_index=0,
)


def _joined(messages: list[dict]) -> str:
    return "\n".join(item["content"] for item in messages)


def test_build_messages_includes_chunk_text_and_filename():
    messages = build_messages("What is the office Wi-Fi password?", [ORCHID_CHUNK])
    blob = _joined(messages)
    assert "orchid-42" in blob
    assert "wifi-and-office.md" in blob


def test_empty_context_instructs_model_not_to_invent():
    messages = build_messages("What is the CEO's pet's name?", [])
    blob = _joined(messages).lower()
    assert "don't know" in blob or "do not know" in blob
    assert "only" in blob and "notes" in blob
    assert "never invent" in blob or "do not invent" in blob or "never invent facts" in blob


def test_generate_answer_calls_chat_with_model_and_temperature():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="The password is orchid-42."))
    ]

    answer = generate_answer(
        "What is the office Wi-Fi password?",
        [ORCHID_CHUNK],
        model="gpt-4o-mini",
        temperature=0.0,
        client=mock_client,
    )

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.0
    assert answer == "The password is orchid-42."


def test_generate_answer_forwards_temperature_from_config():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="ok"))
    ]
    generate_answer(
        "q",
        [ORCHID_CHUNK],
        model="gpt-4o-mini",
        temperature=0.2,
        client=mock_client,
    )
    assert mock_client.chat.completions.create.call_args.kwargs["temperature"] == 0.2
