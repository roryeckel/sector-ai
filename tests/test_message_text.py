"""Tests for text extraction from string and block content."""

from langchain_core.messages import AIMessage, HumanMessage

from sector_ai.message_text import message_refused, message_text


def test_plain_string_content():
    assert message_text(HumanMessage(content="hello")) == "hello"
    assert message_refused(HumanMessage(content="hello")) is False


def test_text_and_refusal_blocks():
    message = AIMessage(content=[{"type": "text", "text": "hi"}, {"type": "refusal", "refusal": "no"}])

    assert message_text(message) == "hino"
    assert message_refused(message) is True


def test_non_dict_blocks_are_skipped():
    from types import SimpleNamespace

    message = SimpleNamespace(content=["hello", {"type": "text", "text": " world"}])

    assert message_text(message) == "hello world"
    assert message_refused(message) is False


def test_none_text_is_ignored():
    from types import SimpleNamespace

    message = SimpleNamespace(content=[{"type": "text", "text": None}])

    assert message_text(message) == ""
