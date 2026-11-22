"""Tests for the markdown-to-Telegram renderer."""

from unittest.mock import MagicMock

from telegram import MessageEntity

from sector_ai.telegram_format import (
    MESSAGE_LIMIT,
    render_chunks,
    render_markdown,
    render_preview,
    split_for_telegram,
    telegram_entities,
    utf16_len,
)


def types(entities):
    return {entity.type for entity in entities}


def test_inline_markup_becomes_entities():
    text, entities = render_markdown("Hello **bold** and `code`")

    assert text == "Hello bold and code"
    assert types(entities) == {"bold", "code"}


def test_a_heading_is_bold_without_an_emoji_prefix():
    text, entities = render_markdown("# Title")

    assert text == "Title"
    assert {"bold", "underline"} <= types(entities)


def test_lists_keep_their_markers_as_text():
    text, entities = render_markdown("- one\n- two")

    assert text.strip() == "• one\n• two"
    assert entities == []


def test_a_code_fence_keeps_its_language():
    text, entities = render_markdown("```python\nprint(1)\n```")

    assert text == "print(1)"
    assert entities[0].type == "pre"
    assert entities[0].language == "python"


def test_a_table_becomes_a_monospace_block():
    _, entities = render_markdown("| a | b |\n|---|---|\n| 1 | 2 |")

    assert entities[0].type == "pre"


def test_an_unsafe_link_is_not_attached():
    text, entities = render_markdown("[click](javascript:alert(1)) and [ok](https://example.com)")

    assert text == "click and ok"
    assert [entity.url for entity in entities] == ["https://example.com"]


def test_offsets_count_surrogate_pairs():
    text, entities = render_markdown("**🥰 hi**")

    assert text == "🥰 hi"
    # Two UTF-16 units for the emoji plus three for " hi".
    assert entities[0].length == 5


def test_disabled_rendering_returns_the_raw_text():
    text, entities = render_markdown("**bold**", enabled=False)

    assert text == "**bold**"
    assert entities == []


def test_a_conversion_failure_falls_back_to_the_source(monkeypatch):
    monkeypatch.setattr("sector_ai.telegram_format.convert", MagicMock(side_effect=RuntimeError("boom")))

    text, entities = render_markdown("**bold**")

    assert text == "**bold**"
    assert entities == []


def test_a_preview_fits_one_message_including_the_cursor():
    text, _ = render_preview("x" * (MESSAGE_LIMIT + 100), cursor="🦙")

    assert text.endswith("🦙")
    assert utf16_len(text) <= MESSAGE_LIMIT


def test_splitting_clips_entities_at_the_break():
    text, entities = render_markdown("**" + "y" * 5000 + "**")

    chunks = split_for_telegram(text, entities)

    assert [utf16_len(chunk) for chunk, _ in chunks] == [MESSAGE_LIMIT, 904]
    assert [chunk[0].length for _, chunk in chunks] == [MESSAGE_LIMIT, 904]


def test_entities_are_translated_for_python_telegram_bot():
    _, entities = render_markdown("**bold**")

    converted = telegram_entities(entities)

    assert converted == [MessageEntity(type=MessageEntity.BOLD, offset=0, length=4)]


def test_chunks_stay_plain_when_rendering_is_disabled():
    chunks = render_chunks("**bold**", enabled=False)

    assert chunks == [("**bold**", [])]
