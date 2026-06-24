"""Tests for /mood command."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from sector_ai.mood import Emotion, MoodReport, mood_cmd, render_mood_report

# --- render_mood_report ---


def test_render_contains_vibe_check():
    report = MoodReport(
        overall_mood="The chat is lively",
        energy_level=7,
        emotions=[Emotion(name="Joy", emoji="😊", intensity=6)],
        vibe_metaphor="A jazz cafe at midnight",
        top_topics=["AI", "cooking"],
    )
    rendered = render_mood_report(report)
    assert "Vibe Check" in rendered
    assert "The chat is lively" in rendered
    assert "7/10" in rendered
    assert "Joy" in rendered
    assert "A jazz cafe at midnight" in rendered
    assert "AI" in rendered
    assert "cooking" in rendered


def test_render_energy_bar():
    report = MoodReport(
        overall_mood="Calm",
        energy_level=3,
        emotions=[Emotion(name="Peace", emoji="☮️", intensity=5)],
        vibe_metaphor="A still lake",
        top_topics=["nature"],
    )
    rendered = render_mood_report(report)
    assert "⚡⚡⚡" in rendered
    assert "3/10" in rendered


def test_render_escapes_markdown_chars():
    """Markdown control chars in LLM output must be escaped."""
    report = MoodReport(
        overall_mood="The chat is *wild* and _crazy_",
        energy_level=5,
        emotions=[Emotion(name="Joy_and_Fear", emoji="😊", intensity=5)],
        vibe_metaphor="A [secret] garden",
        top_topics=["C++", "node_js"],
    )
    rendered = render_mood_report(report)
    # Escaped chars should not appear unescaped
    assert "\\*wild\\*" in rendered
    assert "\\_crazy\\_" in rendered
    assert "\\_and\\_" in rendered
    assert "\\[secret]" in rendered


# --- mood_cmd ---


async def test_mood_empty_history(mock_update, mock_context):
    update = mock_update(text="/mood")
    await mood_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once_with("No recent messages to analyze!")


async def test_mood_invokes_chain(mock_update, mock_context):
    mock_context.chat_message_history.append(HumanMessage(name="alice", content="alice: hello"))
    update = mock_update(text="/mood")

    mock_report = MoodReport(
        overall_mood="Happy vibes",
        energy_level=8,
        emotions=[Emotion(name="Joy", emoji="😊", intensity=7)],
        vibe_metaphor="Sunshine",
        top_topics=["greetings"],
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_report

    with patch("sector_ai.mood.PydanticOutputParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser.get_format_instructions.return_value = "format"
        mock_parser_cls.return_value = mock_parser

        mock_context.get_templated_messages.return_value.__or__ = MagicMock(
            return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
        )

        await mood_cmd(update, mock_context)

    mock_chain.invoke.assert_called_once()
    update.message.reply_text.assert_awaited()
    # Verify format was reset
    assert mock_context.bot_ollama.format == ""
