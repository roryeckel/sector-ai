"""Tests for /roast and /hype commands."""

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from sector_ai.roast import _resolve_target, hype_cmd, roast_cmd

# --- _resolve_target ---


def test_resolve_self_roast(mock_update, mock_context):
    """No @mention defaults to sender's username."""
    update = mock_update(text="/roast", username="alice")
    mock_context.chat_message_history.append(HumanMessage(name="alice", content="alice: hello world"))
    result = _resolve_target(update, mock_context)
    assert result is not None
    assert result[0] == "alice"
    assert "hello world" in result[1]


def test_resolve_with_at_prefix(mock_update, mock_context):
    update = mock_update(text="/roast @bob", username="alice")
    mock_context.chat_message_history.append(HumanMessage(name="bob", content="bob: I love python"))
    result = _resolve_target(update, mock_context)
    assert result is not None
    assert result[0] == "bob"


def test_resolve_without_at_prefix(mock_update, mock_context):
    update = mock_update(text="/roast bob", username="alice")
    mock_context.chat_message_history.append(HumanMessage(name="bob", content="bob: I love python"))
    result = _resolve_target(update, mock_context)
    assert result is not None
    assert result[0] == "bob"


def test_resolve_case_insensitive(mock_update, mock_context):
    update = mock_update(text="/roast BOB", username="alice")
    mock_context.chat_message_history.append(HumanMessage(name="bob", content="bob: hey there"))
    result = _resolve_target(update, mock_context)
    assert result is not None
    assert result[0] == "BOB"


def test_resolve_no_messages_found(mock_update, mock_context):
    update = mock_update(text="/roast @ghost", username="alice")
    result = _resolve_target(update, mock_context)
    assert result is None


def test_resolve_no_username_self_roast(mock_update, mock_context):
    """User without a Telegram username should get None (not crash)."""
    update = mock_update(text="/roast")
    update.message.from_user.username = None
    result = _resolve_target(update, mock_context)
    assert result is None


def test_resolve_filters_only_human_messages(mock_update, mock_context):
    update = mock_update(text="/roast alice", username="alice")
    mock_context.chat_message_history.append(AIMessage(content="I am a bot"))
    mock_context.chat_message_history.append(HumanMessage(name="alice", content="alice: real message"))
    result = _resolve_target(update, mock_context)
    assert result is not None
    assert "real message" in result[1]
    assert "I am a bot" not in result[1]


# --- roast_cmd ---


async def test_roast_no_messages(mock_update, mock_context):
    update = mock_update(text="/roast @nobody", username="testuser")
    await roast_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once()
    assert "nobody" in update.message.reply_text.call_args[0][0]


async def test_roast_no_username_error(mock_update, mock_context):
    """User without a username gets a helpful message instead of a crash."""
    update = mock_update(text="/roast")
    update.message.from_user.username = None
    await roast_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once()
    assert "username" in update.message.reply_text.call_args[0][0].lower()


async def test_roast_streams_response(mock_update, mock_context):
    mock_context.chat_message_history.append(HumanMessage(name="bob", content="bob: hello"))
    update = mock_update(text="/roast @bob", username="alice")
    update.message.reply_text = AsyncMock(return_value=MagicMock())

    mock_chain = MagicMock()
    mock_chain.stream.return_value = iter([])

    mock_context.get_templated_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

    with patch("sector_ai.roast.handle_streaming_response", new_callable=AsyncMock) as mock_stream:
        await roast_cmd(update, mock_context)
        mock_stream.assert_awaited_once()


# --- hype_cmd ---


async def test_hype_no_messages(mock_update, mock_context):
    update = mock_update(text="/hype @nobody", username="testuser")
    await hype_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once()
    assert "nobody" in update.message.reply_text.call_args[0][0]


async def test_hype_streams_response(mock_update, mock_context):
    mock_context.chat_message_history.append(HumanMessage(name="bob", content="bob: hello"))
    update = mock_update(text="/hype @bob", username="alice")
    update.message.reply_text = AsyncMock(return_value=MagicMock())

    mock_chain = MagicMock()
    mock_chain.stream.return_value = iter([])

    mock_context.get_templated_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

    with patch("sector_ai.roast.handle_streaming_response", new_callable=AsyncMock) as mock_stream:
        await hype_cmd(update, mock_context)
        mock_stream.assert_awaited_once()
