"""Tests for admin security boundary and admin commands."""

from unittest.mock import AsyncMock

from sector_ai.admin import admin_only, clear_cmd, temperature_cmd

# --- admin_only decorator ---


async def test_allows_admin(mock_update, mock_context):
    inner = AsyncMock()
    wrapped = admin_only(inner)
    update = mock_update(text="/test", username="ex0dus")
    await wrapped(update, mock_context)
    inner.assert_awaited_once()


async def test_blocks_non_admin(mock_update, mock_context):
    inner = AsyncMock()
    wrapped = admin_only(inner)
    update = mock_update(text="/test", username="random")
    await wrapped(update, mock_context)
    inner.assert_not_awaited()
    update.message.reply_text.assert_awaited_once_with("You are not authorized to use this command.")


async def test_context_none_passthrough(mock_update):
    inner = AsyncMock()
    wrapped = admin_only(inner)
    update = mock_update(text="/test", username="anyone")
    await wrapped(update, None)
    inner.assert_awaited_once()


# --- temperature_cmd ---


async def test_temperature_sets_value(mock_update, mock_context):
    update = mock_update(text="/temperature 0.7", username="ex0dus")
    await temperature_cmd.__wrapped__(update, mock_context)
    assert mock_context.bot_ollama.temperature == 0.7
    update.message.reply_text.assert_awaited_once()
    assert "0.7" in update.message.reply_text.call_args[0][0]


async def test_temperature_clears_without_arg(mock_update, mock_context):
    update = mock_update(text="/temperature", username="ex0dus")
    await temperature_cmd.__wrapped__(update, mock_context)
    assert mock_context.bot_ollama.temperature is None
    update.message.reply_text.assert_awaited_once()
    assert "Cleared" in update.message.reply_text.call_args[0][0]


# --- clear_cmd ---


async def test_clear_cmd_resets_state(mock_update, mock_context, sample_config):
    # Pre-populate history
    history = mock_context.chat_message_history
    from langchain_core.messages import HumanMessage

    history.append(HumanMessage(content="old message"))
    assert len(history) == 1

    update = mock_update(text="/clear", username="ex0dus")
    await clear_cmd(update, mock_context)

    assert len(history) == 0
    # clear_cmd sets chat_system_prompt = config_default_system_prompt
    # On a MagicMock this goes through __setattr__, so verify it was assigned
    assert mock_context.chat_system_prompt == sample_config["system_prompts"]["default"]
    update.message.reply_text.assert_awaited_once()
