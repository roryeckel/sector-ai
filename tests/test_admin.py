"""Tests for admin security boundary and admin commands."""

from unittest.mock import AsyncMock

from sector_ai.admin import admin_only, clear_cmd, model_callback, models_cmd, system_prompt_cmd, temperature_cmd

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


# --- models_cmd / model_callback ---


async def test_models_cmd_builds_keyboard(make_context, make_update, monkeypatch):
    context = make_context("unused")
    monkeypatch.setattr(
        context,
        "get_models",
        lambda: [{"model": "a:latest", "label": "✅ a:latest"}, {"model": "b:latest", "label": "b:latest"}],
    )
    update = make_update(text="/models", username="ex0dus")

    await models_cmd(update, context)

    _, kwargs = update.message.calls_to("reply_text")[0]
    keyboard = kwargs["reply_markup"].inline_keyboard
    assert [row[0].text for row in keyboard] == ["✅ a:latest", "b:latest"]
    assert [row[0].callback_data for row in keyboard] == ["a:latest", "b:latest"]


async def test_model_callback_loads_and_reports_details(make_context, make_update, monkeypatch):
    context = make_context("unused")
    loaded = []
    monkeypatch.setattr(context, "load_model", lambda name=None: loaded.append(name))
    monkeypatch.setattr(
        context,
        "get_model_details",
        lambda: {
            "details": {"families": ["llama"], "parameter_size": "7B", "quantization_level": "Q4_K_M"},
            "model_info": {
                "general.finetune": "chat",
                "general.languages": ["en", "de"],
                "general.tags": ["text-generation"],
            },
        },
    )
    update = make_update(text="/models", username="ex0dus", callback_data="llama3:latest")

    await model_callback(update, context)

    assert loaded == ["llama3:latest"]
    update.callback_query.answer.assert_awaited_once()
    report = update.callback_query.edit_message_text.call_args.kwargs["text"]
    assert "Loaded model: llama3:latest" in report
    assert "Families: llama" in report
    assert "Parameters: 7B" in report
    assert "Quantization Level: Q4_K_M" in report
    assert "Finetune: chat" in report
    assert "Languages: en, de" in report
    assert "Tags: text-generation" in report


# --- system_prompt_cmd ---


async def test_system_prompt_cmd_sets_prompt(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/system be nice", username="ex0dus")

    await system_prompt_cmd(update, context)

    assert context.chat_system_prompt == "be nice"


async def test_system_prompt_cmd_resets_to_default(make_context, make_update):
    context = make_context("unused")
    context.chat_system_prompt = "custom"
    update = make_update(text="/system", username="ex0dus")

    await system_prompt_cmd(update, context)

    assert context.chat_system_prompt == context.config_default_system_prompt
