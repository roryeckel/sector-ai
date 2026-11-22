"""Tests for admin security boundary and admin commands."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram.error import BadRequest

from sector_ai.admin import (
    MODELS_PER_PAGE,
    _model_key,
    admin_only,
    clear_cmd,
    compact_cmd,
    model_callback,
    models_cmd,
    system_prompt_cmd,
    temperature_cmd,
)

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
    assert mock_context.bot_llm.temperature == 0.7
    update.message.reply_text.assert_awaited_once()
    assert "0.7" in update.message.reply_text.call_args[0][0]


async def test_temperature_clears_without_arg(mock_update, mock_context):
    update = mock_update(text="/temperature", username="ex0dus")
    await temperature_cmd.__wrapped__(update, mock_context)
    assert mock_context.bot_llm.temperature is None
    update.message.reply_text.assert_awaited_once()
    assert "Cleared" in update.message.reply_text.call_args[0][0]


# --- clear_cmd ---


async def test_clear_cmd_resets_state(mock_update, mock_context):
    update = mock_update(text="/clear", username="ex0dus")
    await clear_cmd(update, mock_context)

    mock_context.clear_history.assert_called_once()
    # clear_cmd sets chat_system_prompt to the settings default
    assert mock_context.chat_system_prompt == mock_context.settings.system_prompts.default
    update.message.reply_text.assert_awaited_once()


# --- compact_cmd ---


async def test_compact_cmd_reports_the_token_change(make_context, make_update, monkeypatch):
    context = make_context("unused")
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(side_effect=[900, 120]))
    monkeypatch.setattr(
        context,
        "compact",
        AsyncMock(return_value={"compacted": True, "evicted": 4, "kept": 8, "summary": "s"}),
    )
    update = make_update(text="/compact", username="ex0dus")

    await compact_cmd(update, context)

    assert update.message.texts("reply_text") == ["Compacting..."]
    assert update.message.texts("edit_text") == ["Compacted 4 messages into a summary: 900 → 120 tokens."]


async def test_compact_cmd_needs_something_to_compact(make_context, make_update, monkeypatch):
    context = make_context("unused")
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=10))
    monkeypatch.setattr(
        context,
        "compact",
        AsyncMock(return_value={"compacted": False, "evicted": 0, "kept": 2, "summary": ""}),
    )
    update = make_update(text="/compact", username="ex0dus")

    await compact_cmd(update, context)

    assert update.message.texts("edit_text") == ["Not enough history to compact yet."]


async def test_compact_cmd_reports_a_failure(make_context, make_update, monkeypatch):
    context = make_context("unused")
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=10))
    monkeypatch.setattr(context, "compact", AsyncMock(side_effect=RuntimeError("summarizer down")))
    update = make_update(text="/compact", username="ex0dus")

    await compact_cmd(update, context)

    assert update.message.texts("edit_text") == ["Compaction failed; the history was left unchanged. Check the logs."]


async def test_compact_cmd_is_admin_only(make_context, make_update, monkeypatch):
    context = make_context("unused")
    count = AsyncMock(return_value=10)
    monkeypatch.setattr(context, "count_context_tokens", count)
    update = make_update(text="/compact", username="random")

    await compact_cmd(update, context)

    assert update.message.texts("reply_text") == ["You are not authorized to use this command."]
    count.assert_not_awaited()


# --- models_cmd / model_callback ---


async def test_models_cmd_builds_keyboard(make_context, make_update, monkeypatch):
    context = make_context("unused")
    models = [
        {"model": "a:latest", "label": "✅ a:latest"},
        {"model": "b:latest", "label": "b:latest"},
        {"model": "x" * 100, "label": "x" * 100},
    ]

    async def get_models():
        return models

    monkeypatch.setattr(context, "get_models", get_models)
    update = make_update(text="/models", username="ex0dus")

    await models_cmd(update, context)

    _, kwargs = update.message.calls_to("reply_text")[0]
    keyboard = kwargs["reply_markup"].inline_keyboard
    assert [row[0].text for row in keyboard[:2]] == ["✅ a:latest", "b:latest"]
    assert keyboard[2][0].text.endswith("…")
    assert len(keyboard[2][0].text.encode("utf-8")) <= 60
    keys = [row[0].callback_data for row in keyboard]
    assert all(len(key.encode("utf-8")) <= 64 for key in keys)
    assert [context.chat_data["model_choices"][key] for key in keys] == [model["model"] for model in models]


async def test_models_cmd_handles_no_models(make_context, make_update, monkeypatch):
    context = make_context("unused")
    context.chat_data["models"] = [{"model": "old", "label": "old"}]
    context.chat_data["model_choices"] = {"old": "old"}

    async def get_models():
        return []

    monkeypatch.setattr(context, "get_models", get_models)
    update = make_update(text="/models", username="ex0dus")

    await models_cmd(update, context)

    assert update.message.texts("reply_text") == ["No models available."]
    assert "models" not in context.chat_data
    assert "model_choices" not in context.chat_data


async def test_models_cmd_ignores_an_invalid_page(make_context, make_update, monkeypatch):
    context = make_context("unused")

    async def get_models():
        return [{"model": "a:latest", "label": "a:latest"}]

    monkeypatch.setattr(context, "get_models", get_models)
    update = make_update(text="/models nope", username="ex0dus")

    await models_cmd(update, context)

    assert "page 1/1" in update.message.texts("reply_text")[0]


async def test_model_callback_loads_and_reports_details(make_context, make_update, monkeypatch):
    context = make_context("unused")
    loaded = []
    monkeypatch.setattr(context, "load_model", lambda name=None: loaded.append(name))

    async def get_model_details():
        return {
            "details": {"families": ["llama"], "parameter_size": "7B", "quantization_level": "Q4_K_M"},
            "model_info": {
                "general.finetune": "chat",
                "general.languages": ["en", "de"],
                "general.tags": ["text-generation"],
            },
        }

    monkeypatch.setattr(context, "get_model_details", get_model_details)
    key = _model_key("llama3:latest")
    context.chat_data["model_choices"] = {key: "llama3:latest"}
    update = make_update(text="/models", username="ex0dus", callback_data=key)

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


async def test_model_callback_rejects_an_expired_menu(make_context, make_update):
    context = make_context("unused")
    context.chat_data["model_choices"] = {"fresh": "model"}
    update = make_update(text="/models", username="ex0dus", callback_data="stale")

    await model_callback(update, context)

    update.callback_query.answer.assert_awaited_once()
    assert "expired" in update.callback_query.answer.call_args[0][0]
    update.callback_query.edit_message_text.assert_not_awaited()


async def test_model_callback_rejects_a_missing_mapping(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/models", username="ex0dus", callback_data="stale")

    await model_callback(update, context)

    update.callback_query.answer.assert_awaited_once()
    assert "expired" in update.callback_query.answer.call_args[0][0]


async def test_models_cmd_paginates(make_context, make_update, monkeypatch):
    context = make_context("unused")
    models = [{"model": f"model-{i}", "label": f"model-{i}"} for i in range(MODELS_PER_PAGE * 2 + 21)]

    async def get_models():
        return models

    monkeypatch.setattr(context, "get_models", get_models)
    update = make_update(text="/models", username="ex0dus")

    await models_cmd(update, context)

    _, kwargs = update.message.calls_to("reply_text")[0]
    keyboard = kwargs["reply_markup"].inline_keyboard
    assert len(keyboard) == MODELS_PER_PAGE + 1
    assert keyboard[-1][0].callback_data == "page:1"
    assert len(context.chat_data["models"]) == len(models)
    assert len(context.chat_data["model_choices"]) == len(models)


async def test_models_cmd_second_page(make_context, make_update, monkeypatch):
    context = make_context("unused")
    models = [{"model": f"model-{i}", "label": f"model-{i}"} for i in range(MODELS_PER_PAGE * 2 + 21)]

    async def get_models():
        return models

    monkeypatch.setattr(context, "get_models", get_models)
    update = make_update(text="/models 2", username="ex0dus")

    await models_cmd(update, context)

    _, kwargs = update.message.calls_to("reply_text")[0]
    keyboard = kwargs["reply_markup"].inline_keyboard
    assert keyboard[0][0].text == f"model-{MODELS_PER_PAGE}"
    assert [button.callback_data for button in keyboard[-1]] == ["page:0", "page:2"]


async def test_model_callback_page_requires_a_menu(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/models", username="ex0dus", callback_data="page:1")

    await model_callback(update, context)

    update.callback_query.answer.assert_awaited_once()
    assert "expired" in update.callback_query.answer.call_args[0][0]


async def test_model_callback_changes_page(make_context, make_update):
    context = make_context("unused")
    context.chat_data["models"] = [{"model": f"model-{i}", "label": f"model-{i}"} for i in range(81)]
    update = make_update(text="/models", username="ex0dus", callback_data="page:1")

    await model_callback(update, context)

    update.callback_query.answer.assert_awaited_once()
    kwargs = update.callback_query.edit_message_text.call_args.kwargs
    assert update.callback_query.edit_message_text.call_args.args[0] == "Available models (page 2/3):"
    assert len(kwargs["reply_markup"].inline_keyboard) == MODELS_PER_PAGE + 1


async def test_model_callback_ignores_a_duplicate_edit(make_context, make_update):
    context = make_context("unused")
    context.chat_data["models"] = [{"model": f"model-{i}", "label": f"model-{i}"} for i in range(81)]
    update = make_update(text="/models", username="ex0dus", callback_data="page:1")
    update.callback_query.edit_message_text = AsyncMock(side_effect=BadRequest("Message is not modified"))

    await model_callback(update, context)  # must not raise

    update.callback_query.answer.assert_awaited_once()


async def test_non_admin_callback_is_denied(make_context):
    context = make_context("unused")
    query = SimpleNamespace(answer=AsyncMock())
    update = SimpleNamespace(
        callback_query=query,
        message=None,
        effective_message=None,
        effective_user=SimpleNamespace(username="random"),
    )

    await model_callback(update, context)

    query.answer.assert_awaited_once()
    assert "not authorized" in query.answer.call_args[0][0]


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

    assert context.chat_system_prompt == context.settings.system_prompts.default
