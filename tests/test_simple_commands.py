"""Tests for the thin commands that delegate to handle_chat or the model."""

from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, SystemMessage

from sector_ai.characterize import characterize_cmd
from sector_ai.summarize import summarize_cmd
from sector_ai.tokens import tokens_cmd


async def test_summarize_uses_summarization_prompt(make_context, make_update):
    context = make_context("a short summary")
    update = make_update(text="/summarize")

    await summarize_cmd(update, context)

    sent = context.bot_llm.received[0]
    expected = context.settings.system_prompts.summarization.format(MAX_LEN=4096, USERNAME="testbot")
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == expected
    assert update.message.shown()[-1] == "a short summary"


async def test_tokens_reports_the_current_context(make_context, make_update, monkeypatch):
    context = make_context("unused")

    async def count():
        context.chat_data["token_count"] = {"tokens": 162, "exact": True}
        return 162

    monkeypatch.setattr(context, "count_context_tokens", count)
    monkeypatch.setattr(context, "get_context_length", AsyncMock(return_value=1000))
    update = make_update(text="/tokens", username="ex0dus")

    await tokens_cmd(update, context)

    assert update.message.texts("reply_text") == ["The current context is 162 tokens of 1000 (16.2%)."]


async def test_tokens_labels_estimates(make_context, make_update, monkeypatch):
    context = make_context("unused")

    async def count():
        context.chat_data["token_count"] = {"tokens": 40, "exact": False}
        return 40

    monkeypatch.setattr(context, "count_context_tokens", count)
    monkeypatch.setattr(context, "get_context_length", AsyncMock(return_value=1000))
    update = make_update(text="/tokens", username="ex0dus")

    await tokens_cmd(update, context)

    assert update.message.texts("reply_text") == ["The current context is 40 tokens (estimated) of 1000 (4.0%)."]


async def test_tokens_omits_the_budget_when_the_server_does_not_report_one(make_context, make_update, monkeypatch):
    context = make_context("unused")

    async def count():
        context.chat_data["token_count"] = {"tokens": 40, "exact": True}
        return 40

    monkeypatch.setattr(context, "count_context_tokens", count)
    monkeypatch.setattr(context, "get_context_length", AsyncMock(return_value=None))
    update = make_update(text="/tokens", username="ex0dus")

    await tokens_cmd(update, context)

    assert update.message.texts("reply_text") == ["The current context is 40 tokens."]


async def test_tokens_is_admin_only(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/tokens", username="random")

    await tokens_cmd(update, context)

    assert update.message.texts("reply_text") == ["You are not authorized to use this command."]


async def test_tokens_reports_count_failures(make_context, make_update, monkeypatch):
    context = make_context("unused")

    async def count():
        raise RuntimeError("server down")

    monkeypatch.setattr(context, "count_context_tokens", count)
    update = make_update(text="/tokens", username="ex0dus")

    await tokens_cmd(update, context)

    assert update.message.texts("reply_text") == ["Could not count tokens. Check the logs for details."]


async def test_characterize_replaces_system_prompt(make_context, make_update):
    context = make_context("You are a pirate.")
    context.chat_message_history.append(HumanMessage(content="old"))
    update = make_update(text="/characterize a pirate")

    await characterize_cmd(update, context)

    assert "You are a pirate." in update.message.texts("reply_text")[0]
    assert list(context.chat_message_history) == []
    assert context.chat_system_prompt == "You are a pirate."


async def test_characterize_handles_content_blocks(make_context, make_update):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = GenericFakeChatModel(messages=iter([AIMessage(content=[{"type": "text", "text": "You are a pirate."}])]))
    context = make_context(model=model)
    update = make_update(text="/characterize a pirate")

    await characterize_cmd(update, context)

    assert context.chat_system_prompt == "You are a pirate."


async def test_characterize_rejects_a_refusal(make_context, make_update):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = GenericFakeChatModel(
        messages=iter([AIMessage(content=[{"type": "refusal", "refusal": "I cannot do that."}])])
    )
    context = make_context(model=model)
    context.chat_system_prompt = "original"
    update = make_update(text="/characterize a pirate")

    await characterize_cmd(update, context)

    assert context.chat_system_prompt == "original"
    assert "I cannot do that." in update.message.texts("reply_text")[0]


async def test_characterize_rejects_an_empty_response(make_context, make_update):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = GenericFakeChatModel(messages=iter([AIMessage(content="")]))
    context = make_context(model=model)
    context.chat_system_prompt = "original"
    update = make_update(text="/characterize a pirate")

    await characterize_cmd(update, context)

    assert context.chat_system_prompt == "original"
    assert "empty" in update.message.texts("reply_text")[0]


async def test_characterize_without_prompt(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/characterize")

    await characterize_cmd(update, context)

    assert update.message.texts("reply_text") == ["No prompt provided."]
    assert context.bot_llm.call_count == 0
