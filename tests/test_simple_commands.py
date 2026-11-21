"""Tests for the thin commands that delegate to handle_chat or the model."""

from langchain_core.messages import HumanMessage, SystemMessage

from sector_ai.characterize import characterize_cmd
from sector_ai.summarize import summarize_cmd
from sector_ai.tokens import tokens_cmd


async def test_summarize_uses_summarization_prompt(make_context, make_update):
    context = make_context("a short summary")
    update = make_update(text="/summarize")

    await summarize_cmd(update, context)

    sent = context.bot_ollama.received[0]
    expected = context.config_summarization_system_prompt.format(MAX_LEN=4096, USERNAME="testbot")
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == expected
    assert update.message.texts("edit_text")[-1] == "a short summary"


async def test_tokens_reports_zero_for_empty_context(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/tokens")

    await tokens_cmd(update, context)

    assert update.message.texts("reply_text") == ["The current context is 0 tokens."]
    assert context.bot_ollama.call_count == 0


async def test_tokens_counts_existing_history(make_context, make_update):
    context = make_context("unused")
    context.chat_message_history.append(HumanMessage(content="hello world"))
    update = make_update(text="/tokens")

    await tokens_cmd(update, context)

    expected = context.bot_ollama.get_num_tokens_from_messages(context.chat_message_history)
    assert update.message.texts("reply_text") == [f"The current context is {expected} tokens."]


async def test_characterize_replaces_system_prompt(make_context, make_update):
    context = make_context("You are a pirate.")
    context.chat_message_history.append(HumanMessage(content="old"))
    update = make_update(text="/characterize a pirate")

    await characterize_cmd(update, context)

    assert "You are a pirate." in update.message.texts("reply_text")[0]
    assert list(context.chat_message_history) == []
    assert context.chat_system_prompt == "You are a pirate."


async def test_characterize_without_prompt(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/characterize")

    await characterize_cmd(update, context)

    assert update.message.texts("reply_text") == ["No prompt provided."]
    assert context.bot_ollama.call_count == 0
