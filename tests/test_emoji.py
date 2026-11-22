"""Tests for /emoji."""

from telegram.constants import MessageLimit

from sector_ai.emoji import emoji_cmd


async def test_emoji_sends_response(make_context, make_update):
    context = make_context("🎉🥳")
    update = make_update(text="/emoji party")

    await emoji_cmd(update, context)

    assert update.message.texts("reply_text") == ["🎉🥳"]


async def test_emoji_prompt_excludes_command_word(make_context, make_update):
    context = make_context("🎉")
    update = make_update(text="/emoji a big party")

    await emoji_cmd(update, context)

    sent = context.bot_llm.received[0][0].content
    assert "a big party" in sent
    assert "/emoji" not in sent


async def test_emoji_truncates_to_telegram_limit(make_context, make_update):
    context = make_context("x" * (MessageLimit.MAX_TEXT_LENGTH + 500))
    update = make_update(text="/emoji spam")

    await emoji_cmd(update, context)

    assert len(update.message.texts("reply_text")[0]) == MessageLimit.MAX_TEXT_LENGTH


async def test_emoji_reports_empty_response(make_context, make_update):
    context = make_context("")
    update = make_update(text="/emoji party")

    await emoji_cmd(update, context)

    assert update.message.texts("reply_text") == ["The model returned an empty response."]


async def test_emoji_without_prompt(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/emoji")

    await emoji_cmd(update, context)

    assert update.message.texts("reply_text") == ["Please enter some text to turn into emojis"]
    assert context.bot_llm.call_count == 0
