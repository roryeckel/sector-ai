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

    sent = "\n".join(message.content for message in context.bot_llm.received[0])
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


async def test_emoji_without_prompt_or_conversation_asks_for_text(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/emoji")

    await emoji_cmd(update, context)

    assert update.message.texts("reply_text") == ["Please enter some text to turn into emojis"]
    assert context.bot_llm.call_count == 0


async def test_emoji_without_prompt_uses_the_recent_conversation(make_context, make_update):
    context = make_context("🍕🔥")
    context.save_user_message(make_update(text="we are ordering pizza tonight", message_id=1).message)
    update = make_update(text="/emoji", message_id=2)

    await emoji_cmd(update, context)

    assert update.message.texts("reply_text") == ["🍕🔥"]
    sent = "\n".join(message.content for message in context.bot_llm.received[0])
    assert "we are ordering pizza tonight" in sent
    assert "the recent conversation above" in sent
    # The command itself is the request, not part of the context it describes.
    assert "/emoji" not in sent
    assert context.chat_message_history[-1].content == "🍕🔥"


async def test_emoji_mention_without_prompt_uses_the_conversation(make_context, make_update):
    context = make_context("🎈")
    context.save_user_message(make_update(text="it is a birthday", message_id=1).message)
    update = make_update(text="/emoji@testbot", message_id=2)

    await emoji_cmd(update, context)

    assert update.message.texts("reply_text") == ["🎈"]
    sent = "\n".join(message.content for message in context.bot_llm.received[0])
    assert "it is a birthday" in sent
    assert "@testbot" not in sent


async def test_emoji_without_prompt_drops_the_oldest_context_first(make_context, make_settings, make_update):
    context = make_context("🙂", settings=make_settings(context_length=2048))
    for index in range(6):
        context.save_user_message(make_update(text=f"message {index} " + "filler " * 200, message_id=index + 1).message)
    update = make_update(text="/emoji", message_id=99)

    await emoji_cmd(update, context)

    sent = "\n".join(message.content for message in context.bot_llm.received[0])
    assert "message 5" in sent
    assert "message 0" not in sent


async def test_emoji_without_prompt_reports_a_conversation_that_cannot_fit(make_context, make_settings, make_update):
    context = make_context("unused", settings=make_settings(context_length=90))
    context.save_user_message(make_update(text="hello " * 500, message_id=1).message)
    update = make_update(text="/emoji", message_id=2)

    await emoji_cmd(update, context)

    assert update.message.texts("reply_text") == ["The recent conversation is too large to turn into emojis."]
    assert context.bot_llm.call_count == 0
