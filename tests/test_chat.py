"""Integration tests for /chat and the shared handle_chat path."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from telegram.error import NetworkError

from sector_ai.chat import chat_cmd, handle_chat


@pytest.fixture
def quiet_settings(make_settings):
    return make_settings(streaming_interval_sec=999, streaming_chunk_size=999_999)


class ExplodingModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "exploding"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("model unavailable")


async def test_handle_chat_streams_and_saves_history(make_context, make_update, quiet_settings):
    context = make_context("hello world", settings=quiet_settings)
    update = make_update(text="/chat hello")

    await handle_chat(update, context)

    message = update.message
    assert message.texts("reply_text") == ["Processing..."]
    assert message.texts("edit_text")[-1] == "hello world"
    assert [(type(m).__name__, m.content) for m in context.chat_message_history] == [("AIMessage", "hello world")]


async def test_handle_chat_strips_ai_prefix(make_context, make_update, quiet_settings):
    context = make_context("AI: the answer is 42", settings=quiet_settings)
    update = make_update(text="/chat question")

    await handle_chat(update, context)

    assert context.chat_message_history[-1].content == "the answer is 42"


async def test_handle_chat_does_not_save_empty_response(make_context, make_update, quiet_settings):
    context = make_context("", settings=quiet_settings)
    update = make_update(text="/chat question")

    await handle_chat(update, context)

    assert list(context.chat_message_history) == []


async def test_handle_chat_can_skip_saving(make_context, make_update, quiet_settings):
    context = make_context("summary text", settings=quiet_settings)
    update = make_update(text="/summarize")

    await handle_chat(update, context, save_ai_message=False)

    assert update.message.texts("edit_text")[-1] == "summary text"
    assert list(context.chat_message_history) == []


async def test_handle_chat_reports_model_error(make_context, make_update):
    context = make_context(model=ExplodingModel())
    update = make_update(text="/chat question")

    await handle_chat(update, context)

    assert "model unavailable" in update.message.texts("edit_text")[-1]


async def test_prompt_sent_to_model_has_expected_shape(make_context, make_update, quiet_settings):
    context = make_context("hi", settings=quiet_settings)
    context.chat_message_history.append(HumanMessage(name="alice", content="alice: earlier message"))
    update = make_update(text="/chat hello")

    await handle_chat(update, context)

    sent = context.bot_llm.received[0]
    assert [type(m) for m in sent] == [SystemMessage, SystemMessage, HumanMessage, SystemMessage]
    # Static instructions first, then retained history, then the snapshot.
    assert "Telegram chatbot" in sent[0].content
    assert "observed data" in sent[1].content
    assert sent[2].name == "alice"
    assert sent[2].content == "alice: earlier message"
    assert "current_time_utc" in sent[3].content


async def test_handle_chat_reports_setup_failure(make_context, make_update, monkeypatch):
    context = make_context("unused")
    monkeypatch.setattr(
        context,
        "get_system_template_dict",
        AsyncMock(side_effect=RuntimeError("telegram down")),
    )
    update = make_update(text="/chat question")

    await handle_chat(update, context)

    assert "Error processing chat response: telegram down" in update.message.texts("edit_text")[-1]


async def test_handle_chat_retries_after_a_network_error(make_context, make_update, quiet_settings, monkeypatch):
    monkeypatch.setattr("sector_ai.chat.asyncio.sleep", AsyncMock())
    context = make_context("hello again", settings=quiet_settings)
    # The outage in the incident hit get_me() inside get_system_template_dict.
    monkeypatch.setattr(
        context,
        "get_system_template_dict",
        AsyncMock(side_effect=[NetworkError("502 Bad Gateway"), {"USERNAME": "testbot", "MAX_LEN": 4000}]),
    )
    update = make_update(text="/chat hello")

    await handle_chat(update, context)

    assert update.message.texts("edit_text")[-1] == "hello again"
    assert context.chat_message_history[-1].content == "hello again"


async def test_handle_chat_lets_a_persistent_network_error_propagate(
    make_context, make_update, quiet_settings, monkeypatch
):
    monkeypatch.setattr("sector_ai.chat.asyncio.sleep", AsyncMock())
    context = make_context("hello", settings=quiet_settings)
    monkeypatch.setattr(context, "get_system_template_dict", AsyncMock(side_effect=NetworkError("still down")))
    update = make_update(text="/chat hello")

    with pytest.raises(NetworkError, match="still down"):
        await handle_chat(update, context)

    assert list(context.chat_message_history) == []


async def test_handle_chat_retries_the_placeholder(make_context, make_update, quiet_settings, monkeypatch):
    monkeypatch.setattr("sector_ai.chat.asyncio.sleep", AsyncMock())
    context = make_context("hello", settings=quiet_settings)
    update = make_update(text="/chat hi")
    posted = AsyncMock()
    posted.edit_text = AsyncMock()
    update.message.reply_text = AsyncMock(side_effect=[NetworkError("blip"), posted])

    await handle_chat(update, context)

    assert update.message.reply_text.await_count == 2
    assert posted.edit_text.await_args[0][0] == "hello"


async def test_chat_cmd_without_argument_prompts_for_text(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/chat")

    await chat_cmd(update, context)

    assert update.message.texts("reply_text") == ["Please enter your message to me."]
    assert list(context.chat_message_history) == []
    assert context.bot_llm.call_count == 0


async def test_chat_cmd_records_user_message(make_context, make_update, quiet_settings):
    context = make_context("the joke", settings=quiet_settings)
    update = make_update(text="/chat tell me a joke")

    await chat_cmd(update, context)

    assert isinstance(context.chat_message_history[0], HumanMessage)
    assert context.chat_message_history[0].content == "testuser: tell me a joke"
    assert isinstance(context.chat_message_history[1], AIMessage)


async def test_plain_reply_checks_actual_custom_prompt_budget(make_context, make_settings, make_update):
    context = make_context("unused", settings=make_settings(context_length=2000))
    update = make_update(text="/chat hi")
    await handle_chat(update, context, system_prompt="x" * 12000)
    assert context.bot_llm.call_count == 0
    assert "context window" in update.message.texts("edit_text")[-1]
