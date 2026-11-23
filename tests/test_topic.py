"""Tests for /topic: structured output, poll payload and format flag reset."""

import pytest
from langchain_core.messages import HumanMessage
from telegram.constants import PollLimit

from sector_ai.topic import topic_poll_cmd


@pytest.fixture(autouse=True)
def local_context_budget(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr("sector_ai.sector_context.SectorContext.get_context_budget", AsyncMock(return_value=8192))


def topic_json(*topics, title="Chat topics"):
    import json

    return json.dumps({"title": title, "topics": list(topics)})


async def test_topic_sends_poll(make_context, make_update):
    context = make_context(topic_json("alpha", "beta", "gamma"))
    update = make_update(text="/topic")

    await topic_poll_cmd(update, context)

    question, options = update.message.calls_to("reply_poll")[0][0]
    assert question == "Chat topics"
    assert options == ["alpha", "beta", "gamma"]
    assert update.message.calls_to("reply_poll")[0][1]["is_anonymous"] is False


async def test_topic_requests_json_mode(make_context, make_update):
    context = make_context(topic_json("alpha", "beta"))
    update = make_update(text="/topic")

    await topic_poll_cmd(update, context)

    assert context.bot_llm.received_kwargs[-1]["response_format"] == {"type": "json_object"}


async def test_topic_prompt_includes_task_when_template_has_no_query_placeholder(make_context, make_update):
    context = make_context(topic_json("alpha", "beta"))
    update = make_update(text="/topic")

    await topic_poll_cmd(update, context)

    sent = context.bot_llm.received[0]
    assert "Create a poll about the recent conversation." in "\n".join(message.content for message in sent)
    assert isinstance(sent[-1], HumanMessage)
    assert "current_time_utc" in sent[-2].content


async def test_topic_rejects_too_few_topics(make_context, make_update):
    context = make_context(topic_json(*["only"] * (PollLimit.MIN_OPTION_NUMBER - 1)))
    update = make_update(text="/topic")

    await topic_poll_cmd(update, context)
    assert update.message.calls_to("reply_poll") == []
    assert context.chat_message_history[-1].status == "error"


async def test_topic_rejects_overlong_topic(make_context, make_update):
    context = make_context(topic_json("a" * (PollLimit.MAX_OPTION_LENGTH + 1), "beta"))
    update = make_update(text="/topic")

    await topic_poll_cmd(update, context)
    assert update.message.calls_to("reply_poll") == []
    assert context.chat_message_history[-1].status == "error"


async def test_topic_falls_back_to_default_title(make_context, make_update):
    context = make_context(topic_json("alpha", "beta", title=""))
    update = make_update(text="/topic")

    await topic_poll_cmd(update, context)

    question, _ = update.message.calls_to("reply_poll")[0][0]
    assert question == "Topics"
