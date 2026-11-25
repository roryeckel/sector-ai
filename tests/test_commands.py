"""Tests for decision, autoreply, and poll commands."""

import json
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from telegram.constants import PollLimit

from sector_ai.autoreply import should_respond
from sector_ai.decision import Decision, decide_cmd, make_decision
from sector_ai.poll import poll_cmd


@pytest.fixture(autouse=True)
def local_context_budget(monkeypatch):
    monkeypatch.setattr("sector_ai.sector_context.SectorContext.get_context_budget", AsyncMock(return_value=8192))


# --- decide_cmd ---


async def test_decide_no_prompt(mock_update, mock_context):
    update = mock_update(text="/decide", username="testuser")
    await decide_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once_with("No prompt provided.")


async def test_decide_with_prompt(mock_update, mock_context):
    update = mock_update(text="/decide is sky blue?", username="testuser")
    with patch("sector_ai.decision.make_decision", new_callable=AsyncMock, return_value=True):
        await decide_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once_with("True")


# --- make_decision ---


async def test_make_decision_returns_bool(mock_context):
    mock_context.get_system_template_dict = AsyncMock(return_value={"MAX_LEN": 4096, "USERNAME": "testbot"})
    mock_context.bind_cache = lambda model, workload: model
    mock_context.bot_llm.ainvoke = AsyncMock(return_value=AIMessage(content="true"))

    with patch("sector_ai.decision.PydanticOutputParser") as parser_class:
        parser_class.return_value.invoke.return_value = Decision(result=True)
        result = await make_decision("test prompt", mock_context)

    assert result is True


async def test_make_decision_returns_string_on_parse_error(mock_context):
    mock_context.get_system_template_dict = AsyncMock(return_value={"MAX_LEN": 4096, "USERNAME": "testbot"})
    mock_context.bind_cache = lambda model, workload: model
    mock_context.bot_llm.ainvoke = AsyncMock(return_value=AIMessage(content="garbage"))

    with patch("sector_ai.decision.PydanticOutputParser") as parser_class:
        parser_class.return_value.invoke.side_effect = OutputParserException("bad", llm_output="garbage")
        result = await make_decision("test prompt", mock_context)

    assert result == "garbage"


# --- should_respond ---


async def test_should_respond_returns_none_when_save_fails(mock_update, mock_context):
    update = mock_update(text="/clear", username="testuser")
    mock_context.save_user_message.return_value = False
    result = await should_respond(update, mock_context)
    assert result is None


async def test_should_respond_returns_none_when_autoreply_disabled(mock_update, mock_context):
    update = mock_update(text="hello", username="testuser")
    mock_context.save_user_message.return_value = True
    type(mock_context).chat_autoreply_mode = PropertyMock(return_value=False)
    result = await should_respond(update, mock_context)
    assert result is None


# --- poll ---


async def test_poll_no_prompt(mock_update, mock_context):
    update = mock_update(text="/poll", username="testuser")
    await poll_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once_with("No prompt provided.")


async def test_poll_validates_option_count(make_context, make_update):
    context = make_context(json.dumps({"question": "Test?", "options": []}))
    update = make_update(text="/poll make a poll")
    await poll_cmd(update, context)
    assert update.message.calls_to("reply_poll") == []
    assert json.loads(context.chat_message_history[-1].content)["status"] == "error"


# --- poll end-to-end with a fake model ---


async def test_poll_sends_parsed_poll(make_context, make_update):
    payload = json.dumps({"question": "Pick one", "options": ["alpha", "beta", "gamma"]})
    context = make_context(payload)
    update = make_update(text="/poll pick something")

    await poll_cmd(update, context)

    question, options = update.message.calls_to("reply_poll")[0][0]
    assert question == "Pick one"
    assert options == ["alpha", "beta", "gamma"]
    assert update.message.calls_to("reply_poll")[0][1]["is_anonymous"] is False


async def test_poll_rejects_overlong_option(make_context, make_update):
    payload = json.dumps({"question": "Pick one", "options": ["x" * (PollLimit.MAX_OPTION_LENGTH + 1), "beta"]})
    context = make_context(payload)
    update = make_update(text="/poll pick something")

    await poll_cmd(update, context)

    assert update.message.calls_to("reply_poll") == []
    assert json.loads(context.chat_message_history[-1].content)["status"] == "error"
