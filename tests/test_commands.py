"""Tests for decision, autoreply, and poll commands."""

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from langchain_core.exceptions import OutputParserException
from telegram.constants import PollLimit

from sector_ai.autoreply import should_respond
from sector_ai.decision import Decision, decide_cmd, make_decision
from sector_ai.poll import BasicPoll, poll_cmd

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

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=Decision(result=True))

    with patch("sector_ai.decision.PydanticOutputParser"), patch("sector_ai.decision.PromptTemplate") as mock_template:
        mock_template.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
        result = await make_decision("test prompt", mock_context)

    assert result is True


async def test_make_decision_returns_string_on_parse_error(mock_context):
    mock_context.get_system_template_dict = AsyncMock(return_value={"MAX_LEN": 4096, "USERNAME": "testbot"})

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=OutputParserException("bad", llm_output="garbage"))

    with patch("sector_ai.decision.PydanticOutputParser"), patch("sector_ai.decision.PromptTemplate") as mock_template:
        mock_template.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
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


async def test_poll_validates_option_count(mock_update, mock_context):
    mock_context.get_system_template_dict = AsyncMock(return_value={"MAX_LEN": 4096, "USERNAME": "testbot"})

    # Chain returns a poll with fewer options than the API allows (Bot API 10.0 allows 1)
    bad_poll = BasicPoll(question="Test?", options=["Only one"] * (PollLimit.MIN_OPTION_NUMBER - 1))

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=bad_poll)

    update = mock_update(text="/poll make a poll", username="testuser")

    with patch("sector_ai.poll.PydanticOutputParser"), patch("sector_ai.poll.PromptTemplate") as mock_template:
        mock_template.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
        with pytest.raises(OutputParserException):
            await poll_cmd(update, mock_context)


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

    with pytest.raises(OutputParserException):
        await poll_cmd(update, context)

    assert update.message.calls_to("reply_poll") == []
