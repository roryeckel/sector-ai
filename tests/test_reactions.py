"""Reaction targeting, emoji normalization, and honest delivery outcomes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from telegram.error import BadRequest, NetworkError

from sector_ai.capabilities import builtin_tools
from sector_ai.reactions import (
    ALLOWED_REACTIONS,
    canonical_reaction,
    last_bot_message_id,
    reaction_tool,
    resolve_target,
)


def test_canonical_reaction_accepts_common_variation_selectors():
    assert canonical_reaction("👍") == "👍"
    assert canonical_reaction(" ❤️ ") == "\u2764"
    assert canonical_reaction("☃️") == "\u2603"
    assert canonical_reaction("❤‍🔥") == "❤️‍🔥"
    assert canonical_reaction("🦄🦄") is None
    assert canonical_reaction("") is None
    assert canonical_reaction(None) is None


def test_resolve_target_defaults_to_the_current_message(make_context, make_update):
    context = make_context()
    update = make_update(text="hello", message_id=77)

    assert resolve_target(context, update, {}) == 77
    assert resolve_target(context, update, {"target": "current"}) == 77


def test_resolve_target_accepts_an_explicit_message_id(make_context, make_update):
    context = make_context()
    update = make_update(text="hello", message_id=77)

    assert resolve_target(context, update, {"message_id": 55}) == 55
    with pytest.raises(ValueError, match="not both"):
        resolve_target(context, update, {"message_id": 55, "target": "current"})
    with pytest.raises(ValueError, match="positive integer"):
        resolve_target(context, update, {"message_id": 0})
    with pytest.raises(ValueError, match="positive integer"):
        resolve_target(context, update, {"message_id": True})


def test_resolve_target_uses_the_newest_bot_text(make_context, make_update):
    context = make_context()
    context.chat_message_history.extend(
        [
            AIMessage(
                content="older",
                additional_kwargs={"telegram": {"message_id": 41, "sender": {"is_bot": True}}},
            ),
            AIMessage(
                content="newest",
                additional_kwargs={"telegram": {"message_id": 43, "sender": {"is_bot": True}}},
            ),
            AIMessage(
                content="a person",
                additional_kwargs={"telegram": {"message_id": 44, "sender": {"id": 99}}},
            ),
        ]
    )
    update = make_update(text="hello", message_id=77)

    assert last_bot_message_id(context) == 43
    assert resolve_target(context, update, {"target": "last_bot"}) == 43


def test_resolve_target_last_bot_requires_an_earlier_bot_message(make_context, make_update):
    context = make_context()
    update = make_update(text="hello", message_id=77)

    assert last_bot_message_id(context) is None
    with pytest.raises(ValueError, match="earlier bot message"):
        resolve_target(context, update, {"target": "last_bot"})


def test_resolve_target_uses_the_replied_to_message(make_context, make_update):
    context = make_context()
    update = make_update(text="hello", message_id=77, reply_to_message=SimpleNamespace(message_id=33))

    assert resolve_target(context, update, {"target": "reply_to"}) == 33
    with pytest.raises(ValueError, match="not a reply"):
        resolve_target(context, make_update(text="hello", message_id=78), {"target": "reply_to"})
    with pytest.raises(ValueError, match="Unknown reaction target"):
        resolve_target(context, update, {"target": "previous"})


async def test_reaction_tool_defaults_to_the_current_message(make_context, make_update):
    context = make_context()
    update = make_update(text="react", message_id=77)
    tool = reaction_tool(update, context)

    result = await tool.invoke({"emoji": "👍"})

    assert result["status"] == "delivered"
    assert result["reaction"] == {"emoji": "👍", "message_id": 77, "status": "delivered"}
    kwargs = context.bot.set_message_reaction.await_args.kwargs
    assert kwargs["chat_id"] == 1
    assert kwargs["message_id"] == 77
    assert kwargs["reaction"][0].emoji == "👍"


async def test_reaction_tool_accepts_an_explicit_past_message(make_context, make_update):
    context = make_context()
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    result = await tool.invoke({"emoji": "🔥", "message_id": 55})

    assert result["status"] == "delivered"
    assert context.bot.set_message_reaction.await_args.kwargs["message_id"] == 55


async def test_reaction_tool_rejects_an_unsupported_emoji(make_context, make_update):
    context = make_context()
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    result = await tool.invoke({"emoji": "not-an-emoji"})

    assert result["status"] == "error"
    assert "standard reaction" in result["error"]
    context.bot.set_message_reaction.assert_not_awaited()


async def test_reaction_tool_reports_a_rejected_target(make_context, make_update):
    context = make_context()
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    result = await tool.invoke({"emoji": "👍", "message_id": 55, "target": "current"})

    assert result["status"] == "error"
    assert "not both" in result["error"]
    context.bot.set_message_reaction.assert_not_awaited()


async def test_reaction_tool_reports_a_telegram_rejection(make_context, make_update):
    context = make_context()
    context.bot.set_message_reaction = AsyncMock(side_effect=BadRequest("reactions are disabled"))
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    result = await tool.invoke({"emoji": "👍"})

    assert result["status"] == "delivery_failed"
    assert "reactions are disabled" in result["reaction"]["error"]


async def test_reaction_tool_retries_a_network_failure_once(make_context, make_update):
    context = make_context()
    context.bot.set_message_reaction = AsyncMock(side_effect=[NetworkError("lost"), True])
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    result = await tool.invoke({"emoji": "👍"})

    assert result["status"] == "delivered"
    assert context.bot.set_message_reaction.await_count == 2


async def test_reaction_tool_never_claims_an_uncertain_reaction(make_context, make_update):
    context = make_context()
    context.bot.set_message_reaction = AsyncMock(side_effect=NetworkError("lost"))
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    result = await tool.invoke({"emoji": "👍"})

    assert result["status"] == "delivery_uncertain"
    assert context.bot.set_message_reaction.await_count == 2


async def test_reaction_tool_reports_a_false_confirmation(make_context, make_update):
    context = make_context()
    context.bot.set_message_reaction = AsyncMock(return_value=False)
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    result = await tool.invoke({"emoji": "👍"})

    assert result["status"] == "delivery_failed"
    assert "did not confirm" in result["reaction"]["error"]


async def test_reaction_tool_records_an_interrupted_delivery(make_context, make_update):
    context = make_context()
    context.conversation_data["active_tool_call_id"] = "call-1"
    context.bot.set_message_reaction = AsyncMock(side_effect=asyncio.CancelledError())
    tool = reaction_tool(make_update(text="react", message_id=77), context)

    with pytest.raises(asyncio.CancelledError):
        await tool.invoke({"emoji": "👍"})

    interrupted = context.conversation_data["interrupted_delivery"]
    assert interrupted["tool_call_id"] == "call-1"
    assert interrupted["result"]["status"] == "delivery_uncertain"


def test_builtin_tools_expose_the_reaction_without_text(make_context, make_update):
    context = make_context()
    specs = builtin_tools(make_update(text="react", message_id=77), context, context.bot_llm)

    assert "react_to_message" in {tool.name for tool in specs}
    reaction = next(tool for tool in specs if tool.name == "react_to_message")
    parameters = reaction.schema["function"]["parameters"]
    assert parameters["required"] == ["emoji"]
    assert parameters["properties"]["emoji"]["enum"] == list(ALLOWED_REACTIONS)
    assert "text" not in parameters["properties"]
