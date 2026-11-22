"""Tests for the autoreply gate and its toggle."""

from unittest.mock import AsyncMock

from telegram.error import NetworkError

from sector_ai import autoreply as autoreply_module
from sector_ai.autoreply import autoreply_cmd, should_respond


async def test_should_respond_records_message_and_returns_decision(make_context, make_update, monkeypatch):
    monkeypatch.setattr(autoreply_module, "make_decision", AsyncMock(return_value=True))
    context = make_context("unused")
    update = make_update(text="hello there")

    assert await should_respond(update, context) is True
    assert context.chat_message_history[-1].content == "testuser: hello there"


async def test_should_respond_skips_decision_when_disabled(make_context, make_update, monkeypatch):
    decision = AsyncMock(return_value=True)
    monkeypatch.setattr(autoreply_module, "make_decision", decision)
    context = make_context("unused")
    context.chat_autoreply_mode = False
    update = make_update(text="hello there")

    assert await should_respond(update, context) is None
    decision.assert_not_awaited()


async def test_should_respond_retries_network_errors(make_context, make_update, monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    decision = AsyncMock(side_effect=[NetworkError("502 Bad Gateway"), True])
    monkeypatch.setattr(autoreply_module, "make_decision", decision)
    context = make_context("unused")
    update = make_update(text="hello there")

    assert await should_respond(update, context) is True

    assert decision.await_count == 2
    # The user message is recorded once, not once per retry.
    assert sum(m.content == "testuser: hello there" for m in context.chat_message_history) == 1


async def test_should_respond_gives_up_after_repeated_network_errors(make_context, make_update, monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    decision = AsyncMock(side_effect=NetworkError("still down"))
    monkeypatch.setattr(autoreply_module, "make_decision", decision)
    context = make_context("unused")
    update = make_update(text="hello there")

    assert await should_respond(update, context) is False
    assert decision.await_count == 3


async def test_should_respond_treats_parser_errors_as_no(make_context, make_update, monkeypatch):
    monkeypatch.setattr(autoreply_module, "make_decision", AsyncMock(return_value="garbage output"))
    context = make_context("unused")
    update = make_update(text="hello there")

    assert await should_respond(update, context) is False


async def test_should_respond_swallows_decision_errors(make_context, make_update, monkeypatch):
    monkeypatch.setattr(autoreply_module, "make_decision", AsyncMock(side_effect=RuntimeError("boom")))
    context = make_context("unused")
    update = make_update(text="hello there")

    assert await should_respond(update, context) is False


async def test_should_respond_ignores_bare_commands(make_context, make_update, monkeypatch):
    decision = AsyncMock(return_value=True)
    monkeypatch.setattr(autoreply_module, "make_decision", decision)
    context = make_context("unused")
    update = make_update(text="/clear")

    assert await should_respond(update, context) is None
    decision.assert_not_awaited()
    assert list(context.chat_message_history) == []


async def test_autoreply_cmd_toggles_both_ways(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/autoreply")

    await autoreply_cmd(update, context)
    assert context.chat_autoreply_mode is False
    assert "disabled" in update.message.texts("reply_text")[0]

    await autoreply_cmd(update, context)
    assert context.chat_autoreply_mode is True
    assert "enabled" in update.message.texts("reply_text")[1]
