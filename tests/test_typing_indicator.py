"""Tests for the typing indicator."""

import asyncio
from unittest.mock import AsyncMock

from telegram.constants import ChatAction
from telegram.error import NetworkError

from sector_ai import typing_indicator
from sector_ai.typing_indicator import typing


async def test_typing_is_sent_before_the_work_starts(recording_message):
    message = recording_message()

    async with typing(message):
        assert message.calls_to("reply_chat_action") == [((ChatAction.TYPING,), {})]


async def test_typing_is_refreshed_until_the_block_exits(recording_message, monkeypatch):
    monkeypatch.setattr(typing_indicator, "REFRESH_SEC", 0.01)
    message = recording_message()

    async with typing(message):
        # Wait for the refreshes themselves, not a fixed time: a loaded runner
        # may need longer than three intervals.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while message.typing_count() < 3:
            assert loop.time() < deadline, "typing was not refreshed"
            await asyncio.sleep(0.005)
    count = message.typing_count()
    # Several refresh intervals later, still nothing new: the loop stopped.
    await asyncio.sleep(0.05)

    assert message.typing_count() == count


async def test_a_failed_typing_action_never_fails_the_work():
    message = AsyncMock()
    message.reply_chat_action = AsyncMock(side_effect=NetworkError("down"))

    async with typing(message):
        result = "done"

    assert result == "done"
    message.reply_chat_action.assert_awaited()


async def test_typing_without_a_message_is_a_no_op():
    async with typing(None):
        pass
