"""Tests for the shared network-retry helper."""

from unittest.mock import AsyncMock

import pytest
from telegram.error import NetworkError, TimedOut

from sector_ai.retry import retry_on_network_error


async def test_retries_network_errors_until_success(monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    calls = AsyncMock(side_effect=[NetworkError("502"), TimedOut(), "ok"])

    assert await retry_on_network_error(calls, what="thing") == "ok"
    assert calls.await_count == 3


async def test_gives_up_after_the_final_attempt_and_returns_none(monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    calls = AsyncMock(side_effect=NetworkError("still down"))

    assert await retry_on_network_error(calls, what="thing") is None
    assert calls.await_count == 3


async def test_other_exceptions_propagate():
    calls = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await retry_on_network_error(calls, what="thing")

    assert calls.await_count == 1
