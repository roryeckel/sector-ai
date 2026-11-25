"""Tests for the shared network-retry helper."""

from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from sector_ai.retry import retry_on_network_error


async def test_bad_request_is_not_retried(monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    calls = AsyncMock(side_effect=BadRequest("not modified"))

    with pytest.raises(BadRequest):
        await retry_on_network_error(calls, what="thing")

    assert calls.await_count == 1


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


async def test_flood_control_waits_and_then_succeeds(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", sleep)
    calls = AsyncMock(side_effect=[RetryAfter(10), "ok"])

    assert await retry_on_network_error(calls, what="thing") == "ok"
    assert calls.await_count == 2
    sleep.assert_awaited_once_with(10.0)


async def test_flood_control_gives_up_after_the_final_attempt(monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    calls = AsyncMock(side_effect=RetryAfter(10))

    assert await retry_on_network_error(calls, what="thing") is None
    assert calls.await_count == 3


async def test_a_flood_window_longer_than_the_cap_fails_without_retrying_early(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", sleep)
    calls = AsyncMock(side_effect=RetryAfter(120))

    assert await retry_on_network_error(calls, what="thing") is None
    assert calls.await_count == 1
    sleep.assert_not_awaited()


async def test_other_exceptions_propagate():
    calls = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await retry_on_network_error(calls, what="thing")

    assert calls.await_count == 1
