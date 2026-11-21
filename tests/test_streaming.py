"""Tests for the streaming message updater."""

import copy
from types import SimpleNamespace

import pytest

from sector_ai.streaming_handler import handle_streaming_response


@pytest.fixture
def fast_config(sample_config):
    config = copy.deepcopy(sample_config)
    config["streaming"]["interval_sec"] = 0
    config["streaming"]["chunk_size"] = 0
    return config


@pytest.fixture
def quiet_config(sample_config):
    """Large thresholds: only the final flush touches Telegram."""
    config = copy.deepcopy(sample_config)
    config["streaming"]["interval_sec"] = 999
    config["streaming"]["chunk_size"] = 999_999
    return config


def chunks(*parts):
    return [SimpleNamespace(content=part) for part in parts]


async def test_single_flush_when_thresholds_not_reached(make_context, recording_message, quiet_config):
    context = make_context(config=quiet_config)
    message = recording_message()

    response = await handle_streaming_response(context, message, iter(chunks("alpha", " ", "beta")), "Chat")

    assert response == "alpha beta"
    assert message.texts("edit_text") == ["alpha beta"]


async def test_cursor_present_during_updates_absent_at_end(make_context, recording_message, fast_config):
    context = make_context(config=fast_config)
    message = recording_message()
    cursor = context.config_streaming_cursor

    await handle_streaming_response(context, message, iter(chunks("alpha", " ", "beta")), "Chat")

    edits = message.texts("edit_text")
    assert len(edits) > 1
    assert all(text.endswith(cursor) for text in edits[:-1])
    assert edits[-1] == "alpha beta"


async def test_no_duplicate_consecutive_edits(make_context, recording_message, fast_config):
    """Regression: the trailing flush used to rewrite unchanged text."""
    context = make_context(config=fast_config)
    message = recording_message()

    await handle_streaming_response(context, message, iter(chunks("alpha", " ", "beta")), "Chat")

    edits = message.texts("edit_text")
    assert all(previous != current for previous, current in zip(edits, edits[1:]))


async def test_generator_error_is_reported_and_returns_empty(make_context, recording_message):
    context = make_context()
    message = recording_message()

    def exploding():
        yield SimpleNamespace(content="partial")
        raise RuntimeError("boom")

    response = await handle_streaming_response(context, message, exploding(), "Chat")

    assert response == ""
    assert "Error processing chat: boom" in message.texts("edit_text")[-1]


async def test_edit_failure_stops_streaming_early(make_context, recording_message, fast_config):
    context = make_context(config=fast_config)
    message = recording_message(fail_times=1)

    response = await handle_streaming_response(context, message, iter(chunks("alpha", " ", "beta")), "Chat")

    # Streaming stops at the first failed edit; the text received so far is then
    # flushed once more so the user sees a partial answer rather than a spinner.
    assert response.strip() == "alpha"
    assert len(message.calls_to("edit_text")) == 2


async def test_falsy_chunk_ends_stream(make_context, recording_message):
    context = make_context()
    message = recording_message()

    response = await handle_streaming_response(context, message, iter([None]), "Chat")

    assert response == ""
    assert message.calls_to("edit_text") == []


async def test_pending_update_cancelled_on_error(make_context, recording_message, fast_config):
    context = make_context(config=fast_config)
    message = recording_message()

    def exploding():
        yield SimpleNamespace(content="alpha")
        raise RuntimeError("boom")

    response = await handle_streaming_response(context, message, exploding(), "Chat")

    assert response == ""
    # The error message must survive; the pending cursor edit is cancelled.
    assert message.texts("edit_text")[-1] == "Error processing chat: boom"
