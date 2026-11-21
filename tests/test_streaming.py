"""Tests for the streaming message updater."""

from types import SimpleNamespace

import pytest

from sector_ai.streaming_handler import handle_streaming_response


@pytest.fixture
def fast_settings(make_settings):
    return make_settings(streaming_interval_sec=0, streaming_chunk_size=0)


@pytest.fixture
def quiet_settings(make_settings):
    """Large thresholds: only the final flush touches Telegram."""
    return make_settings(streaming_interval_sec=999, streaming_chunk_size=999_999)


def chunks(*parts):
    return [SimpleNamespace(content=part) for part in parts]


async def aiter_chunks(items):
    """The handler consumes the async generator that ``astream`` returns."""
    for item in items:
        yield item


async def test_single_flush_when_thresholds_not_reached(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("alpha", " ", "beta")), "Chat")

    assert response == "alpha beta"
    assert message.texts("edit_text") == ["alpha beta"]


async def test_cursor_present_during_updates_absent_at_end(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    message = recording_message()
    cursor = context.settings.streaming_cursor

    await handle_streaming_response(context, message, aiter_chunks(chunks("alpha", " ", "beta")), "Chat")

    edits = message.texts("edit_text")
    assert len(edits) > 1
    assert all(text.endswith(cursor) for text in edits[:-1])
    assert edits[-1] == "alpha beta"


async def test_no_duplicate_consecutive_edits(make_context, recording_message, fast_settings):
    """Regression: the trailing flush used to rewrite unchanged text."""
    context = make_context(settings=fast_settings)
    message = recording_message()

    await handle_streaming_response(context, message, aiter_chunks(chunks("alpha", " ", "beta")), "Chat")

    edits = message.texts("edit_text")
    assert all(previous != current for previous, current in zip(edits, edits[1:]))


async def test_generator_error_is_reported_and_returns_empty(make_context, recording_message):
    context = make_context()
    message = recording_message()

    async def exploding():
        yield SimpleNamespace(content="partial")
        raise RuntimeError("boom")

    response = await handle_streaming_response(context, message, exploding(), "Chat")

    assert response == ""
    assert "Error processing chat: boom" in message.texts("edit_text")[-1]


async def test_edit_failure_stops_streaming_early(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    message = recording_message(fail_times=1)

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("alpha", " ", "beta")), "Chat")

    # Streaming stops at the first failed edit; the text received so far is then
    # flushed once more so the user sees a partial answer rather than a spinner.
    assert response.strip() == "alpha"
    assert len(message.calls_to("edit_text")) == 2


async def test_falsy_chunk_ends_stream(make_context, recording_message):
    context = make_context()
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks([None]), "Chat")

    assert response == ""
    assert message.calls_to("edit_text") == []


async def test_pending_update_cancelled_on_error(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    message = recording_message()

    async def exploding():
        yield SimpleNamespace(content="alpha")
        raise RuntimeError("boom")

    response = await handle_streaming_response(context, message, exploding(), "Chat")

    assert response == ""
    # The error message must survive; the pending cursor edit is cancelled.
    assert message.texts("edit_text")[-1] == "Error processing chat: boom"
