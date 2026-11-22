"""Tests for the streaming message updater."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessageChunk
from telegram.error import BadRequest, NetworkError

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


async def test_a_failed_intermediate_edit_still_delivers_the_full_answer(
    make_context, recording_message, fast_settings
):
    context = make_context(settings=fast_settings)
    message = recording_message(fail_times=1)

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("alpha", " ", "beta")), "Chat")

    # The first edit fails, so updates pause while the rest of the stream is
    # consumed; the final flush then delivers the whole answer.
    assert response == "alpha beta"
    assert len(message.calls_to("edit_text")) == 2


async def test_when_every_edit_fails_the_answer_is_not_delivered(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    message = recording_message(fail_times=10)

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("alpha", " ", "beta")), "Chat")

    assert response == ""


async def test_network_error_from_an_edit_is_retried_then_returns_empty(make_context, quiet_settings, monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    context = make_context(settings=quiet_settings)
    message = SimpleNamespace(edit_text=AsyncMock(side_effect=NetworkError("502")))

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("hello")), "Chat")

    assert response == ""
    assert message.edit_text.await_count == 3


async def test_a_permanent_edit_failure_is_not_delivered(make_context, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = SimpleNamespace(edit_text=AsyncMock(side_effect=BadRequest("Message to edit not found")))

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("hello")), "Chat")

    assert response == ""


async def test_not_modified_edit_counts_as_delivered(make_context, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = SimpleNamespace(edit_text=AsyncMock(side_effect=BadRequest("Message is not modified")))

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("hello")), "Chat")

    assert response == "hello"


async def test_markdown_is_sent_as_entities(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("**bold**")), "Chat")

    assert response == "**bold**"
    assert message.texts("edit_text") == ["bold"]
    entities = message.calls_to("edit_text")[-1][1]["entities"]
    assert entities[0].type == "bold"


async def test_markdown_can_be_disabled(make_context, recording_message, make_settings):
    settings = make_settings(markdown=False, streaming_interval_sec=999, streaming_chunk_size=999_999)
    context = make_context(settings=settings)
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("**bold**")), "Chat")

    assert response == "**bold**"
    assert message.texts("edit_text") == ["**bold**"]
    assert message.calls_to("edit_text")[-1][1]["entities"] is None


class FormattingRejectingMessage:
    """Accepts plain text but rejects any styled edit, like Telegram would."""

    def __init__(self):
        self.chat_id = 1
        self.calls: list[tuple[str, dict]] = []

    async def edit_text(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if kwargs.get("entities"):
            raise BadRequest("Can't parse entities")
        return True


async def test_rejected_formatting_falls_back_to_plain(make_context, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = FormattingRejectingMessage()

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("**bold**")), "Chat")

    assert response == "**bold**"
    assert message.calls[0][0] == "bold"
    assert message.calls[0][1]["entities"]
    assert message.calls[-1] == ("bold", {"entities": None})


async def test_a_long_reply_is_split_across_messages(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("x" * 5000)), "Chat")

    assert response == "x" * 5000
    assert len(message.texts("edit_text")[0]) == 4096
    assert context.bot.send_message.await_count == 1
    assert len(context.bot.send_message.await_args.kwargs["text"]) == 904


async def test_a_failed_follow_up_message_is_not_saved(make_context, recording_message, quiet_settings, monkeypatch):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    context = make_context(settings=quiet_settings)
    context.bot.send_message = AsyncMock(side_effect=NetworkError("502"))
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("x" * 5000)), "Chat")

    assert response == ""
    assert context.bot.send_message.await_count == 3


async def test_falsy_chunk_ends_stream(make_context, recording_message):
    context = make_context()
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks([None]), "Chat")

    assert response == ""
    assert message.calls_to("edit_text") == []


async def test_streaming_handles_content_blocks(make_context, recording_message):
    context = make_context()
    message = recording_message()
    chunk = AIMessageChunk(content=[{"type": "text", "text": "hello"}])

    response = await handle_streaming_response(context, message, aiter_chunks([chunk]), "Chat")

    assert response == "hello"
    assert message.texts("edit_text") == ["hello"]


async def test_streaming_handles_refusal_blocks(make_context, recording_message):
    context = make_context()
    message = recording_message()
    chunk = AIMessageChunk(content=[{"type": "refusal", "refusal": "I cannot help with that."}])

    response = await handle_streaming_response(context, message, aiter_chunks([chunk]), "Chat")

    assert response == "I cannot help with that."
    assert message.texts("edit_text") == ["I cannot help with that."]


async def test_streaming_reports_an_empty_response(make_context, recording_message):
    context = make_context()
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks([AIMessageChunk(content="")]), "Chat")

    assert response == ""
    assert message.texts("edit_text")[-1] == "The model returned an empty response."


async def test_empty_stream_reports_no_response(make_context, recording_message):
    context = make_context()
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks([]), "Chat")

    assert response == ""
    assert message.texts("edit_text")[-1] == "The model returned an empty response."


async def test_streaming_reports_a_refusal_finish(make_context, recording_message):
    context = make_context()
    message = recording_message()
    chunk = AIMessageChunk(content="", response_metadata={"normalized_finish_reason": "refusal"})

    response = await handle_streaming_response(context, message, aiter_chunks([chunk]), "Chat")

    assert response == ""
    assert message.texts("edit_text")[-1] == "The model refused the request."


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
