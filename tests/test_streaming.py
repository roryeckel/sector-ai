"""Tests for the streaming message updater."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessageChunk
from telegram.error import BadRequest, NetworkError

from sector_ai.streaming_handler import LazyReply, handle_streaming_response


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
    assert message.texts("edit_text")[-1].startswith("Error processing chat: Something went wrong (reference ")


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
    assert "message_thread_id" not in context.bot.send_message.await_args.kwargs


async def test_successful_delivery_keeps_tracking_until_the_turn_finishes(
    make_context, recording_message, quiet_settings
):
    context = make_context(settings=quiet_settings)
    context.conversation_data["pending_placeholder"] = {"chat_id": 1, "message_id": 42}
    message = recording_message()

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("Hello")), "Chat")

    assert response == "Hello"
    # A tool loop may run another model call, so a mid-turn edit must not
    # forget which placeholder belongs to this turn.
    assert context.conversation_data["pending_placeholder"]["message_id"] == 42
    assert "placeholder_unsettled" not in context.conversation_data

    context.finish_placeholder()
    assert "pending_placeholder" not in context.conversation_data


async def test_a_later_failed_send_never_marks_an_earlier_delivered_reply(
    make_context, recording_message, quiet_settings
):
    # Reply 1 is delivered and settled. Reply 2 fails on its first send, so no
    # message exists for it. Nothing may be deleted at the next turn.
    context = make_context(settings=quiet_settings)
    first_source = recording_message()
    first_source.message_id = 100
    await handle_streaming_response(context, LazyReply(context, first_source), aiter_chunks(chunks("First")), "Chat")
    second = LazyReply(context, recording_message(fail_times=1))

    response = await handle_streaming_response(context, second, aiter_chunks(chunks("Second")), "Chat")
    context.finish_placeholder()
    await context.remove_pending_placeholder()

    assert response == ""
    assert second.failed is True
    context.bot.delete_message.assert_not_awaited()


async def test_failed_delivery_keeps_pending_placeholder_for_cleanup(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    context.conversation_data["pending_placeholder"] = {"chat_id": 1, "message_id": 42}
    message = recording_message(fail_times=1)
    message.message_id = 42

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("Hello")), "Chat")

    assert response == ""
    assert context.conversation_data["pending_placeholder"]["message_id"] == 42
    assert context.conversation_data["placeholder_unsettled"] == 42

    context.finish_placeholder()
    assert context.conversation_data["pending_placeholder"]["message_id"] == 42


async def test_a_failed_edit_of_an_untracked_message_gets_its_own_cleanup_slot(
    make_context, recording_message, quiet_settings
):
    context = make_context(settings=quiet_settings)
    context.conversation_data["pending_placeholder"] = {"chat_id": 1, "message_id": 42}
    message = recording_message(fail_times=1)
    message.message_id = 43

    await handle_streaming_response(context, message, aiter_chunks(chunks("Hello")), "Chat")
    context.finish_placeholder()
    await context.remove_pending_placeholder()

    deleted = [call.kwargs["message_id"] for call in context.bot.delete_message.await_args_list]
    assert deleted == [43]


async def test_remove_pending_placeholder_deletes_the_message(make_context):
    context = make_context()
    context.conversation_data["pending_placeholder"] = {"chat_id": 1, "message_id": 42}
    context.conversation_data["placeholder_unsettled"] = True

    await context.remove_pending_placeholder()

    context.bot.delete_message.assert_awaited_once_with(chat_id=1, message_id=42)
    assert "pending_placeholder" not in context.conversation_data
    assert "placeholder_unsettled" not in context.conversation_data


async def test_a_later_reply_does_not_forget_an_earlier_cursor_left_behind(make_context):
    context = make_context()
    context.remember_placeholder(SimpleNamespace(message_id=42))
    context.mark_placeholder_unsettled(SimpleNamespace(message_id=42))  # its final edit failed

    context.remember_placeholder(SimpleNamespace(message_id=43))
    context.finish_placeholder()
    await context.remove_pending_placeholder()

    deleted = [call.kwargs["message_id"] for call in context.bot.delete_message.await_args_list]
    assert deleted == [42]
    assert "abandoned_placeholders" not in context.conversation_data


async def test_streaming_records_provider_cache_usage(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = recording_message()
    chunk = AIMessageChunk(
        content="hi",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 1,
            "total_tokens": 11,
            "input_token_details": {"cache_read": 6},
        },
    )

    response = await handle_streaming_response(context, message, aiter_chunks([chunk]), "Chat")

    assert response == "hi"
    entry = context.bot_data["usage_stats"][("fake-model", "Chat")]
    assert entry["requests"] == 1
    assert entry["cache_read"] == 6


async def test_leading_telegram_envelope_is_not_delivered(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = recording_message()
    envelope = '[telegram {"message_id":1,"sender":{"is_bot":true}}]\n'

    response = await handle_streaming_response(context, message, aiter_chunks(chunks(envelope, "Hello")), "Chat")

    assert response == "Hello"
    assert all("[telegram " not in text for text in message.texts("edit_text"))


async def test_leading_telegram_envelope_is_hidden_during_preview(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    message = recording_message()
    envelope = '[telegram {"message_id":1,"sender":{"is_bot":true}}]\n'

    response = await handle_streaming_response(context, message, aiter_chunks(chunks(envelope, "Hello")), "Chat")

    assert response == "Hello"
    assert all("[telegram " not in text for text in message.texts("edit_text"))


async def test_long_reply_continuations_stay_in_the_originating_topic(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    message = recording_message()
    message.message_thread_id = 42
    message.is_topic_message = True

    response = await handle_streaming_response(context, message, aiter_chunks(chunks("x" * 5000)), "Chat")

    assert response == "x" * 5000
    assert context.bot.send_message.await_count == 1
    assert context.bot.send_message.await_args.kwargs["message_thread_id"] == 42


async def test_a_reply_chain_thread_id_is_not_sent_back(make_context, recording_message, quiet_settings):
    # Replies in ordinary supergroups carry a thread ID that names the reply
    # chain, not a forum topic.
    context = make_context(settings=quiet_settings)
    message = recording_message()
    message.message_thread_id = 42
    message.is_topic_message = False

    await handle_streaming_response(context, message, aiter_chunks(chunks("x" * 5000)), "Chat")

    assert "message_thread_id" not in context.bot.send_message.await_args.kwargs


async def test_a_failed_follow_up_message_saves_only_the_delivered_part(
    make_context, recording_message, quiet_settings, monkeypatch
):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    context = make_context(settings=quiet_settings)
    context.bot.send_message = AsyncMock(side_effect=NetworkError("502"))
    reply = LazyReply(context, recording_message())

    response = await handle_streaming_response(context, reply, aiter_chunks(chunks("x" * 5000)), "Chat")

    assert response == "x" * 4096
    assert reply.failed is True
    assert context.bot.send_message.await_count == 3


async def test_a_rejected_formatted_follow_up_is_sent_plain(make_context, recording_message, quiet_settings):
    context = make_context(settings=quiet_settings)
    sent = []

    async def send_message(**kwargs):
        if kwargs.get("entities"):
            raise BadRequest("can't parse entities")
        sent.append(kwargs)
        return SimpleNamespace(message_id=7)

    context.bot.send_message = send_message
    reply = LazyReply(context, recording_message())

    response = await handle_streaming_response(
        context, reply, aiter_chunks(chunks("x" * 4090 + "\n\n**bold** tail")), "Chat"
    )

    assert reply.failed is False
    assert response.endswith("tail")
    assert sent and sent[-1]["entities"] is None


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


async def test_lazy_reply_is_posted_with_the_first_text(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    source = recording_message()
    source.message_id = 9
    cursor = context.settings.streaming_cursor
    reply = LazyReply(context, source)

    response = await handle_streaming_response(context, reply, aiter_chunks(chunks(" ", "\n", "Hello")), "Chat")

    assert response == "Hello"
    # Whitespace never produced a cursor-only message; the post carries text.
    assert source.texts("reply_text") == ["Hello" + cursor]
    assert source.texts("edit_text") == ["Hello"]
    assert reply.message is source
    assert context.conversation_data["pending_placeholder"]["message_id"] == 9


async def test_tool_only_stream_posts_nothing(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    source = recording_message()
    reply = LazyReply(context, source)

    # Tool-call chunks carry no text, but the update timer still fires.
    response = await handle_streaming_response(
        context, reply, aiter_chunks(chunks("", "", "")), "Agent", allow_empty=True
    )

    assert response == ""
    assert source.shown() == []
    assert reply.message is None


async def test_a_later_round_never_shows_a_bare_cursor(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    message = recording_message()
    cursor = context.settings.streaming_cursor
    reply = LazyReply(context, message=message)

    await handle_streaming_response(context, reply, aiter_chunks(chunks("", "", "Next")), "Agent", allow_empty=True)

    edits = message.texts("edit_text")
    assert cursor not in [text.strip() for text in edits]
    assert edits == ["Next" + cursor, "Next"]


async def test_an_error_before_any_text_is_posted_as_a_reply(make_context, recording_message):
    context = make_context()
    source = recording_message()

    async def exploding():
        raise RuntimeError("boom")
        yield  # pragma: no cover

    response = await handle_streaming_response(context, LazyReply(context, source), exploding(), "Chat")

    assert response == ""
    assert source.texts("reply_text")[0].startswith("Error processing chat: Something went wrong (reference ")


async def test_pending_update_cancelled_on_error(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    message = recording_message()

    async def exploding():
        yield SimpleNamespace(content="alpha")
        raise RuntimeError("boom")

    response = await handle_streaming_response(context, message, exploding(), "Chat")

    assert response == ""
    # The error message must survive; the pending cursor edit is cancelled.
    assert message.texts("edit_text")[-1].startswith("Error processing chat: Something went wrong")


async def test_a_notice_after_visible_text_is_posted_separately(make_context, recording_message):
    context = make_context()
    source = recording_message()
    reply = LazyReply(context, source)
    await reply.write("partial answer")

    assert await reply.notify("Error processing chat response: timed out") is True

    assert source.texts("reply_text") == ["partial answer", "Error processing chat response: timed out"]
    assert source.texts("edit_text") == []


async def test_a_notice_that_telegram_rejects_never_raises(make_context, recording_message):
    context = make_context()
    reply = LazyReply(context, recording_message(fail_times=1))

    assert await reply.notify("Error") is False


async def test_a_cancelled_turn_strips_the_cursor_instead_of_leaving_it(make_context, recording_message, fast_settings):
    # A run timeout cancels mid-stream. The preview must not keep the cursor.
    context = make_context(settings=fast_settings)
    message = recording_message()
    first_chunk = asyncio.Event()

    async def endless():
        yield SimpleNamespace(content="alpha ")
        first_chunk.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(handle_streaming_response(context, message, endless(), "Chat"))
    await asyncio.wait_for(first_chunk.wait(), timeout=5)
    while not message.texts("edit_text"):
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The preview carried the cursor; the last edit removed it again.
    assert message.shown()
    assert context.settings.streaming_cursor in message.shown()[0]
    assert context.settings.streaming_cursor not in message.shown()[-1]
    context.finish_placeholder()
    assert "pending_placeholder" not in context.conversation_data


async def test_an_error_after_delivery_never_resends_the_reply(
    make_context, recording_message, quiet_settings, monkeypatch
):
    context = make_context(settings=quiet_settings)
    context.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    source = recording_message()
    reply = LazyReply(context, source)
    monkeypatch.setattr(
        "sector_ai.streaming_handler.record_usage", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("usage"))
    )

    chunk = AIMessageChunk(
        content="x" * 5000,
        usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
    )

    response = await handle_streaming_response(context, reply, aiter_chunks([chunk]), "Chat")

    assert context.bot.send_message.await_count == 1
    assert response.replace("\n", "") == "x" * 5000
    assert "Something went wrong (reference " in source.texts("reply_text")[-1]


async def test_an_error_with_visible_text_returns_what_is_on_screen(make_context, recording_message, fast_settings):
    context = make_context(settings=fast_settings)
    source = recording_message()

    async def exploding():
        yield SimpleNamespace(content="alpha beta")
        # Let the preview reach the chat, then fail mid-stream.
        while not source.texts("reply_text"):
            await asyncio.sleep(0.001)
        raise RuntimeError("boom")

    response = await handle_streaming_response(context, LazyReply(context, source), exploding(), "Chat")

    assert response == "alpha beta"
    # The preview's cursor is replaced by the final partial text.
    assert source.texts("edit_text")[-1] == "alpha beta"
    assert "Something went wrong (reference " in source.texts("reply_text")[-1]


async def test_a_permanent_delete_failure_is_not_retried_forever(make_context):
    from telegram.error import Forbidden

    context = make_context()
    context.conversation_data["pending_placeholder"] = {"chat_id": 1, "message_id": 42}
    context.conversation_data["placeholder_unsettled"] = 42
    context.bot.delete_message = AsyncMock(side_effect=Forbidden("no rights"))

    for _ in range(3):
        await context.remove_pending_placeholder()

    assert context.bot.delete_message.await_count == 1
    assert "pending_placeholder" not in context.conversation_data


async def test_a_transient_delete_failure_is_retried_on_the_next_turn(make_context):
    from telegram.error import NetworkError

    context = make_context()
    context.conversation_data["pending_placeholder"] = {"chat_id": 1, "message_id": 42}
    context.bot.delete_message = AsyncMock(side_effect=NetworkError("502"))

    await context.remove_pending_placeholder()
    assert context.conversation_data["pending_placeholder"]["message_id"] == 42


async def test_a_network_error_after_delivery_keeps_the_delivered_text(
    make_context, recording_message, quiet_settings, monkeypatch
):
    context = make_context(settings=quiet_settings)
    context.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    source = recording_message()
    chunk = AIMessageChunk(
        content="x" * 5000,
        usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
    )
    monkeypatch.setattr(
        "sector_ai.streaming_handler.record_usage",
        lambda *args, **kwargs: (_ for _ in ()).throw(NetworkError("502")),
    )

    response = await handle_streaming_response(context, LazyReply(context, source), aiter_chunks([chunk]), "Chat")

    assert response.replace("\n", "") == "x" * 5000
    assert context.bot.send_message.await_count == 1
