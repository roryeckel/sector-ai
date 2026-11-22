"""Tests for context compaction and summary injection."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from telegram.error import NetworkError

from sector_ai.chat import chat_cmd, handle_chat


class ExplodingModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "exploding"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("model unavailable")


class RefusingModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "refusing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        refusal = AIMessage(content=[{"type": "refusal", "refusal": "I cannot summarize that."}])
        return ChatResult(generations=[ChatGeneration(message=refusal)])


def _fill(context, count: int, size: int = 0) -> None:
    suffix = "x" * size
    for i in range(count):
        context.chat_message_history.append(HumanMessage(name="alice", content=f"alice: message {i} {suffix}"))


async def test_short_history_is_not_compacted(make_context, monkeypatch):
    context = make_context("summary")
    _fill(context, 8)
    count = AsyncMock(return_value=10_000)
    monkeypatch.setattr(context, "count_context_tokens", count)

    assert await context.maybe_compact() is False
    count.assert_not_awaited()


async def test_a_small_history_still_resolves_the_window(make_context, monkeypatch):
    context = make_context("summary")
    _fill(context, 9)
    lookup = AsyncMock(return_value=262144)
    monkeypatch.setattr(context, "get_context_length", lookup)

    assert await context.maybe_compact() is False
    lookup.assert_awaited_once()


async def test_history_below_the_threshold_is_not_compacted(make_context, monkeypatch):
    context = make_context("summary")
    _fill(context, 10)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=100))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=50))

    assert await context.maybe_compact() is False
    assert len(context.chat_message_history) == 10
    assert context.context_summary == ""


async def test_old_messages_are_folded_into_the_summary(make_context, monkeypatch):
    context = make_context("Rory likes teal.")
    _fill(context, 12, size=200)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=1000))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=900))

    assert await context.maybe_compact() is True

    assert context.context_summary == "Rory likes teal."
    assert len(context.chat_message_history) == 8
    assert context.chat_message_history[-1].content.startswith("alice: message 11")
    assert context.chat_data["context_compaction"]["evicted"] == 4
    # The whole context still changes after a compaction.
    assert "token_count" not in context.chat_data

    sent = context.bot_llm.received[0]
    assert isinstance(sent[0], SystemMessage)
    assert "running memory" in sent[0].content
    assert "alice: message 0" in sent[1].content
    assert "alice: message 4" not in sent[1].content


async def test_compaction_merges_the_previous_summary(make_context, make_settings, monkeypatch):
    context = make_context("merged summary", settings=make_settings(compaction_keep_messages=2))
    context.context_summary = "Rory lives in Ohio."
    _fill(context, 4)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=8192))

    result = await context.compact()

    assert result["compacted"] is True
    assert result["evicted"] == 2
    assert len(context.chat_message_history) == 2
    sent = context.bot_llm.received[0]
    assert "Rory lives in Ohio." in sent[1].content
    assert "alice: message 1" in sent[1].content


async def test_an_oversized_transcript_is_summarized_in_chunks(make_context, make_settings, fake_model, monkeypatch):
    settings = make_settings(compaction_keep_messages=1)
    context = make_context(model=fake_model("first", "second"), settings=settings)
    for _ in range(5):
        context.chat_message_history.append(HumanMessage(content="x" * 3000))
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=4000))

    result = await context.compact()

    assert result["compacted"] is True
    assert result["summary"] == "second"
    assert context.bot_llm.call_count == 2
    assert "first" in context.bot_llm.received[1][1].content


def test_take_summary_chunk_stops_at_the_limit():
    from sector_ai.sector_context import SectorContext

    messages = [HumanMessage(content="x" * 40) for _ in range(5)]

    chunk, rest = SectorContext._take_summary_chunk(messages, limit=30)

    assert len(chunk) == 3
    assert rest == messages[3:]


def test_take_summary_chunk_splits_a_single_oversized_message():
    from sector_ai.sector_context import SectorContext

    chunk, rest = SectorContext._take_summary_chunk([HumanMessage(content="x" * 400)], limit=20)

    assert len(chunk) == 1
    assert len(chunk[0].content) == 80
    assert len(rest) == 4
    assert chunk[0].content + "".join(message.content for message in rest) == "x" * 400


async def test_a_single_message_cannot_be_compacted(make_context):
    context = make_context("summary")
    _fill(context, 1)

    result = await context.compact()

    assert result["compacted"] is False
    assert context.bot_llm.call_count == 0


async def test_summary_failure_keeps_the_transcript_when_it_fits(make_context, monkeypatch):
    context = make_context(model=ExplodingModel())
    _fill(context, 12, size=250)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=2000))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=1800))

    assert await context.maybe_compact() is False
    assert len(context.chat_message_history) == 12
    assert context.context_summary == ""


async def test_summary_failure_drops_messages_when_the_request_cannot_fit(make_context, monkeypatch):
    context = make_context(model=ExplodingModel())
    _fill(context, 12, size=300)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=1000))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=900))

    assert await context.maybe_compact() is False
    assert len(context.chat_message_history) < 12
    assert context.context_summary == ""


async def test_a_long_summary_is_truncated(make_context, make_settings, monkeypatch):
    settings = make_settings(compaction_keep_messages=2, compaction_summary_max_chars=200)
    context = make_context("x" * 500, settings=settings)
    _fill(context, 4)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=8192))

    await context.compact()

    assert len(context.context_summary) == 200


async def test_the_summary_is_injected_after_the_system_prompt(make_context, make_settings, _update):
    settings = make_settings(streaming_interval_sec=999, streaming_chunk_size=999_999)
    context = make_context("fresh reply", settings=settings)
    context.context_summary = "Rory likes teal."

    await chat_cmd(_update(), context)

    sent = context.bot_llm.received[0]
    assert [type(message) for message in sent] == [SystemMessage, SystemMessage, HumanMessage]
    assert sent[1].content == "Summary of the earlier conversation:\nRory likes teal."
    assert sent[2].content == "testuser: hello"


async def test_handle_chat_compacts_before_replying(make_context, make_settings, fake_model, monkeypatch, _update):
    settings = make_settings(streaming_interval_sec=999, streaming_chunk_size=999_999)
    context = make_context(model=fake_model("compacted summary", "fresh reply"), settings=settings)
    _fill(context, 12, size=200)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=1000))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=900))

    await handle_chat(_update(), context)

    assert context.context_summary == "compacted summary"
    assert len(context.chat_message_history) == 9  # 8 kept plus the new reply
    assert isinstance(context.chat_message_history[-1], AIMessage)

    reply_prompt = context.bot_llm.received[-1]
    assert isinstance(reply_prompt[1], SystemMessage)
    assert "compacted summary" in reply_prompt[1].content


async def test_a_failed_delivery_does_not_regenerate(make_context, make_settings, fake_model, monkeypatch, _update):
    monkeypatch.setattr("sector_ai.retry.asyncio.sleep", AsyncMock())
    settings = make_settings(streaming_interval_sec=999, streaming_chunk_size=999_999)
    context = make_context(model=fake_model("first answer", "second answer"), settings=settings)
    update = _update()
    posted = SimpleNamespace(edit_text=AsyncMock(side_effect=NetworkError("502")))
    update.message.reply_text = AsyncMock(return_value=posted)

    await handle_chat(update, context)

    assert context.chat_message_history == []
    # A streamed generation records itself in _stream and in _generate; the
    # delivery failure must not start another one.
    assert len(context.bot_llm.received) == 2


async def test_processing_is_posted_before_compaction(make_context, make_update, monkeypatch):
    context = make_context("reply")
    update = make_update(text="/chat hello")
    seen = {}

    async def compact():
        seen["posted"] = update.message.texts("reply_text")
        return False

    monkeypatch.setattr(context, "maybe_compact", compact)

    await handle_chat(update, context)

    assert seen["posted"] == ["Processing..."]


async def test_clear_history_drops_the_summary(make_context):
    context = make_context("unused")
    _fill(context, 3)
    context.context_summary = "old summary"
    context.chat_data["token_count"] = {"tokens": 42}

    context.clear_history()

    assert context.chat_message_history == []
    assert context.context_summary == ""
    assert "context_compaction" not in context.chat_data
    assert "token_count" not in context.chat_data


async def test_a_tail_that_still_does_not_fit_is_trimmed(make_context, make_settings, fake_model, monkeypatch):
    settings = make_settings(compaction_keep_messages=4)
    context = make_context(model=fake_model("first", "second"), settings=settings)
    for _ in range(10):
        context.chat_message_history.append(HumanMessage(content="x" * 3000))
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=4000))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=3800))

    assert await context.maybe_compact() is True
    # Four kept messages at ~750 tokens each cannot fit once the summary and
    # prompt overhead are counted.
    assert len(context.chat_message_history) < 4
    assert context.context_summary == "second"


async def test_a_refusal_is_not_saved_as_a_summary(make_context, monkeypatch):
    context = make_context(model=RefusingModel())
    _fill(context, 10)
    monkeypatch.setattr(context, "get_context_budget", AsyncMock(return_value=8192))

    with pytest.raises(RuntimeError, match="refused"):
        await context.compact()

    assert len(context.chat_message_history) == 10
    assert context.context_summary == ""


def test_passive_messages_respect_the_cap(make_context, make_settings):
    context = make_context(settings=make_settings(compaction_max_messages=5))

    for i in range(12):
        context.save_user_message(SimpleNamespace(text=f"hello {i}", from_user=SimpleNamespace(username="alice")))

    assert len(context.chat_message_history) == 5
    assert context.chat_message_history[-1].content == "alice: hello 11"


async def test_disabling_compaction_only_enforces_the_cap(make_context, make_settings):
    context = make_context("summary", settings=make_settings(compaction_enabled=False, compaction_max_messages=5))
    _fill(context, 8)

    assert await context.maybe_compact() is False
    assert len(context.chat_message_history) == 5
    assert context.context_summary == ""


@pytest.fixture
def _update():
    def make(text="/chat hello"):
        message = SimpleNamespace(
            text=text,
            from_user=SimpleNamespace(username="testuser"),
            date=datetime.now(UTC),
            reply_text=AsyncMock(),
            edit_text=AsyncMock(),
            reply_to_message=None,
        )
        return SimpleNamespace(message=message, effective_message=message, effective_user=message.from_user)

    return make
