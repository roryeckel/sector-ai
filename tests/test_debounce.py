"""A newer relevant message restarts a turn that has not said anything yet."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sector_ai import debounce
from sector_ai.agent import run_agent
from sector_ai.chat_events import handle_edited_message
from sector_ai.sector_context import SectorContext
from sector_ai.streaming_handler import LazyReply


class GatedModel(BaseChatModel):
    """Answers in order; a response paired with an event waits for it first."""

    responses: list = []
    received: list = []

    @property
    def _llm_type(self):
        return "test-gated"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("use the async path")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected extra model invocation")
        message, gate = self.responses.pop(0)
        if gate is not None:
            await gate.wait()
        return ChatResult(generations=[ChatGeneration(message=message)])


def text_of(messages) -> str:
    return "\n".join(str(message.content) for message in messages)


@pytest.fixture
def decision_gates():
    """Text -> event: the engagement decision for that text waits for the event."""
    return {}


@pytest.fixture
def decide(monkeypatch, decision_gates):
    """Engagement decision without a model: anything not marked irrelevant engages."""

    async def should_respond(update, context):
        text = update.effective_message.text or ""
        if text in decision_gates:
            await decision_gates[text].wait()
        return "irrelevant" not in text

    mock = AsyncMock(side_effect=should_respond)
    monkeypatch.setattr("sector_ai.agent.should_respond", mock)
    monkeypatch.setattr("sector_ai.autoreply.should_respond", mock)
    return mock


@pytest.fixture
def chat(make_context, make_settings, decide):
    """Two handler contexts over one chat, as PTB gives concurrent updates."""

    def make(*responses, **settings):
        model = GatedModel(responses=list(responses), received=[])
        first = make_context(
            model=model,
            settings=make_settings(
                context_length=8192, streaming_interval_sec=999, streaming_chunk_size=999999, **settings
            ),
        )
        contexts = [first, *(SectorContext(first.application, chat_id=1, user_id=2) for _ in range(3))]
        return model, contexts

    return make


async def until(condition, timeout=10.0):
    """Wait in real time: parts of a turn (token counting) may run in a thread."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition was never met")
        await asyncio.sleep(0.005)


def humans(context):
    return [message.content for message in context.chat_message_history if isinstance(message, HumanMessage)]


async def test_a_relevant_message_restarts_an_unanswered_turn(chat, make_update):
    gate = asyncio.Event()
    model, (first, second, *_) = chat((AIMessage(content="answer to one"), gate), (AIMessage(content="both"), None))
    one = make_update(text="what is", message_id=10)
    two = make_update(text="the capital of France", message_id=11)

    task = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    await run_agent(two, second, debounce=True)
    await task

    # The first turn said nothing; the second answers with both messages read.
    assert one.message.shown() == []
    assert two.message.texts("reply_text") == ["both"]
    assert len(model.received) == 2
    assert "what is" in text_of(model.received[1]) and "the capital of France" in text_of(model.received[1])
    assert any("what is" in content for content in humans(first))
    assert first.conversation_data.get(debounce.TURNS) == []


async def test_a_turn_that_already_acted_is_not_restarted(chat, make_update):
    gate = asyncio.Event()
    react = AIMessage(
        content="",
        tool_calls=[{"name": "react_to_message", "args": {"emoji": "👍"}, "id": "call-r", "type": "tool_call"}],
    )
    model, (first, second, *_) = chat(
        (react, None), (AIMessage(content="first done"), gate), (AIMessage(content="second"), None)
    )
    one = make_update(text="one", message_id=10)
    two = make_update(text="two", message_id=11)

    task = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(model.received) == 2)
    waiting = asyncio.create_task(run_agent(two, second, debounce=True))
    await until(lambda: len(first.conversation_data[debounce.TURNS]) == 2)
    for _ in range(20):
        await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(task, waiting)

    first.bot.set_message_reaction.assert_awaited_once()
    assert one.message.texts("reply_text") == ["first done"]
    assert two.message.texts("reply_text") == ["second"]


async def test_first_text_commits_the_turn(make_context):
    context = make_context()
    turn = debounce.Turn(message_ids={1}, sender_id=99)
    context._turn = turn
    source = SimpleNamespace(reply_text=AsyncMock(return_value=SimpleNamespace(message_id=5)), chat_id=1)

    await LazyReply(context, source).write("hi")

    assert turn.committed is True


async def test_an_irrelevant_message_does_not_restart(chat, make_update, decide):
    gate = asyncio.Event()
    model, (first, second, *_) = chat((AIMessage(content="answer to one"), gate))
    one = make_update(text="question", message_id=10)
    other = make_update(text="irrelevant aside", message_id=11)
    other.message.from_user.id = 100

    task = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    waiting = asyncio.create_task(run_agent(other, second, debounce=True))
    await until(lambda: decide.await_count >= 2)
    gate.set()
    await asyncio.gather(task, waiting)

    assert one.message.texts("reply_text") == ["answer to one"]
    assert other.message.shown() == []
    assert len(model.received) == 1


async def test_a_burst_waiting_for_the_lock_collapses_into_one_answer(chat, make_update):
    model, (first, second, *_) = chat((AIMessage(content="one answer"), None))
    one = make_update(text="first half", message_id=10)
    two = make_update(text="second half", message_id=11)

    await first.conversation_lock.acquire()
    task_one = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(first.conversation_data.get(debounce.TURNS, [])) == 1)
    task_two = asyncio.create_task(run_agent(two, second, debounce=True))
    await until(lambda: first.conversation_data[debounce.TURNS][0].superseded)
    first.conversation_lock.release()
    await asyncio.gather(task_one, task_two)

    assert one.message.shown() == []
    assert two.message.texts("reply_text") == ["one answer"]
    assert len(model.received) == 1
    assert "first half" in text_of(model.received[0])


async def test_the_restart_cap_stops_superseding(make_context, make_settings, make_update, decide):
    context = make_context(settings=make_settings(debounce_max_restarts=1))
    old = debounce.Turn(message_ids={10}, sender_id=99, sequence=1, restarts=1, engaged=True)
    context.conversation_data[debounce.TURNS] = [old]
    context.conversation_data[debounce.SEQUENCE] = 1

    turn = debounce.register(make_update(text="more", message_id=11), context)
    took_over = await debounce.take_over(make_update(text="more", message_id=11), context, turn)

    assert took_over is False
    assert old.superseded is False
    assert turn.restarts == 0


async def test_zero_restarts_disables_debouncing(make_context, make_settings, make_update, decide):
    context = make_context(settings=make_settings(debounce_max_restarts=0))
    old = debounce.Turn(message_ids={10}, sender_id=99, sequence=1, engaged=True)
    context.conversation_data[debounce.TURNS] = [old]
    context.conversation_data[debounce.SEQUENCE] = 1
    update = make_update(text="more", message_id=11)

    took_over = await debounce.take_over(update, context, debounce.register(update, context))

    assert took_over is False
    assert old.superseded is False


async def test_a_write_queued_by_a_superseded_turn_never_goes_out(make_context):
    # The stream's preview update runs as its own task. If it becomes runnable
    # in the same loop pass that the turn is superseded, it must not send.
    context = make_context()
    context._turn = debounce.Turn(message_ids={1}, sender_id=99, superseded=True)
    source = SimpleNamespace(reply_text=AsyncMock(), chat_id=1)

    with pytest.raises(asyncio.CancelledError):
        await LazyReply(context, source).write("stale preview")

    source.reply_text.assert_not_awaited()
    assert context._turn.committed is False


async def test_a_newer_message_never_loses_one_still_being_checked(chat, make_update, decision_gates):
    gate = asyncio.Event()
    model, (first, second, third, _) = chat((AIMessage(content="stale"), gate), (AIMessage(content="all three"), None))
    a = make_update(text="part one", message_id=10)
    b_text = "part two from someone else"
    decision_gates[b_text] = asyncio.Event()
    b = make_update(text=b_text, message_id=11)
    b.message.from_user.id = 100
    bot_reply = SimpleNamespace(message_id=5, text="earlier bot text", from_user=SimpleNamespace(id=42))
    c = make_update(text="part three", message_id=12, reply_to_message=bot_reply)

    task_a = asyncio.create_task(run_agent(a, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    task_b = asyncio.create_task(run_agent(b, second, debounce=True))
    await until(lambda: len(first.conversation_data[debounce.TURNS]) == 2)
    # C replies to the bot, so it takes over A and B while B is still deciding.
    # Hold the turn objects: the superseded turn leaves TURNS, so a positional
    # slice would soon include C itself.
    turn_a = first.conversation_data[debounce.TURNS][0]
    turn_b = first.conversation_data[debounce.TURNS][1]
    task_c = asyncio.create_task(run_agent(c, third, debounce=True))
    await until(lambda: turn_a.superseded and turn_b.superseded)
    # B's decision now returns "relevant", but B must not cancel the newer C.
    decision_gates[b_text].set()
    await asyncio.gather(task_a, task_b, task_c)

    assert a.message.shown() == [] and b.message.shown() == []
    assert c.message.texts("reply_text") == ["all three"]
    final = text_of(model.received[1])
    assert "part one" in final and b_text in final and "part three" in final


async def test_a_duplicate_delivery_does_not_cancel_the_original(chat, make_update):
    gate = asyncio.Event()
    model, (first, second, *_) = chat((AIMessage(content="answer"), gate))
    one = make_update(text="hello", message_id=10)

    task = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    duplicate = asyncio.create_task(run_agent(one, second, debounce=True))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not first.conversation_data[debounce.TURNS][0].superseded
    gate.set()
    await asyncio.gather(task, duplicate)

    assert one.message.texts("reply_text") == ["answer"]
    assert len(model.received) == 1


async def test_cancelling_during_the_relevance_check_unregisters_the_turn(chat, make_update, decision_gates):
    gate = asyncio.Event()
    model, (first, second, *_) = chat((AIMessage(content="answer"), gate))
    one = make_update(text="question", message_id=10)
    decision_gates["slow aside"] = asyncio.Event()
    other = make_update(text="slow aside", message_id=11)
    other.message.from_user.id = 100

    task = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    checking = asyncio.create_task(run_agent(other, second, debounce=True))
    await until(lambda: len(first.conversation_data[debounce.TURNS]) == 2)
    checking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await checking

    assert [sorted(turn.message_ids) for turn in first.conversation_data[debounce.TURNS]] == [[10]]
    gate.set()
    await task


async def test_an_edit_restarts_an_unanswered_turn(chat, make_update, recording_message):
    gate = asyncio.Event()
    model, (first, second, *_) = chat(
        (AIMessage(content="stale"), gate), (AIMessage(content="answer to the edit"), None)
    )
    one = make_update(text="capital of Spain?", message_id=10)
    edited = recording_message(text="capital of Italy?")
    edited.message_id = 10
    edit = SimpleNamespace(
        update_id=500,
        message=None,
        edited_message=edited,
        effective_message=edited,
        effective_user=one.effective_user,
        effective_chat=one.effective_chat,
        callback_query=None,
    )

    task = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    await handle_edited_message(edit, second)
    await task

    assert one.message.shown() == []
    assert edited.texts("reply_text") == ["answer to the edit"]
    assert "capital of Italy?" in text_of(model.received[1])
    assert any("capital of Italy?" in content for content in humans(first))
    assert not any(content.endswith("capital of Spain?") for content in humans(first))


async def test_an_edit_after_the_reply_only_updates_history(chat, make_update, recording_message):
    model, (first, second, *_) = chat((AIMessage(content="answered"), None))
    one = make_update(text="hello", message_id=10)
    await run_agent(one, first, debounce=True)
    edited = recording_message(text="hello there")
    edited.message_id = 10
    edit = SimpleNamespace(
        update_id=501,
        message=None,
        edited_message=edited,
        effective_message=edited,
        effective_user=one.effective_user,
        effective_chat=one.effective_chat,
        callback_query=None,
    )

    await handle_edited_message(edit, second)

    assert edited.shown() == []
    assert len(model.received) == 1
    assert any("message_edited" in str(message.content) for message in first.chat_message_history)
    assert any("hello there" in content for content in humans(first))


def edit_update(edited, *, update_id, like):
    return SimpleNamespace(
        update_id=update_id,
        message=None,
        edited_message=edited,
        effective_message=edited,
        effective_user=like.effective_user,
        effective_chat=like.effective_chat,
        callback_query=None,
    )


def position(context, text):
    return next(index for index, content in enumerate(humans(context)) if text in content)


async def test_messages_are_stored_in_arrival_order_without_a_takeover(chat, make_update, decision_gates):
    # A is running. B (another sender) is still being checked. C is judged
    # irrelevant, so it takes nothing over, yet it must still be stored after B.
    model, (first, second, third, _) = chat(
        (AIMessage(content="stale"), asyncio.Event()), (AIMessage(content="answer to A and B"), None)
    )
    a = make_update(text="part one", message_id=10)
    b_text = "part two from someone else"
    decision_gates[b_text] = asyncio.Event()
    b = make_update(text=b_text, message_id=11)
    b.message.from_user.id = 100
    c = make_update(text="an irrelevant aside", message_id=12)
    c.message.from_user.id = 101

    task_a = asyncio.create_task(run_agent(a, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    task_b = asyncio.create_task(run_agent(b, second, debounce=True))
    await until(lambda: len(first.conversation_data[debounce.TURNS]) == 2)
    task_c = asyncio.create_task(run_agent(c, third, debounce=True))
    await until(lambda: len(first.conversation_data[debounce.TURNS]) == 3)
    for _ in range(50):
        await asyncio.sleep(0)
    decision_gates[b_text].set()
    await asyncio.gather(task_a, task_b, task_c)

    assert position(first, "part one") < position(first, b_text) < position(first, "an irrelevant aside")
    assert b.message.texts("reply_text") == ["answer to A and B"]


async def test_an_edit_before_the_turn_stores_its_message_is_answered_as_edited(
    chat, make_update, recording_message, decision_gates
):
    # B waits for the lock while A answers. B's text is edited meanwhile: B
    # must store, decide on and answer the edited text.
    gate = asyncio.Event()
    model, (first, second, third, _) = chat(
        (AIMessage(content="answer to A"), gate), (AIMessage(content="answer to the edit"), None)
    )
    a = make_update(text="question A", message_id=10)
    b = make_update(text="irrelevant v1", message_id=11)
    b.message.from_user.id = 100
    edited = recording_message(text="relevant v2", user_id=100)
    edited.message_id = 11

    task_a = asyncio.create_task(run_agent(a, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    task_b = asyncio.create_task(run_agent(b, second, debounce=True))
    await until(lambda: len(first.conversation_data[debounce.TURNS]) == 2)
    await handle_edited_message(edit_update(edited, update_id=700, like=b), third)
    gate.set()
    await asyncio.gather(task_a, task_b)

    assert any("relevant v2" in content for content in humans(first))
    assert not any("irrelevant v1" in content for content in humans(first))
    assert not any("message_edited" in str(item.content) for item in first.chat_message_history)
    assert "relevant v2" in text_of(model.received[1])
    assert edited.texts("reply_text") == ["answer to the edit"]


async def test_a_redelivered_edit_does_not_restart_the_turn(chat, make_update, recording_message):
    gate = asyncio.Event()
    model, (first, second, third, _) = chat(
        (AIMessage(content="stale"), gate), (AIMessage(content="answer to the edit"), None)
    )
    one = make_update(text="capital of Spain?", message_id=10)
    edited = recording_message(text="capital of Italy?")
    edited.message_id = 10
    edit = edit_update(edited, update_id=500, like=one)

    task = asyncio.create_task(run_agent(one, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    restarted = asyncio.create_task(handle_edited_message(edit, second))
    await until(lambda: len(model.received) == 2)
    restarted_turn = first.conversation_data[debounce.TURNS][-1]
    # The same edit delivered again: no second restart, no extra model call.
    await handle_edited_message(edit, third)
    assert not restarted_turn.superseded
    await asyncio.gather(task, restarted)

    assert len(model.received) == 2
    assert edited.texts("reply_text") == ["answer to the edit"]


async def test_a_redelivery_of_a_superseded_unstored_update_is_ignored(make_context, make_update):
    context = make_context()
    update = make_update(text="hello", message_id=10)
    original = debounce.register(update, context)
    debounce.supersede([original])

    assert debounce.register(update, context) is None
    # An edit restart of that message may still register.
    assert debounce.register(update, context, edited=True) is not None


async def test_a_second_edit_during_an_edit_restart_is_not_lost(make_context, make_update, recording_message):
    # The replacement turn was registered (ingest=False) but has not stored
    # anything; a second edit must go through the edit handler, not turn.edits.
    context = make_context()
    original = make_update(text="version zero", message_id=10)
    context.bind_update(original)
    context.save_user_message(original.message)
    replacement = debounce.register(original, context, edited=True)
    second = recording_message(text="version two")
    second.message_id = 10

    await handle_edited_message(edit_update(second, update_id=502, like=original), context)

    assert replacement.edits == {}
    assert humans(context)[0] == "testuser: version two"
    # The original text survives only as the event's "previous excerpt".
    assert not any(content.startswith("testuser: version zero") for content in humans(context))


async def test_a_context_only_turn_never_takes_over_another(chat, make_update, make_photo):
    gate = asyncio.Event()
    model, (first, second, *_) = chat((AIMessage(content="answer to the question"), gate))
    question = make_update(text="what is this?", message_id=10)
    task = asyncio.create_task(run_agent(question, first, debounce=True))
    await until(lambda: len(model.received) == 1)
    turn = first.conversation_data[debounce.TURNS][0]
    late = make_update(text=None, caption="late photo", photos=[make_photo()], message_id=11)
    # The first turn holds the conversation lock while its model is gated.
    context_only = asyncio.create_task(run_agent(late, second, photos=[late.message], engage=False, debounce=True))
    await until(lambda: len(first.conversation_data[debounce.TURNS]) == 2)
    gate.set()
    await asyncio.gather(task, context_only)

    assert turn.superseded is False
    assert late.message.shown() == []
    assert question.message.texts("reply_text") == ["answer to the question"]
    assert any("late photo" in content for content in humans(first))
