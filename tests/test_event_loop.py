"""Regression tests: model and Ollama work must not stall the event loop.

``python-telegram-bot`` polls and sends on the same loop the handlers run on. A
synchronous model call blocks that loop, so long generations starve outbound
requests until they time out. The first test holds the model for 300 ms and
counts how often a concurrent ticker still runs. The second checks that the
``/models`` lookup is awaited, which is what keeps its HTTP calls off the loop.
"""

import asyncio
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sector_ai.admin import models_cmd
from sector_ai.chat import handle_chat

TICK = 0.01


class SlowModel(BaseChatModel):
    """Answers after a synchronous ``time.sleep``, like a cold Ollama load."""

    delay: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="slow answer"))])


async def count_ticks(duration: float, started: asyncio.Event | None = None) -> int:
    ticks = 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration
    if started is not None:
        started.set()
    while loop.time() < deadline:
        ticks += 1
        await asyncio.sleep(TICK)
    return ticks


async def start_ticker(duration: float) -> tuple[asyncio.Task, asyncio.Event]:
    """Start the ticker and wait until its deadline exists.

    Without the wait the task is only *scheduled*: a handler that blocks on its
    first call would finish before the ticker ever runs, and the ticker would
    then measure a quiet loop and pass. Seeing the event means the clock is
    already running before the code under test is called.
    """
    started = asyncio.Event()
    task = asyncio.create_task(count_ticks(duration, started))
    await started.wait()
    return task, started


async def test_handle_chat_does_not_block_the_event_loop(make_context, make_update, make_settings):
    settings = make_settings(streaming_interval_sec=999, streaming_chunk_size=999_999)
    context = make_context(model=SlowModel(), settings=settings)
    update = make_update(text="/chat hi")

    ticker, _ = await start_ticker(0.25)
    await handle_chat(update, context)
    ticks = await ticker

    assert update.message.shown()[-1] == "slow answer"
    assert ticks >= 5, f"event loop stalled during the model call (only {ticks} ticks)"


async def test_models_cmd_awaits_its_lookup(make_context, make_update, monkeypatch):
    context = make_context("unused")
    awaited = []

    async def get_models():
        # If the handler forgets the await, this coroutine never starts and the
        # keyboard is built from a coroutine object, which fails loudly.
        awaited.append("started")
        return [{"model": "a:latest", "label": "a:latest"}]

    monkeypatch.setattr(context, "get_models", get_models)
    update = make_update(text="/models", username="ex0dus")

    await models_cmd(update, context)

    assert awaited == ["started"]
    assert update.message.calls_to("reply_text")[0][1]["reply_markup"] is not None
