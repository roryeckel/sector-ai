"""Shared fixtures and fakes for the sector_ai test suite.

Two styles are supported:

* ``mock_context`` / ``mock_update`` - fast ``MagicMock`` stand-ins, used by the
  pure logic tests that only need to verify branching.
* ``make_context`` / ``make_update`` - a real ``SectorContext`` over in-memory
  application stores plus a ``FakeOllamaModel``, used by the integration tests
  that assert on the prompt going in and the payloads going out.
"""

from collections import defaultdict, deque
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from telegram.error import BadRequest

from sector_ai.sector_context import SectorContext
from sector_ai.settings import Settings

TEST_TOKEN = "123:test-token"


@pytest.fixture
def make_settings():
    """Build settings with test defaults.

    ``_env_file=None`` keeps a developer's local ``.env`` out of the suite; the
    precedence tests supply their own file explicitly.
    """

    def _make(**overrides) -> Settings:
        return Settings(telegram_bot_token=TEST_TOKEN, _env_file=None, **overrides)

    return _make


@pytest.fixture
def sample_settings(make_settings):
    return make_settings()


@pytest.fixture
def mock_context(sample_settings):
    ctx = MagicMock(spec=SectorContext)

    # bot_data / chat_data backing stores
    ctx.bot_data = {"settings": sample_settings}
    ctx.settings = sample_settings
    chat_data = {"message_history": deque(maxlen=15), "autoreply_mode": True}
    ctx.chat_data = chat_data

    # Chat properties backed by chat_data
    type(ctx).chat_message_history = PropertyMock(return_value=chat_data["message_history"])
    type(ctx).chat_autoreply_mode = PropertyMock(return_value=True)
    # Don't use PropertyMock for chat_system_prompt - let MagicMock handle
    # attribute assignment naturally so tests can verify setter calls
    ctx.chat_system_prompt = sample_settings.system_prompts.default

    # Async methods
    ctx.get_system_template_dict = AsyncMock(return_value={"MAX_LEN": 4096, "USERNAME": "testbot"})

    # bot_ollama
    ctx.bot_ollama = MagicMock(model="test-model")

    return ctx


@pytest.fixture
def mock_update():
    def _factory(text="/cmd", username="testuser"):
        update = MagicMock()
        update.message.text = text
        update.effective_message.text = text
        update.message.from_user.username = username
        update.effective_user.username = username
        update.message.reply_text = AsyncMock()
        update.message.reply_poll = AsyncMock()
        return update

    return _factory


# --- Fake LLM -------------------------------------------------------------


class FakeOllamaModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` with the ``ChatOllama`` fields the bot touches.

    ``GenericFakeChatModel`` streams by splitting each message on whitespace,
    which is exactly what the streaming tests need. The extra fields mirror the
    subset of ``ChatOllama`` that ``SectorContext`` and the commands read or
    assign (``model``, ``temperature``, ``format``).
    """

    model: str = "fake-model"
    temperature: float | None = None
    format: str = ""
    received: list[list[Any]] = []
    call_count: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        self.call_count += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        self.call_count += 1
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        # Deterministic, and avoids langchain's GPT-2 fallback tokenizer.
        return len(text.split())


@pytest.fixture
def fake_model():
    """Build a fake model that answers with the given contents, in order."""

    def _make(*contents: str, **kwargs) -> FakeOllamaModel:
        messages = [AIMessage(content=c) for c in (contents or ("",))]
        return FakeOllamaModel(messages=iter(messages), **kwargs)

    return _make


# --- Fake Telegram side ---------------------------------------------------


class RecordingMessage:
    """Stands in for ``telegram.Message`` in both directions.

    Incoming attributes (``text``, ``caption``, ``photo`` ...) model a received
    update; ``reply_*`` / ``edit_text`` record what the bot sends so tests can
    assert on the exact payload instead of just "something was awaited".

    ``fail_times`` makes the first N outbound calls raise ``BadRequest``, which
    is how the retry loops are exercised.
    """

    def __init__(
        self,
        text=None,
        caption=None,
        username="testuser",
        user_id=99,
        date=None,
        photos=None,
        reply_to_message=None,
        fail_times=0,
    ):
        self.text = text
        self.caption = caption
        self.date = date or datetime.now(UTC)
        self.photo = photos or []
        self.reply_to_message = reply_to_message
        self.from_user = SimpleNamespace(username=username, id=user_id)
        self.chat = SimpleNamespace(id=1)
        self.calls: list[tuple[str, tuple, dict]] = []
        self._fail_times = fail_times

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))
        if self._fail_times:
            self._fail_times -= 1
            raise BadRequest(f"{name} failed")

    async def reply_text(self, text, **kwargs):
        self._record("reply_text", (text,), kwargs)
        return self

    async def reply_markdown_v2(self, text, **kwargs):
        self._record("reply_markdown_v2", (text,), kwargs)
        return self

    async def reply_photo(self, photo=None, **kwargs):
        self._record("reply_photo", (), {"photo": photo, **kwargs})
        return self

    async def reply_document(self, document=None, **kwargs):
        self._record("reply_document", (), {"document": document, **kwargs})
        return self

    async def reply_poll(self, question, options, **kwargs):
        self._record("reply_poll", (question, options), kwargs)
        return self

    async def edit_text(self, text, **kwargs):
        self._record("edit_text", (text,), kwargs)
        return self

    # Assertion helpers

    def calls_to(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for method, args, kwargs in self.calls if method == name]

    def last_text(self) -> str:
        return self.calls_to("reply_text")[-1][0][0]

    def texts(self, name: str = "reply_text") -> list[str]:
        return [args[0] for args, _ in self.calls_to(name)]


@pytest.fixture
def recording_message():
    def _make(**kwargs) -> RecordingMessage:
        return RecordingMessage(**kwargs)

    return _make


@pytest.fixture
def make_update(recording_message):
    def _make(
        text="/cmd",
        username="testuser",
        *,
        caption=None,
        photos=None,
        reply_to_message=None,
        date=None,
        callback_data=None,
        fail_times=0,
    ):
        message = recording_message(
            text=text,
            caption=caption,
            username=username,
            photos=photos,
            reply_to_message=reply_to_message,
            date=date,
            fail_times=fail_times,
        )
        update = SimpleNamespace(
            message=message,
            effective_message=message,
            effective_user=SimpleNamespace(username=username, id=99),
            effective_chat=SimpleNamespace(id=1),
            callback_query=None,
        )
        if callback_data is not None:
            update.callback_query = SimpleNamespace(
                data=callback_data,
                answer=AsyncMock(),
                edit_message_text=AsyncMock(),
            )
        return update

    return _make


@pytest.fixture
def make_photo():
    def _make(file_size=1024, data=b"image-bytes"):
        async def get_file(read_timeout=None, connect_timeout=None):
            async def download_as_bytearray():
                return bytearray(data)

            return SimpleNamespace(download_as_bytearray=download_as_bytearray)

        return SimpleNamespace(file_size=file_size, get_file=get_file)

    return _make


# --- Real context over fake application -----------------------------------


def make_application(settings, model=None, bot_id: int = 42, username: str = "testbot"):
    async def get_me():
        return SimpleNamespace(username=username)

    bot_data = {"settings": settings}
    if model is not None:
        bot_data["ollama"] = model

    # telegram.ext.Application keeps these as defaultdict(dict); a plain dict
    # raises KeyError from CallbackContext.chat_data before the bot sees it.
    return SimpleNamespace(
        bot_data=bot_data,
        chat_data=defaultdict(dict),
        user_data=defaultdict(dict),
        bot=SimpleNamespace(id=bot_id, get_me=get_me),
    )


@pytest.fixture
def make_context(fake_model, sample_settings):
    def _make(*contents: str, model=None, settings=None, chat_id=1, user_id=2, **kwargs) -> SectorContext:
        if settings is None:
            settings = sample_settings
        if model is None:
            model = fake_model(*contents, **kwargs)
        application = make_application(settings, model=model)
        return SectorContext(application, chat_id=chat_id, user_id=user_id)

    return _make
