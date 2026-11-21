"""Tests for SectorContext core state management."""

import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from telegram.constants import MessageLimit

from sector_ai.sector_context import SectorContext

# --- save_user_message ---


def _make_mock_self(maxlen=15):
    """Create a mock self with a real deque for chat_message_history."""
    mock_self = MagicMock()
    mock_self.chat_message_history = deque(maxlen=maxlen)
    return mock_self


def _make_message(text, username="testuser"):
    msg = MagicMock()
    msg.text = text
    msg.from_user.username = username
    return msg


def _client_with(handler):
    """Point SectorContext's ``AsyncClient`` at a mock transport instead of the network."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    return patch("sector_ai.sector_context.httpx.AsyncClient", side_effect=factory)


def test_save_plain_text():
    mock_self = _make_mock_self()
    msg = _make_message("hello world")
    result = SectorContext.save_user_message(mock_self, msg)
    assert result is True
    assert len(mock_self.chat_message_history) == 1
    saved = mock_self.chat_message_history[0]
    assert isinstance(saved, HumanMessage)
    assert saved.content == "testuser: hello world"


def test_save_command_with_arg():
    mock_self = _make_mock_self()
    msg = _make_message("/chat tell me a joke")
    result = SectorContext.save_user_message(mock_self, msg)
    assert result is True
    assert mock_self.chat_message_history[0].content == "testuser: tell me a joke"


def test_save_bare_command():
    mock_self = _make_mock_self()
    msg = _make_message("/clear")
    result = SectorContext.save_user_message(mock_self, msg)
    assert result is False
    assert len(mock_self.chat_message_history) == 0


def test_save_preserves_username():
    mock_self = _make_mock_self()
    msg = _make_message("hi", username="alice")
    SectorContext.save_user_message(mock_self, msg)
    assert mock_self.chat_message_history[0].name == "alice"


def test_save_history_maxlen():
    mock_self = _make_mock_self(maxlen=15)
    for i in range(16):
        msg = _make_message(f"message {i}")
        SectorContext.save_user_message(mock_self, msg)
    assert len(mock_self.chat_message_history) == 15
    # First message evicted, oldest is message 1
    assert "message 1" in mock_self.chat_message_history[0].content


# --- message_exists ---


def test_exists_matching():
    mock_self = _make_mock_self()
    mock_self.chat_message_history.append(HumanMessage(content="x"))
    # Use HumanMessage for the check so .type matches ("human" == "human")
    check = HumanMessage(content="x")
    result = SectorContext.message_exists(mock_self, check)
    assert result is True


def test_exists_no_match():
    mock_self = _make_mock_self()
    check = HumanMessage(content="anything")
    result = SectorContext.message_exists(mock_self, check)
    assert result is False


# --- settings plumbing ---


async def test_ollama_url_comes_from_settings(make_context, make_settings):
    context = make_context(settings=make_settings(ollama_url="http://custom:1234"))
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404)

    with _client_with(handler):
        await context.get_model_details()

    assert seen == ["http://custom:1234/api/show"]


# --- get_model_details ---


async def test_get_model_details_strips_fields(make_context):
    context = make_context("unused")
    context.bot_ollama.model = "test-model"

    payload = {
        "license": "MIT",
        "modelfile": "FROM ...",
        "parameters": "num_ctx 2048",
        "template": "{{ .System }}",
        "details": {"family": "llama", "parameter_size": "7B"},
        "model_info": {"general.architecture": "llama"},
    }
    asked = []

    def handler(request):
        asked.append(json.loads(request.content)["name"])
        return httpx.Response(200, json=payload)

    with _client_with(handler):
        result = await context.get_model_details()

    assert asked == ["test-model"]
    assert result["details"] == {"family": "llama", "parameter_size": "7B"}
    assert "model_info" in result
    assert "license" not in result
    assert "modelfile" not in result
    assert "parameters" not in result
    assert "template" not in result


# --- real context over fake application ---


def test_history_is_shared_per_chat(make_context):
    context = make_context("unused", chat_id=1)

    assert SectorContext(context.application, chat_id=1).chat_message_history is context.chat_message_history
    assert SectorContext(context.application, chat_id=2).chat_message_history is not context.chat_message_history


def test_history_is_bounded(make_context):
    context = make_context("unused")
    assert context.chat_message_history.maxlen == 15


def test_autoreply_defaults_on_and_persists(make_context):
    context = make_context("unused")
    assert context.chat_autoreply_mode is True

    context.chat_autoreply_mode = False
    assert SectorContext(context.application, chat_id=1).chat_autoreply_mode is False


def test_svg_size_defaults_and_persists(make_context):
    context = make_context("unused")
    assert context.chat_svg_size == context.settings.default_svg_size

    context.chat_svg_size = 512
    assert SectorContext(context.application, chat_id=1).chat_svg_size == 512


def test_system_prompt_override_persists(make_context):
    context = make_context("unused")
    context.chat_system_prompt = "You are a pirate."

    assert SectorContext(context.application, chat_id=1).chat_system_prompt == "You are a pirate."
    assert SectorContext(context.application, chat_id=2).chat_system_prompt == context.settings.system_prompts.default


async def test_system_template_dict_uses_telegram_limits(make_context):
    context = make_context("unused")

    result = await context.get_system_template_dict()

    assert result == {"MAX_LEN": MessageLimit.MAX_TEXT_LENGTH, "USERNAME": "testbot"}


def test_templated_messages_preserve_order_and_names(make_context):
    context = make_context("unused")
    context.chat_message_history.append(HumanMessage(name="alice", content="alice: hi"))
    context.chat_message_history.append(AIMessage(content="hello"))

    messages = context.get_templated_messages().format_messages(MAX_LEN=4096, USERNAME="bot")

    assert [type(message) for message in messages] == [SystemMessage, HumanMessage, AIMessage]
    assert str(MessageLimit.MAX_TEXT_LENGTH) in messages[0].content
    assert "bot" in messages[0].content
    assert messages[1].name == "alice"


def test_templated_messages_honour_num_history(make_context):
    context = make_context("unused")
    context.chat_message_history.append(HumanMessage(content="one"))
    context.chat_message_history.append(HumanMessage(content="two"))

    messages = context.get_templated_messages(num_history=1).format_messages(MAX_LEN=4096, USERNAME="bot")

    assert [message.content for message in messages[1:]] == ["two"]


def test_load_model_passes_ollama_options(make_context):
    context = make_context("unused")

    with patch("sector_ai.sector_context.ChatOllama") as chat_ollama:
        result = context.load_model("custom:latest")

    assert result is chat_ollama.return_value
    assert context.bot_data["ollama"] is result
    assert chat_ollama.call_args.kwargs == {
        "model": "custom:latest",
        "base_url": context.settings.ollama_url,
        "num_ctx": context.settings.ollama_context_length,
        "client_kwargs": {
            "timeout": context.settings.ollama_timeout,
            "headers": context.settings.ollama_headers,
        },
    }


async def test_get_models_filters_disallowed_and_labels(make_context):
    context = make_context("unused")
    context.bot_ollama.model = context.settings.default_model
    context.settings.ollama_disallowed_models = ["blocked:latest"]

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"model": context.settings.default_model},
                        {"model": "other:latest"},
                        {"model": "blocked:latest"},
                    ]
                },
            )
        name = json.loads(request.content)["name"]
        return httpx.Response(200, json={"details": {"family": "llama"}, "name": name})

    with _client_with(handler):
        models = await context.get_models()

    names = [model["model"] for model in models]
    assert "blocked:latest" not in names
    assert len(models) == 2
    default = next(model for model in models if model["model"] == context.settings.default_model)
    assert default["is_default_active"] is True
    assert default["is_vision_active"] is True
    assert default["details"] == {"family": "llama"}
    assert default["label"].startswith("✅")


def test_settings_setter_updates_bot_data(make_context, make_settings):
    context = make_context("unused")
    new_settings = make_settings(default_model="other:latest")

    context.settings = new_settings

    assert context.bot_data["settings"] is new_settings


def test_chat_message_history_setter(make_context):
    context = make_context("unused")
    history = deque([HumanMessage(content="x")], maxlen=3)

    context.chat_message_history = history

    assert context.chat_message_history is history
    assert context.chat_message_history.maxlen == 3


def test_decide_autoreply_prompt_comes_from_settings(make_context, sample_settings):
    context = make_context("unused")

    assert context.settings.system_prompts.decide_autoreply == sample_settings.system_prompts.decide_autoreply


def test_bot_ollama_loads_lazily(make_context):
    context = make_context("unused")
    del context.bot_data["ollama"]

    with patch("sector_ai.sector_context.ChatOllama") as chat_ollama:
        model = context.bot_ollama

    assert model is chat_ollama.return_value
    assert context.bot_data["ollama"] is model


async def test_get_username_is_cached(make_context):
    context = make_context("unused")
    calls = []

    async def get_me():
        calls.append(1)
        return SimpleNamespace(username="cached_bot")

    context.application.bot.get_me = get_me

    assert await context.get_username() == "cached_bot"
    assert await context.get_username() == "cached_bot"
    assert len(calls) == 1


async def test_get_model_details_returns_empty_on_failure(make_context):
    context = make_context("unused")

    with _client_with(lambda request: httpx.Response(500)):
        assert await context.get_model_details() == {}


async def test_get_models_returns_empty_on_failure(make_context):
    context = make_context("unused")

    with _client_with(lambda request: httpx.Response(500)):
        assert await context.get_models() == []
