"""Tests for SectorContext core state management."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from telegram.constants import MessageLimit

from sector_ai.sector_context import SectorContext

# --- save_user_message ---


def _make_mock_self():
    """Create a mock self with a real list for chat_message_history."""
    mock_self = MagicMock()
    mock_self.chat_message_history = []
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


def test_save_keeps_more_than_fifteen_messages():
    mock_self = _make_mock_self()
    for i in range(16):
        SectorContext.save_user_message(mock_self, _make_message(f"message {i}"))
    assert len(mock_self.chat_message_history) == 16
    assert "message 0" in mock_self.chat_message_history[0].content


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
    context.bot_data["model_name"] = "test-model"

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


def test_history_is_not_cut_off_at_a_fixed_length(make_context):
    context = make_context("unused")
    for i in range(16):
        context.chat_message_history.append(HumanMessage(content=f"message {i}"))

    assert len(context.chat_message_history) == 16
    assert context.chat_message_history[0].content == "message 0"


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
    assert context.bot_data["llm"] is result
    assert context.bot_data["model_name"] == "custom:latest"
    assert chat_ollama.call_args.kwargs == {
        "model": "custom:latest",
        "base_url": context.settings.ollama_url,
        "client_kwargs": {
            "timeout": context.settings.llm_timeout,
            "headers": context.settings.llm_headers,
        },
    }


def test_load_model_uses_the_openai_backend(make_context, make_settings):
    settings = make_settings(
        llm_provider="openai",
        openai_base_url="http://lmstudio:1234/v1",
        openai_api_key="",
        llm_timeout=12.0,
        llm_headers={"X-Test": "yes"},
    )
    context = make_context("unused", settings=settings)

    with patch("sector_ai.sector_context.ChatOpenAI") as chat_openai:
        result = context.load_model("vendor/model")

    assert result is chat_openai.return_value
    assert context.bot_data["llm"] is result
    assert context.bot_data["model_name"] == "vendor/model"
    assert chat_openai.call_args.kwargs == {
        "model": "vendor/model",
        "base_url": "http://lmstudio:1234/v1",
        "api_key": "dummy",
        "timeout": 12.0,
        "default_headers": {"X-Test": "yes"},
        "use_responses_api": False,
        "max_retries": 0,
        "stream_usage": False,
    }


def test_load_model_enables_stream_usage_when_requested(make_context, make_settings):
    settings = make_settings(llm_provider="openai", stream_usage=True)
    context = make_context("unused", settings=settings)

    with patch("sector_ai.sector_context.ChatOpenAI") as chat_openai:
        context.load_model("gpt")

    assert chat_openai.call_args.kwargs["stream_usage"] is True


def test_load_model_keeps_a_configured_openai_key(make_context, make_settings):
    settings = make_settings(llm_provider="openai", openai_api_key="sk-real")
    context = make_context("unused", settings=settings)

    with patch("sector_ai.sector_context.ChatOpenAI") as chat_openai:
        context.load_model("gpt")

    assert chat_openai.call_args.kwargs["api_key"] == "sk-real"


def test_model_is_rebuilt_when_settings_change(make_context, make_settings):
    context = make_context("unused")

    with patch("sector_ai.sector_context.ChatOllama"):
        context.load_model("first")

    context.settings = make_settings(llm_provider="openai", openai_base_url="http://other:1234/v1")

    with patch("sector_ai.sector_context.ChatOpenAI") as chat_openai:
        context._refresh_model_if_settings_changed()

    assert context.bot_data["llm"] is chat_openai.return_value
    assert context.bot_data["llm_config"][:3] == ("openai", "http://other:1234/v1", "first")


async def test_get_model_details_is_empty_for_openai(make_context, make_settings):
    context = make_context("unused", settings=make_settings(llm_provider="openai"))
    asked = []

    def handler(request):
        asked.append(str(request.url))
        return httpx.Response(200, json={})

    with _client_with(handler):
        result = await context.get_model_details()

    assert result == {}
    assert asked == []


async def test_get_models_filters_disallowed_and_labels(make_context):
    context = make_context("unused")
    context.bot_data["model_name"] = context.settings.default_model
    context.settings.llm_disallowed_models = ["blocked:latest"]

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


async def test_get_models_survives_a_failed_metadata_lookup(make_context, caplog):
    context = make_context("unused")
    context.bot_data["model_name"] = "good:latest"

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"model": "good:latest"}, {"model": "slow:latest"}]})
        if json.loads(request.content)["name"] == "slow:latest":
            raise httpx.ReadTimeout("too slow", request=request)
        return httpx.Response(200, json={"details": {"family": "llama"}})

    with _client_with(handler), caplog.at_level("WARNING"):
        models = await context.get_models()

    # The slow model still appears, just without the metadata the others have.
    assert [model["model"] for model in models] == ["good:latest", "slow:latest"]
    assert models[0]["details"] == {"family": "llama"}
    assert "details" not in models[1]
    assert models[1]["label"] == "slow:latest"
    assert "slow:latest" in caplog.text


async def test_get_models_uses_the_openai_endpoint(make_context, make_settings):
    settings = make_settings(
        llm_provider="openai",
        openai_base_url="http://lmstudio:1234/v1/",
        default_model="vendor/default",
        llm_disallowed_models=["blocked"],
    )
    context = make_context("unused", settings=settings)
    context.bot_data["model_name"] = "vendor/default"
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("Authorization")))
        return httpx.Response(
            200,
            json={"data": [{"id": "vendor/default"}, {"id": "vendor/other"}, {"id": "blocked"}]},
        )

    with _client_with(handler):
        models = await context.get_models()

    assert seen == [("http://lmstudio:1234/v1/models", "Bearer dummy")]
    assert [model["model"] for model in models] == ["vendor/default", "vendor/other"]
    assert models[0]["is_default_active"] is True
    assert models[0]["label"].startswith("✅")


async def test_openai_models_use_the_configured_key(make_context, make_settings):
    settings = make_settings(llm_provider="openai", openai_api_key="sk-real")
    context = make_context("unused", settings=settings)
    seen = []

    def handler(request):
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"data": []})

    with _client_with(handler):
        await context.get_models()

    assert seen == ["Bearer sk-real"]


async def test_custom_lowercase_authorization_is_not_duplicated(make_context, make_settings):
    settings = make_settings(llm_provider="openai", llm_headers={"authorization": "Bearer custom"})
    context = make_context("unused", settings=settings)
    seen = []

    def handler(request):
        seen.extend(request.headers.get_list("authorization"))
        return httpx.Response(200, json={"data": []})

    with _client_with(handler):
        await context.get_models()

    assert seen == ["Bearer custom"]


async def test_get_models_openai_failure_returns_empty(make_context, make_settings, caplog):
    context = make_context("unused", settings=make_settings(llm_provider="openai"))

    def handler(request):
        return httpx.Response(500)

    with _client_with(handler), caplog.at_level("WARNING"):
        assert await context.get_models() == []

    assert "500" in caplog.text


async def test_get_models_openai_handles_missing_data(make_context, make_settings):
    context = make_context("unused", settings=make_settings(llm_provider="openai"))

    def handler(request):
        return httpx.Response(200, json={"data": None})

    with _client_with(handler):
        assert await context.get_models() == []


# --- token counting ---


async def test_count_context_tokens_uses_the_ollama_probe(make_context, fake_model):
    usage = {"input_tokens": 42, "output_tokens": 1, "total_tokens": 43}
    context = make_context(model=fake_model("", usage_metadata=usage))
    context.chat_message_history.append(HumanMessage(name="alice", content="alice: hi"))

    assert await context.count_context_tokens() == 42

    sent = context.bot_llm.received[-1]
    assert sent[0].type == "system"
    assert sent[-1].content == "alice: hi"
    assert context.bot_llm.received_kwargs[-1]["options"] == {"num_predict": 1}


async def test_count_context_tokens_prefers_the_openai_endpoint(make_context, make_settings, fake_model):
    settings = make_settings(llm_provider="openai", openai_base_url="http://lmstudio:1234/v1")
    model = fake_model("", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    context = make_context(model=model, settings=settings)
    context.chat_message_history.append(HumanMessage(content="hello"))
    context.chat_message_history.append(AIMessage(content="hi there"))
    context.chat_message_history.append(HumanMessage(content="how are you?"))
    seen = []

    def handler(request):
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 77})

    with _client_with(handler):
        assert await context.count_context_tokens() == 77

    assert model.call_count == 0
    url, payload = seen[0]
    assert url == "http://lmstudio:1234/v1/responses/input_tokens"
    assert payload["model"] == "fake-model"
    assert payload["input"][-2] == {"role": "assistant", "content": "hi there"}
    assert payload["input"][-1] == {"role": "user", "content": "how are you?"}
    assert "You are" in payload["instructions"]


async def test_count_context_tokens_falls_back_to_the_openai_probe(make_context, make_settings, fake_model):
    settings = make_settings(llm_provider="openai")
    usage = {"input_tokens": 42, "output_tokens": 1, "total_tokens": 43}
    model = fake_model("", usage_metadata=usage)
    context = make_context(model=model, settings=settings)

    def handler(request):
        return httpx.Response(404)

    with _client_with(handler):
        assert await context.count_context_tokens() == 42

    assert model.call_count == 1
    assert model.received_kwargs[-1]["extra_body"] == {"max_tokens": 1}
    assert "http://localhost:11434/v1" in context.bot_data["token_count_unavailable"]


async def test_count_context_tokens_falls_back_when_the_endpoint_errors(make_context, make_settings, fake_model):
    settings = make_settings(llm_provider="openai")
    usage = {"input_tokens": 42, "output_tokens": 1, "total_tokens": 43}
    model = fake_model("", usage_metadata=usage)
    context = make_context(model=model, settings=settings)

    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with _client_with(handler):
        assert await context.count_context_tokens() == 42

    assert model.call_count == 1


async def test_count_context_tokens_retries_a_bad_request(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)

    error = Exception("Bad request")
    error.status_code = 400

    rejected = MagicMock()
    rejected.ainvoke = AsyncMock(side_effect=error)
    accepted = MagicMock()
    accepted.ainvoke = AsyncMock(return_value=SimpleNamespace(usage_metadata={"input_tokens": 42}))

    model = MagicMock()
    model._use_responses_api.return_value = False
    model.bind.side_effect = lambda **kwargs: (
        accepted if kwargs.get("extra_body") == {"max_completion_tokens": 16} else rejected
    )
    context.bot_data["llm"] = model

    def handler(request):
        return httpx.Response(404)

    with _client_with(handler):
        assert await context.count_context_tokens() == 42

    assert model.bind.call_count == 2
    assert accepted.ainvoke.await_count == 1


async def test_count_context_tokens_uses_max_output_tokens_for_responses_models(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)

    probe = MagicMock()
    probe.ainvoke = AsyncMock(return_value=SimpleNamespace(usage_metadata={"input_tokens": 42}))
    model = MagicMock()
    model._use_responses_api.return_value = True
    model.bind.return_value = probe
    context.bot_data["llm"] = model

    def handler(request):
        return httpx.Response(404)

    with _client_with(handler):
        assert await context.count_context_tokens() == 42

    model.bind.assert_called_once_with(max_tokens=1)


def test_uses_responses_api_detects_real_models(make_context, make_settings):
    from langchain_openai import ChatOpenAI

    context = make_context("unused", settings=make_settings(llm_provider="openai"))

    context.bot_data["llm"] = ChatOpenAI(model="gpt-5-pro", api_key="dummy")
    assert context._uses_responses_api() is True

    context.bot_data["llm"] = ChatOpenAI(model="gpt-4o-mini", api_key="dummy")
    assert context._uses_responses_api() is False


async def test_token_count_is_cached_until_the_context_changes(make_context, fake_model):
    usage = {"input_tokens": 42, "output_tokens": 1, "total_tokens": 43}
    context = make_context(model=fake_model(*[""] * 5, usage_metadata=usage))

    assert await context.count_context_tokens() == 42
    assert await context.count_context_tokens() == 42
    assert context.bot_llm.call_count == 1

    context.chat_message_history.append(HumanMessage(content="new"))
    assert await context.count_context_tokens() == 42
    assert context.bot_llm.call_count == 2

    context.chat_system_prompt = "A different system prompt."
    assert await context.count_context_tokens() == 42
    assert context.bot_llm.call_count == 3

    context.bot_data["model_name"] = "another-model"
    assert await context.count_context_tokens() == 42
    assert context.bot_llm.call_count == 4


async def test_token_count_falls_back_to_an_estimate(make_context):
    context = make_context("")

    tokens = await context.count_context_tokens()

    assert tokens > 0
    assert context.token_count_is_exact is False


async def test_estimate_works_without_a_tokenizer(make_context, monkeypatch):
    from langchain_ollama import ChatOllama

    context = make_context("unused")
    context.bot_data["llm"] = ChatOllama(model="test", base_url="http://localhost:11434")
    context.bot_data["model_name"] = "test"

    async def no_count(messages):
        raise RuntimeError("no count")

    monkeypatch.setattr(context, "_probe_token_count", no_count)

    tokens = await context.count_context_tokens()

    assert tokens > 0
    assert context.token_count_is_exact is False


async def test_zero_free_count_falls_back_to_the_probe(make_context, make_settings, fake_model):
    settings = make_settings(llm_provider="openai")
    usage = {"input_tokens": 42, "output_tokens": 1, "total_tokens": 43}
    model = fake_model("", usage_metadata=usage)
    context = make_context(model=model, settings=settings)

    def handler(request):
        return httpx.Response(200, json={"input_tokens": 0})

    with _client_with(handler):
        assert await context.count_context_tokens() == 42

    assert model.call_count == 1


async def test_zero_probe_count_falls_back_to_an_estimate(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)

    probe = MagicMock()
    probe.ainvoke = AsyncMock(return_value=SimpleNamespace(usage_metadata={"input_tokens": 0}))
    model = MagicMock()
    model._use_responses_api.return_value = False
    model.bind.return_value = probe
    context.bot_data["llm"] = model

    def handler(request):
        return httpx.Response(404)

    with _client_with(handler):
        tokens = await context.count_context_tokens()

    assert tokens > 0
    assert context.token_count_is_exact is False


def test_settings_setter_updates_bot_data(make_context, make_settings):
    context = make_context("unused")
    new_settings = make_settings(default_model="other:latest")

    context.settings = new_settings

    assert context.bot_data["settings"] is new_settings


def test_chat_message_history_setter(make_context):
    context = make_context("unused")
    history = [HumanMessage(content="x")]

    context.chat_message_history = history

    assert context.chat_message_history is history


def test_decide_autoreply_prompt_comes_from_settings(make_context, sample_settings):
    context = make_context("unused")

    assert context.settings.system_prompts.decide_autoreply == sample_settings.system_prompts.decide_autoreply


def test_bot_llm_loads_lazily(make_context):
    context = make_context("unused")
    del context.bot_data["llm"]
    del context.bot_data["model_name"]

    with patch("sector_ai.sector_context.ChatOllama") as chat_ollama:
        model = context.bot_llm

    assert model is chat_ollama.return_value
    assert context.bot_data["llm"] is model
    assert context.bot_data["model_name"] == context.settings.default_model


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


# --- context window detection ---


async def test_context_length_prefers_the_setting(make_context, make_settings):
    context = make_context("unused", settings=make_settings(context_length=4096))

    assert await context.get_context_length() == 4096


async def test_context_length_reads_the_openai_context_length(make_context, make_settings):
    settings = make_settings(llm_provider="openai", openai_base_url="https://halogen.example/v1")
    context = make_context("unused", settings=settings)
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "fake-model", "context_length": 262144}]})

    with _client_with(handler):
        assert await context.get_context_length() == 262144

    assert seen == ["https://halogen.example/v1/models"]


async def test_context_length_reads_max_model_len(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "fake-model", "max_model_len": 32768}]})

    with _client_with(handler):
        assert await context.get_context_length() == 32768


async def test_context_length_prefers_a_loaded_window_over_a_training_one(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)

    def handler(request):
        return httpx.Response(
            200,
            json={"data": [{"id": "fake-model", "meta": {"n_ctx": 8192, "n_ctx_train": 131072}}]},
        )

    with _client_with(handler):
        assert await context.get_context_length() == 8192


async def test_context_length_ignores_other_models(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "other-model", "context_length": 4096}]})

    with _client_with(handler):
        assert await context.get_context_length() is None


async def test_context_length_uses_the_running_ollama_model(make_context):
    context = make_context("unused")
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": [{"model": "fake-model", "context_length": 16384}]})
        return httpx.Response(500)

    with _client_with(handler):
        assert await context.get_context_length() == 16384

    assert seen == ["/api/ps"]


async def test_context_length_reads_ollama_model_info(make_context):
    context = make_context("unused")

    def handler(request):
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"model_info": {"qwen3.context_length": 32768}})

    with _client_with(handler):
        assert await context.get_context_length() == 32768


async def test_context_length_reads_the_ollama_num_ctx_parameter(make_context):
    context = make_context("unused")

    def handler(request):
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"model_info": {}, "parameters": "num_ctx 8192\nstop <|end|>"})

    with _client_with(handler):
        assert await context.get_context_length() == 8192


async def test_unknown_context_length_is_cached(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "fake-model"}]})

    with _client_with(handler):
        assert await context.get_context_length() is None
        assert await context.get_context_length() is None

    assert len(calls) == 1


async def test_context_budget_falls_back_to_the_default(make_context, monkeypatch):
    context = make_context("unused")
    monkeypatch.setattr(context, "get_context_length", AsyncMock(return_value=None))

    assert await context.get_context_budget() == 8192


async def test_ollama_num_ctx_wins_over_training_capacity(make_context):
    context = make_context("unused")

    def handler(request):
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": []})
        return httpx.Response(
            200,
            json={"model_info": {"qwen3.context_length": 131072}, "parameters": "num_ctx 2048"},
        )

    with _client_with(handler):
        assert await context.get_context_length() == 2048


async def test_provisional_ollama_capacity_is_refreshed(make_context, monkeypatch):
    context = make_context("unused")

    def cold(request):
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"model_info": {"qwen3.context_length": 131072}})

    with _client_with(cold):
        assert await context.get_context_length() == 131072

    def loaded(request):
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": [{"model": "fake-model", "context_length": 4096}]})
        return httpx.Response(500)

    monkeypatch.setattr("sector_ai.sector_context.time", lambda: __import__("time").time() + 61)
    with _client_with(loaded):
        assert await context.get_context_length() == 4096


def test_estimate_is_cautious_by_script_without_failing_ordinary_chats(make_context):
    context = make_context("unused")

    def estimate(text):
        return context._estimate_token_count([HumanMessage(content=text)])

    # Above typical tokenizer rates for each script, so a prompt is never
    # assumed to fit because of its character count...
    assert estimate("你" * 100) == 150
    assert estimate("🔥" * 100) == 300
    # ...but not fourfold: a 2,000-character Russian message (about 1-2k real
    # tokens) fits an 8k window instead of failing the turn.
    assert estimate("Привет" * 333) == 2498
    assert estimate("hello " * 100) == 150


async def test_an_estimate_just_over_the_line_is_confirmed_by_an_exact_count(make_context, make_settings, monkeypatch):
    context = make_context("unused", settings=make_settings(context_length=1000))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=600))
    context.conversation_data["token_count"] = {"fingerprint": None, "tokens": 600, "exact": True}

    assert await context.prompt_exceeds(950, 1000) is False
    # Over the whole window, no probe is sent at all.
    assert await context.prompt_exceeds(1500, 1000) is True
    assert context.count_context_tokens.await_count == 1


async def test_a_transient_context_length_failure_is_not_cached(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        raise httpx.ConnectError("no route", request=request)

    with _client_with(handler):
        assert await context.get_context_length() is None
        assert await context.get_context_length() is None

    assert len(calls) == 2


async def test_a_missing_context_length_route_is_cached(make_context, make_settings):
    settings = make_settings(llm_provider="openai")
    context = make_context("unused", settings=settings)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(404)

    with _client_with(handler):
        assert await context.get_context_length() is None
        assert await context.get_context_length() is None

    assert len(calls) == 1


async def test_context_length_is_looked_up_again_after_a_key_change(make_context, make_settings):
    context = make_context("unused", settings=make_settings(llm_provider="openai", openai_api_key="sk-one"))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "fake-model", "context_length": 4096}]})

    with _client_with(handler):
        assert await context.get_context_length() == 4096
        context.settings = make_settings(llm_provider="openai", openai_api_key="sk-two")
        assert await context.get_context_length() == 4096

    assert len(calls) == 2


def test_bot_llm_rebuilds_when_settings_change(make_context, make_settings):
    context = make_context("unused", settings=make_settings(ollama_url="http://one:11434"))

    with patch("sector_ai.sector_context.ChatOllama") as first:
        context.load_model()
        assert context.bot_llm is first.return_value

    context.settings = make_settings(ollama_url="http://two:11434")

    with patch("sector_ai.sector_context.ChatOllama") as second:
        assert context.bot_llm is second.return_value


def test_a_reply_chain_in_an_ordinary_group_stays_in_the_main_conversation(make_context, make_update):
    context = make_context()
    main = make_update(text="hello")
    context.bind_update(main)
    context.conversation_data["marker"] = "main"

    reply = make_update(text="a reply", thread_id=77)
    reply.message.is_topic_message = False
    context.bind_update(reply)

    assert context.conversation_data.get("marker") == "main"
    assert "topics" not in context.chat_data


def test_a_forum_topic_gets_its_own_conversation(make_context, make_update):
    context = make_context()
    context.bind_update(make_update(text="hello"))
    context.conversation_data["marker"] = "main"

    context.bind_update(make_update(text="in a topic", thread_id=77))

    assert "marker" not in context.conversation_data


def test_the_token_count_cache_survives_a_clock_change(make_context, make_update):
    context = make_context("unused")
    context.bind_update(make_update(text="hello"))
    first = context._token_count_fingerprint(context.context_metadata_messages())
    context._context_time = "2099-01-01T00:00:00Z"

    assert context._token_count_fingerprint(context.context_metadata_messages()) == first


def test_system_prompt_and_autoreply_are_per_topic(make_context, make_update):
    context = make_context()
    context.bind_update(make_update(text="main"))
    context.chat_system_prompt = "main prompt"
    context.chat_autoreply_mode = False

    context.bind_update(make_update(text="topic", thread_id=22))
    assert context.chat_system_prompt == context.settings.system_prompts.default
    assert context.chat_autoreply_mode is True
    context.chat_system_prompt = "topic prompt"

    context.bind_update(make_update(text="main again"))
    assert context.chat_system_prompt == "main prompt"
    assert context.chat_autoreply_mode is False


def test_a_compacted_shared_event_does_not_come_back(make_context):
    context = make_context()
    context.record_shared_chat_event(HumanMessage(content="bot was promoted", additional_kwargs={"telegram": {}}))
    # Compaction folds it into the summary.
    context.chat_message_history = []

    assert context.shared_chat_events() == []


def test_a_split_bot_message_keeps_its_speaker_for_the_summary():
    from sector_ai.sector_context import SectorContext as Context

    chunk, rest = Context._take_summary_chunk([AIMessage(content="x" * 4000)], 200)

    assert isinstance(chunk[0], AIMessage)
    assert all(isinstance(item, AIMessage) for item in rest)


def test_uncapped_photos_need_compaction(make_settings):
    import pytest as _pytest
    from pydantic import ValidationError

    with _pytest.raises(ValidationError, match="never released"):
        make_settings(image_max_attached=0, compaction_enabled=False)


def test_script_weights_cover_scripts_that_tokenize_poorly(make_context):
    context = make_context("unused")

    def estimate(text):
        return context._estimate_token_count([HumanMessage(content=text)])

    assert estimate("Α" * 100) == 200  # Greek uppercase
    assert estimate("ກ" * 100) == 250  # Lao: byte fallback
    assert estimate("ע" * 100) == 175  # Hebrew


def test_abandoned_placeholders_are_bounded(make_context):
    context = make_context()
    for message_id in range(40):
        context.mark_placeholder_unsettled(SimpleNamespace(message_id=message_id, chat_id=1))

    assert len(context.conversation_data["abandoned_placeholders"]) == 16
