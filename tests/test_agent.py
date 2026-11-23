"""Graph contracts: real model tool-call messages, command history and safety."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from telegram.error import NetworkError

from sector_ai.agent import run_agent
from sector_ai.coding import html_cmd, svg_cmd
from sector_ai.poll import poll_cmd
from sector_ai.sector_context import SectorContext
from sector_ai.tools import AgentTool

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect width="20" height="20" fill="blue"/></svg>'
POLL = {"question": "Which color?", "options": ["Blue", "Green"]}


class ToolModel(BaseChatModel):
    responses: list[AIMessage]
    received: list = []
    received_kwargs: list = []

    @property
    def _llm_type(self):
        return "test-tools"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        self.received_kwargs.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected extra model invocation")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name, prompt="choose a color", identifier="call-1", **args):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {"prompt": prompt},
                "id": identifier,
                "type": "tool_call",
            }
        ],
    )


@pytest.fixture
def agent_context(make_context, make_settings):
    def make(*responses, **settings):
        return make_context(
            model=ToolModel(responses=list(responses)),
            settings=make_settings(
                context_length=8192, streaming_interval_sec=999, streaming_chunk_size=999999, **settings
            ),
        )

    return make


async def test_autonomous_poll_uses_shared_capability_and_records_exact_history(agent_context, make_update):
    context = agent_context(tool_call("create_poll"), AIMessage(content=json.dumps(POLL)), AIMessage(content=""))
    update = make_update(text="/chat ask us to choose a color")
    await run_agent(update, context, force=True)

    assert update.message.calls_to("reply_poll")[0][0] == (POLL["question"], POLL["options"])
    assert [type(message) for message in context.chat_message_history] == [HumanMessage, AIMessage, ToolMessage]
    receipt = json.loads(context.chat_message_history[-1].content)
    assert receipt["question"] == POLL["question"]
    assert receipt["options"] == POLL["options"]
    assert receipt["status"] == "delivered"
    assert update.message.calls_to("delete")
    assert all("empty response" not in text for text in update.message.texts("edit_text"))
    # The post-tool model call sees a matched call/result pair; the volatile
    # snapshot is appended after the retained history.
    assert isinstance(context.bot_llm.received[-1][-2], ToolMessage)


async def test_new_turn_removes_an_abandoned_placeholder(agent_context, make_update):
    context = agent_context(AIMessage(content="done"))
    context.conversation_data["pending_placeholder"] = {"chat_id": 1, "message_id": 42}

    await run_agent(make_update(text="hello"), context, force=True)

    context.bot.delete_message.assert_awaited_once_with(chat_id=1, message_id=42)
    assert "pending_placeholder" not in context.conversation_data


async def test_successful_turn_forgets_its_own_placeholder(agent_context, make_update):
    context = agent_context(AIMessage(content="done"))

    await run_agent(make_update(text="hello"), context, force=True)

    assert "pending_placeholder" not in context.conversation_data
    assert "placeholder_unsettled" not in context.conversation_data


async def test_reaction_only_turn_sets_reaction_and_removes_the_placeholder(agent_context, make_update):
    context = agent_context(
        tool_call("react_to_message", emoji="👍", identifier="call-r"),
        AIMessage(content=""),
    )
    update = make_update(text="react to this", message_id=77)

    await run_agent(update, context, force=True)

    context.bot.set_message_reaction.assert_awaited_once()
    kwargs = context.bot.set_message_reaction.await_args.kwargs
    assert kwargs["chat_id"] == 1
    assert kwargs["message_id"] == 77
    assert kwargs["reaction"][0].emoji == "👍"
    receipt = json.loads(context.chat_message_history[-1].content)
    assert receipt["status"] == "delivered"
    assert receipt["reaction"]["emoji"] == "👍"
    assert update.message.calls_to("delete")
    assert update.message.texts("edit_text") == []


async def test_text_and_reaction_in_one_response_keep_both(agent_context, make_update):
    response = AIMessage(
        content="Nice one 👍",
        tool_calls=[{"name": "react_to_message", "args": {"emoji": "🔥"}, "id": "call-r", "type": "tool_call"}],
    )
    context = agent_context(response, AIMessage(content=""))
    update = make_update(text="nice", message_id=77)

    await run_agent(update, context, force=True)

    assert any("Nice one" in text for text in update.message.texts("edit_text"))
    context.bot.set_message_reaction.assert_awaited_once()
    assert context.bot.set_message_reaction.await_args.kwargs["message_id"] == 77
    assert update.message.calls_to("delete") == []


async def test_tools_are_bound_in_stable_name_order(agent_context, make_update, monkeypatch):
    specs = [
        AgentTool("z_tool", "Last", {"type": "object"}, AsyncMock(return_value={"status": "success"}), 1),
        AgentTool("a_tool", "First", {"type": "object"}, AsyncMock(return_value={"status": "success"}), 1),
    ]
    monkeypatch.setattr("sector_ai.agent.builtin_tools", lambda *args, **kwargs: specs)
    context = agent_context(AIMessage(content="done"))

    await run_agent(make_update(text="hello"), context, force=True)

    bound = context.bot_llm.received_kwargs[0]["tools"]
    assert [tool["function"]["name"] for tool in bound] == ["a_tool", "z_tool"]
    assert [schema["function"]["name"] for schema in context.conversation_data["tool_schemas"]] == [
        "a_tool",
        "z_tool",
    ]


@pytest.mark.parametrize(
    "command,source,key",
    [
        (poll_cmd, json.dumps(POLL), "question"),
        (svg_cmd, SVG, "svg"),
        (html_cmd, "<html><body>Blue</body></html>", "html"),
    ],
)
async def test_explicit_commands_enter_history_and_skip_engagement(
    command,
    source,
    key,
    agent_context,
    make_update,
    monkeypatch,
):
    context = agent_context(AIMessage(content=source))
    context.chat_autoreply_mode = False
    decision = AsyncMock(side_effect=AssertionError("Commands must bypass engagement"))
    monkeypatch.setattr("sector_ai.agent.should_respond", decision)
    update = make_update(text=f"/{ {poll_cmd: 'poll', svg_cmd: 'svg', html_cmd: 'html'}[command] } make something blue")
    await command(update, context)
    assert context.chat_message_history[0].content.startswith("testuser: /")
    call, result = context.chat_message_history[-2:]
    assert call.additional_kwargs["origin"].startswith("/")
    assert result.tool_call_id == call.tool_calls[0]["id"]
    assert key in json.loads(result.content)
    assert len(context.bot_llm.received) == 1
    decision.assert_not_awaited()


async def test_poll_followup_receives_the_exact_options(agent_context, make_update):
    context = agent_context(AIMessage(content=json.dumps(POLL)), AIMessage(content="Blue and Green."))
    await poll_cmd(make_update(text="/poll colors"), context)
    await run_agent(make_update(text="What were the options?"), context, force=True)
    sent = context.bot_llm.received[-1]
    result = next(message for message in sent if isinstance(message, ToolMessage))
    assert json.loads(result.content.split("\n", 1)[1])["options"] == ["Blue", "Green"]


async def test_svg_revision_generator_receives_prior_source(agent_context, make_update):
    context = agent_context(AIMessage(content=SVG), AIMessage(content=SVG))
    await svg_cmd(make_update(text="/svg a square"), context)
    await svg_cmd(make_update(text="/svg make it blue"), context)
    assert any(json.dumps(SVG)[1:-1] in message.content for message in context.bot_llm.received[-1])


async def test_uncertain_delivery_is_not_retried_and_preserves_content(agent_context, make_update):
    context = agent_context(AIMessage(content=json.dumps(POLL)))
    update = make_update(text="/poll colors")
    update.message.reply_poll = AsyncMock(side_effect=NetworkError("connection lost"))
    await poll_cmd(update, context)
    update.message.reply_poll.assert_awaited_once()
    result = json.loads(context.chat_message_history[-1].content)
    assert result["status"] == "delivery_uncertain"
    assert result["options"] == POLL["options"]


async def test_repeated_tool_request_does_not_post_twice(agent_context, make_update):
    context = agent_context(
        tool_call("create_poll"),
        AIMessage(content=json.dumps(POLL)),
        tool_call("create_poll", identifier="call-2"),
        AIMessage(content=""),
    )
    update = make_update(text="make a poll")
    await run_agent(update, context, force=True)
    assert len(update.message.calls_to("reply_poll")) == 1
    assert len([message for message in context.chat_message_history if isinstance(message, ToolMessage)]) == 2


async def test_update_deduplication_prevents_reposting(agent_context, make_update):
    context = agent_context(AIMessage(content=json.dumps(POLL)))
    update = make_update(text="/poll colors")
    update.update_id = 987
    await poll_cmd(update, context)
    await poll_cmd(update, context)
    assert len(update.message.calls_to("reply_poll")) == 1
    assert len(context.chat_message_history) == 3


async def test_forbidden_tool_is_not_executed(agent_context, make_update):
    context = agent_context(tool_call("shell"), AIMessage(content="That tool is unavailable."))
    update = make_update(text="do something")
    await run_agent(update, context, force=True)
    result = next(message for message in context.chat_message_history if isinstance(message, ToolMessage))
    assert result.status == "error"
    assert "not permitted" in result.content


async def test_invalid_arguments_return_error_without_delivery(agent_context, make_update):
    context = agent_context(tool_call("create_poll", bogus=True), AIMessage(content="Please give a question."))
    update = make_update(text="poll")
    await run_agent(update, context, force=True)
    assert update.message.calls_to("reply_poll") == []
    result = next(message for message in context.chat_message_history if isinstance(message, ToolMessage))
    assert "Invalid tool arguments" in result.content


async def test_tool_budget_closes_all_calls(agent_context, make_update):
    first = tool_call("create_poll")
    first.tool_calls.append(
        {"name": "generate_html", "args": {"prompt": "a page"}, "id": "call-2", "type": "tool_call"}
    )
    context = agent_context(
        first, AIMessage(content=json.dumps(POLL)), AIMessage(content="Done."), agent_max_tool_calls=1
    )
    update = make_update(text="make a poll and page")
    await run_agent(update, context, force=True)
    assert len(update.message.calls_to("reply_poll")) == 1
    assert update.message.calls_to("reply_document") == []
    results = [message for message in context.chat_message_history if isinstance(message, ToolMessage)]
    assert len(results) == 2
    assert "limit reached" in results[1].content


async def test_iteration_budget_stops_the_graph(agent_context, make_update):
    context = agent_context(
        tool_call("create_poll"),
        AIMessage(content=json.dumps(POLL)),
        agent_max_iterations=1,
    )
    update = make_update(text="make polls forever")
    await run_agent(update, context, force=True)
    assert "iteration limit" in update.message.texts("edit_text")[-1]
    assert isinstance(context.chat_message_history[-2], ToolMessage)


async def test_plain_chat_fallback_projects_command_tool_history(make_context, make_settings, make_update):
    context = make_context(json.dumps(POLL), "Blue and Green.", settings=make_settings(context_length=8192))
    await poll_cmd(make_update(text="/poll colors"), context)
    await run_agent(make_update(text="What were the options?"), context, force=True)
    sent = context.bot_llm.received[-1]
    assert not any(isinstance(message, ToolMessage) for message in sent)
    assert any("Blue" in message.content and "Green" in message.content for message in sent)


async def test_topic_isolation(agent_context, make_update):
    context = agent_context(AIMessage(content=json.dumps(POLL)))
    topic = make_update(text="/poll colors")
    topic.message.message_thread_id = 42
    await poll_cmd(topic, context)
    assert len(context.chat_message_history) == 3
    context.bind_update(make_update(text="another topic"))
    assert context.chat_message_history == []
    context.bind_update(topic)
    assert len(context.chat_message_history) == 3


async def test_mcp_result_returns_to_model_and_tool_access_is_scoped(agent_context, make_update):
    invocation = AsyncMock(return_value={"status": "success", "result": "Sunny", "source": "https://example.org"})
    spec = AgentTool("mcp_weather__lookup", "Read weather", {"type": "object"}, invocation, 1)
    registry = SimpleNamespace(tools_for_chat=AsyncMock(return_value=[spec]))
    context = agent_context(
        tool_call("mcp_weather__lookup", location="NYC"),
        AIMessage(content="It is sunny."),
        mcp_config="unused.json",
    )
    context.bot_data["mcp_registry"] = registry
    await run_agent(make_update(text="weather?"), context, force=True)
    registry.tools_for_chat.assert_awaited_once_with(1)
    invocation.assert_awaited_once_with({"location": "NYC"})
    assert "Sunny" in context.bot_llm.received[-1][-2].content


async def test_run_timeout_closes_pending_tool_call(agent_context, make_update, monkeypatch):
    blocked = asyncio.Event()
    entered = asyncio.Event()

    async def hang(arguments):
        entered.set()
        await blocked.wait()

    spec = AgentTool("create_poll", "poll", {"type": "object"}, hang, 10)
    monkeypatch.setattr("sector_ai.agent.builtin_tools", lambda *args, **kwargs: [spec])
    context = agent_context(AIMessage(content="unused"), agent_timeout=1)
    await poll_cmd(make_update(text="/poll colors"), context)
    assert entered.is_set()
    assert isinstance(context.chat_message_history[-1], ToolMessage)
    assert json.loads(context.chat_message_history[-1].content)["status"] == "execution_uncertain"


async def test_requests_in_same_conversation_are_serialized(agent_context, make_update, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    async def plain(update, context, *args, **kwargs):
        order.append(update.message.text)
        if len(order) == 1:
            entered.set()
            await release.wait()

    monkeypatch.setattr("sector_ai.chat.stream_chat", plain)
    context = agent_context(agent_tools_enabled=False)
    other = SectorContext(context.application, chat_id=1)
    first = asyncio.create_task(run_agent(make_update(text="first"), context, force=True))
    await entered.wait()
    second = asyncio.create_task(run_agent(make_update(text="second"), other, force=True))
    release.set()
    await asyncio.gather(first, second)
    assert order == ["first", "second"]


async def test_uncertain_side_effect_blocks_further_tools_in_same_batch(agent_context, make_update):
    request = tool_call("create_poll")
    request.tool_calls.append(
        {
            "name": "generate_html",
            "args": {"prompt": "a page"},
            "id": "two",
            "type": "tool_call",
        }
    )
    context = agent_context(
        request, AIMessage(content=json.dumps(POLL)), AIMessage(content="I could not confirm delivery.")
    )
    update = make_update(text="poll and page")
    update.message.reply_poll = AsyncMock(side_effect=NetworkError("timeout"))
    await run_agent(update, context, force=True)
    assert update.message.calls_to("reply_document") == []
    results = [json.loads(item.content) for item in context.chat_message_history if isinstance(item, ToolMessage)]
    assert results[0]["status"] == "delivery_uncertain"
    assert "blocked" in results[1]["error"]
    assert "tools" not in context.bot_llm.received_kwargs[-1]


async def test_disabled_autoreply_records_without_model_calls(agent_context, make_update, monkeypatch):
    context = agent_context()
    context.chat_autoreply_mode = False
    decision = AsyncMock(side_effect=AssertionError("disabled"))
    # The function itself observes autoreply=off; it must not call make_decision.
    monkeypatch.setattr("sector_ai.autoreply.make_decision", decision)
    update = make_update(text="hello everyone")
    await run_agent(update, context)
    decision.assert_not_awaited()
    assert len(context.chat_message_history) == 1
    assert update.message.calls == []


async def test_direct_mention_bypasses_decision(agent_context, make_update, monkeypatch):
    context = agent_context(AIMessage(content="Hello!"))
    decision = AsyncMock(side_effect=AssertionError("mention should bypass"))
    monkeypatch.setattr("sector_ai.agent.should_respond", decision)
    await run_agent(make_update(text="@testbot hello"), context)
    decision.assert_not_awaited()
    assert context.chat_message_history[-1].content == "Hello!"


async def test_unsupported_tool_endpoint_falls_back_before_any_action(agent_context, make_update):
    class UnsupportedToolError(Exception):
        status_code = 400

    class RejectingModel(ToolModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if "tools" in kwargs:
                raise UnsupportedToolError("This model does not support tools")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    context = agent_context()
    context.bot_data["llm"] = RejectingModel(responses=[AIMessage(content="Plain answer.")])
    update = make_update(text="hello")
    await run_agent(update, context, force=True)
    assert context.chat_message_history[-1].content == "Plain answer."
    assert update.message.texts("edit_text")[-1] == "Plain answer."
    assert context.bot_data["non_tool_models"]


async def test_model_selection_is_frozen_for_each_request(agent_context, make_update, monkeypatch):
    context = agent_context(agent_tools_enabled=False)
    original = context.bot_llm

    async def stream(update, ctx, *args, **kwargs):
        ctx.bot_data["model_name"] = "changed-in-another-chat"
        ctx.bot_data["llm"] = ToolModel(responses=[])
        assert ctx.get_model() == "llama3.2-vision:11b-instruct-fp16"
        assert ctx.bot_llm is original

    # ToolModel has no name, so SectorContext used the configured default.
    monkeypatch.setattr("sector_ai.chat.stream_chat", stream)
    await run_agent(make_update(text="hi"), context, force=True)
    assert context.get_model() == "changed-in-another-chat"


async def test_remote_schema_refs_cannot_trigger_network_access(agent_context, make_update):
    invoke = AsyncMock()
    spec = AgentTool("remote_ref", "bad schema", {"$ref": "https://never-fetch.invalid/schema"}, invoke, 1)
    context = agent_context(
        tool_call("remote_ref"), AIMessage(content="That tool has an invalid schema."), mcp_config="unused"
    )
    context.bot_data["mcp_registry"] = SimpleNamespace(tools_for_chat=AsyncMock(return_value=[spec]))
    await run_agent(make_update(text="try it"), context, force=True)
    invoke.assert_not_awaited()
    result = next(item for item in context.chat_message_history if isinstance(item, ToolMessage))
    assert result.status == "error"


async def test_legacy_content_commands_record_requests_and_responses(make_context, make_settings, make_update):
    from sector_ai.decision import decide_cmd
    from sector_ai.emoji import emoji_cmd

    context = make_context('{"result": true}', "🎉", settings=make_settings(context_length=8192))
    await decide_cmd(make_update(text="/decide is this fun?"), context)
    await emoji_cmd(make_update(text="/emoji celebrate"), context)
    assert context.chat_message_history[0].content == "testuser: /decide is this fun?"
    assert context.chat_message_history[1].content == "True"
    assert context.chat_message_history[2].content == "testuser: /emoji celebrate"
    assert context.chat_message_history[3].content == "🎉"


class ChunkedToolModel(ToolModel):
    chunks: list[AIMessageChunk]

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)


@pytest.mark.parametrize(
    "arguments,finish_reason",
    [
        ('{"prompt": "only the beginning', "length"),
        ('{"prompt": "only the beginning', "tool_calls"),
        ('{"prompt": "only the beginning', None),
        ('{"prompt": "complete"}', "length"),
        ('{"prompt": "complete"}', "content_filter"),
        ('{"prompt": "complete"}', "refusal"),
    ],
)
async def test_incomplete_or_aborted_tool_stream_never_executes(arguments, finish_reason, agent_context, make_update):
    invocation = AsyncMock(return_value={"status": "success"})
    spec = AgentTool("mcp_demo__write", "Write", {"type": "object"}, invocation, 1)
    context = agent_context(mcp_config="unused", agent_max_iterations=1)
    context.bot_data["mcp_registry"] = SimpleNamespace(tools_for_chat=AsyncMock(return_value=[spec]))
    context.bot_data["llm"] = ChunkedToolModel(
        responses=[],
        chunks=[
            AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": spec.name, "args": arguments, "id": "partial", "index": 0}],
                response_metadata={"finish_reason": finish_reason} if finish_reason else {},
            )
        ],
    )
    update = make_update(text="write something")
    await run_agent(update, context, force=True)
    invocation.assert_not_awaited()
    assert not any(isinstance(item, AIMessage) and item.tool_calls for item in context.chat_message_history)


async def test_complete_split_tool_stream_executes_exact_arguments(agent_context, make_update):
    invocation = AsyncMock(return_value={"status": "success"})
    spec = AgentTool("mcp_demo__write", "Write", {"type": "object"}, invocation, 1)
    context = agent_context(mcp_config="unused", agent_max_iterations=1)
    context.bot_data["mcp_registry"] = SimpleNamespace(tools_for_chat=AsyncMock(return_value=[spec]))
    context.bot_data["llm"] = ChunkedToolModel(
        responses=[],
        chunks=[
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": spec.name, "args": '{"prompt":', "id": "whole", "index": 0},
                ],
            ),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": ' "the entire request"}', "id": None, "index": 0},
                ],
                response_metadata={"finish_reason": "tool_calls"},
            ),
        ],
    )
    await run_agent(make_update(text="write something"), context, force=True)
    invocation.assert_awaited_once_with({"prompt": "the entire request"})


async def test_real_openai_adapter_accepts_normalized_tool_call_completion(agent_context, make_update):
    from langchain_openai import ChatOpenAI

    invocation = AsyncMock(return_value={"status": "success"})
    spec = AgentTool("mcp_demo__write", "Write", {"type": "object"}, invocation, 1)
    context = agent_context(mcp_config="unused", agent_max_iterations=1)
    context.bot_data["mcp_registry"] = SimpleNamespace(tools_for_chat=AsyncMock(return_value=[spec]))
    events = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "complete",
                                "type": "function",
                                "function": {"name": spec.name, "arguments": '{"prompt": "the entire request"}'},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {},
                }
            ],
        },
    ]

    def respond(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        context.bot_data["llm"] = ChatOpenAI(
            model="test-model",
            api_key="synthetic-unit-test-key",
            base_url="https://test.invalid/v1",
            http_async_client=client,
            max_retries=0,
            use_responses_api=False,
        )
        await run_agent(make_update(text="write something"), context, force=True)
    invocation.assert_awaited_once_with({"prompt": "the entire request"})
    response = next(item for item in context.chat_message_history if isinstance(item, AIMessage) and item.tool_calls)
    assert response.response_metadata["normalized_finish_reason"] == "tool_call"


@pytest.mark.parametrize("status,max_calls", [("success", 1), ("execution_uncertain", 6)])
async def test_post_tool_plain_fallback_does_not_send_oversized_context(
    status, max_calls, make_context, make_settings, make_update, monkeypatch
):
    model = ToolModel(responses=[tool_call("mcp_demo__read"), AIMessage(content="Short summary.")])
    context = make_context(
        model=model,
        settings=make_settings(
            context_length=2000,
            agent_max_tool_calls=max_calls,
            mcp_config="unused",
            streaming_interval_sec=999,
            streaming_chunk_size=999999,
        ),
    )
    invocation = AsyncMock(return_value={"status": status, "result": "x" * 12000})
    spec = AgentTool("mcp_demo__read", "Read", {"type": "object"}, invocation, 1)
    context.bot_data["mcp_registry"] = SimpleNamespace(tools_for_chat=AsyncMock(return_value=[spec]))
    monkeypatch.setattr(context, "count_context_tokens", AsyncMock(return_value=4000))
    compact = AsyncMock(wraps=context.maybe_compact)
    monkeypatch.setattr(context, "maybe_compact", compact)
    update = make_update(text="read it")
    await run_agent(update, context, force=True)
    invocation.assert_awaited_once()
    assert all(context._estimate_token_count(messages) <= 1800 for messages in model.received)
    assert compact.await_count >= 2
    assert "context window" in update.message.texts("edit_text")[-1]


def test_interruption_cleanup_uses_only_current_transaction(agent_context):
    from sector_ai.agent import _close_pending

    earlier = tool_call("create_poll", identifier="reused")
    current = tool_call("create_poll", identifier="reused")
    context = agent_context()
    context.chat_message_history = [
        earlier,
        ToolMessage(content="earlier result", tool_call_id="reused"),
        current,
    ]
    _close_pending(context, current)
    assert len(context.chat_message_history) == 4
    assert isinstance(context.chat_message_history[-1], ToolMessage)
    assert context.chat_message_history[-1].tool_call_id == "reused"


def test_interruption_cleanup_attributes_delivery_only_to_the_active_call(agent_context):
    from sector_ai.agent import _close_pending

    pending = tool_call("generate_html", identifier="html-call", prompt="a page")
    pending.tool_calls.append(
        {"name": "create_poll", "args": {"prompt": "a poll"}, "id": "poll-call", "type": "tool_call"}
    )
    context = agent_context()
    context.chat_message_history = [pending]
    context.conversation_data["interrupted_delivery"] = {
        "tool_call_id": "html-call",
        "result": {"kind": "html", "html": "<html>page</html>", "status": "delivery_uncertain"},
    }
    _close_pending(context, pending)
    results = {message.tool_call_id: json.loads(message.content) for message in context.chat_message_history[1:]}
    assert results["html-call"]["status"] == "delivery_uncertain"
    assert results["html-call"]["html"] == "<html>page</html>"
    assert results["poll-call"]["status"] == "not_executed"
    assert "html" not in results["poll-call"]
    assert "interrupted_delivery" not in context.conversation_data
    assert "active_tool_call_id" not in context.conversation_data


async def test_run_timeout_does_not_copy_interrupted_delivery_to_unstarted_tools(
    agent_context, make_update, monkeypatch
):
    from sector_ai.capabilities import GeneratedOutput, deliver_output

    html_started = asyncio.Event()
    poll = AsyncMock(return_value={"status": "delivered"})

    async def stalled_html(arguments):
        html_started.set()
        output = GeneratedOutput("html", {"html": "<html>page</html>"}, arguments["prompt"])
        await deliver_output(output, update, context)

    specs = [
        AgentTool("generate_html", "html", {"type": "object"}, stalled_html, 30),
        AgentTool("create_poll", "poll", {"type": "object"}, poll, 30),
    ]
    monkeypatch.setattr("sector_ai.agent.builtin_tools", lambda *args, **kwargs: specs)
    request = tool_call("generate_html", identifier="html-call", prompt="a page")
    request.tool_calls.append(
        {"name": "create_poll", "args": {"prompt": "a poll"}, "id": "poll-call", "type": "tool_call"}
    )
    context = agent_context(request, agent_timeout=0.2)
    update = make_update(text="make a page and a poll")

    async def blocked_reply(*args, **kwargs):
        html_started.set()
        await asyncio.Event().wait()

    update.message.reply_document = blocked_reply
    await run_agent(update, context, force=True)
    await asyncio.wait_for(html_started.wait(), timeout=1)
    poll.assert_not_awaited()
    results = {
        message.tool_call_id: json.loads(message.content)
        for message in context.chat_message_history
        if isinstance(message, ToolMessage)
    }
    assert results["html-call"]["status"] == "delivery_uncertain"
    assert results["html-call"]["html"] == "<html>page</html>"
    assert results["poll-call"]["status"] == "not_executed"
    assert "html" not in results["poll-call"]
    assert "interrupted_delivery" not in context.conversation_data
    assert "active_tool_call_id" not in context.conversation_data


async def test_new_tool_transaction_does_not_reuse_historical_ids(agent_context, make_update):
    context = agent_context(
        tool_call("create_poll", identifier="reused"),
        AIMessage(content=json.dumps(POLL)),
        AIMessage(content=""),
    )
    context.chat_message_history = [
        tool_call("create_poll", identifier="reused"),
        ToolMessage(content="earlier result", tool_call_id="reused"),
    ]
    await run_agent(make_update(text="make a poll"), context, force=True)
    calls = [item for item in context.chat_message_history if isinstance(item, AIMessage) and item.tool_calls]
    assert len(calls) == 2
    assert calls[0].tool_calls[0]["id"] != calls[1].tool_calls[0]["id"]
