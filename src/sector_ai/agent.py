"""One bounded LangGraph workflow for chat, explicit capabilities and MCP.

``conversation`` is the canonical mutable in-process state exposed by
SectorContext. The graph never maintains a second transcript/checkpoint copy.
"""

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TypedDict
from uuid import uuid4

from jsonschema import ValidationError as SchemaError
from jsonschema.validators import validator_for
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, message_chunk_to_message
from langgraph.graph import END, START, StateGraph
from referencing import Registry
from telegram.error import NetworkError, TelegramError

from . import debounce as debounce_turns
from .autoreply import should_respond
from .capabilities import builtin_tools
from .chat_metadata import is_message_edit, message_metadata, utc_now
from .images import load_photos
from .logging_utils import redact_for_user
from .mcp_tools import MCPRegistry
from .prompts import AGENT_SYSTEM_PROMPT
from .streaming_handler import LazyReply, handle_streaming_response
from .tools import result_content
from .typing_indicator import typing

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    conversation: dict
    accepted: bool
    engage: bool
    pending: list[dict]
    iterations: int
    calls: int
    delivered: bool
    finished: bool
    tools_blocked: bool


def unsupported_tools(error: Exception) -> bool:
    if isinstance(error, NotImplementedError):
        return True
    text = str(error).lower()
    return getattr(error, "status_code", None) in {400, 404, 422} and any(
        phrase in text
        for phrase in (
            "does not support tools",
            "tools are not supported",
            "tool calling is not supported",
            "does not support tool",
            "unsupported tool",
            "unknown parameter: tools",
            "unrecognized request argument supplied: tools",
        )
    )


async def run_agent(
    update,
    context,
    *,
    command=None,
    prompt=None,
    force=False,
    ingest=True,
    system_prompt=None,
    save_ai_message=True,
    photos=None,
    debounce=False,
    engage=None,
    edited=False,
    restarts=0,
    turn=None,
) -> None:
    """Run one turn. ``photos`` lists the messages whose photos belong to it (an album).

    ``debounce`` lets a newer relevant message cancel this turn while it has
    not sent anything yet (see :mod:`sector_ai.debounce`). ``engage`` skips the
    engagement decision. ``edited`` runs a turn for an edit that restarted one.
    """
    if is_message_edit(update) and not edited:
        from .chat_events import handle_message_edit

        await handle_message_edit(update, context)
        return
    context.bind_update(update)
    if turn is None and debounce and context.settings.debounce_max_restarts > 0 and not command:
        # Synchronous, so the finally below always unregisters it. A caller
        # may register earlier to claim its place in arrival order.
        turn = debounce_turns.register(update, context, photos=photos, restarts=restarts, edited=edited)
    try:
        if turn is not None:
            if await debounce_turns.take_over(update, context, turn):
                engage = True
            # Every earlier message is stored first, so history keeps arrival
            # order and the answer reads everything it supersedes.
            await debounce_turns.wait_for_earlier(context, turn)
        await _run_turn(
            update,
            context,
            turn,
            command=command,
            prompt=prompt,
            force=force,
            ingest=ingest,
            system_prompt=system_prompt,
            save_ai_message=save_ai_message,
            photos=photos,
            engage=engage,
        )
    finally:
        if turn is not None:
            debounce_turns.leave(context, turn)


async def _run_turn(
    update, context, turn, *, command, prompt, force, ingest, system_prompt, save_ai_message, photos, engage
) -> None:
    async with context.conversation_lock:
        await context.remove_pending_placeholder()
        for key in ("active_tool_call_id", "started_tool_call_ids", "interrupted_delivery"):
            context.conversation_data.pop(key, None)
        model = context.bot_llm
        context._request_model_name = context.get_model()
        context._request_model = model
        tools = {}
        bound_model = None
        # Posted on the first visible text; never an empty placeholder.
        placeholder = None
        # Typing starts once the turn has committed to responding.
        indicator = contextlib.AsyncExitStack()
        executions = {}
        pending_message = None
        model_key = context._model_config(context.get_model())
        unsupported = context.bot_data.setdefault("non_tool_models", set())

        async def ingest_node(state):
            nonlocal update
            accepted = context.accept_update(update) if ingest else True
            if accepted and ingest:
                message = update.effective_message
                reply = getattr(message, "reply_to_message", None)
                if reply and getattr(getattr(reply, "from_user", None), "id", None) == context.bot.id:
                    text = getattr(reply, "text", None)
                    if isinstance(text, str) and text and not context.message_exists(AIMessage(content=text)):
                        context.chat_message_history.append(
                            AIMessage(
                                content=text,
                                additional_kwargs={"telegram": message_metadata(reply)},
                            )
                        )
                include_command = bool(command) or bool(system_prompt)
                if photos:
                    images = await load_photos(photos, context.settings)
                # Read after the last await: an edit that arrived before this
                # point is stored instead of the original.
                edits = turn.edits if turn is not None else {}
                latest = edits.get(getattr(message, "message_id", None), message)
                if latest is not message:
                    # Decide and answer on the edited text, too.
                    message = latest
                    update = _WithMessage(update, latest)
                if photos:
                    album = [edits.get(getattr(item, "message_id", None), item) for item in photos]
                    context.save_user_message(message, include_command=include_command, album=album, images=images)
                else:
                    context.save_user_message(message, include_command=include_command)
            debounce_turns.mark_stored(turn)
            return {"accepted": accepted}

        async def engagement_node(state):
            if turn is not None and turn.superseded:
                # A newer message took over; this one is context for its answer.
                return {"engage": False}
            result = await decide()
            if turn is not None:
                if turn.superseded:
                    return {"engage": False}
                turn.engaged = result
            return {"engage": result}

        async def decide() -> bool:
            if force or command or engage is True:
                return True
            if engage is False:
                return False
            message = update.effective_message
            date = getattr(message, "date", None)
            if isinstance(date, datetime) and datetime.now(UTC) - date > timedelta(minutes=2):
                return False
            # An explicitly addressed bot should not run an LLM just to decide
            # whether it was addressed. Respect autoreply=off for mentions.
            if await debounce_turns.addressed_to_bot(update, context):
                return True
            return bool(await should_respond(update, context))

        async def prepare_node(state):
            nonlocal bound_model, placeholder, pending_message
            await indicator.enter_async_context(typing(update.effective_message))
            if not command:
                placeholder = LazyReply(context, update.effective_message)
            specs = builtin_tools(update, context, model, topic=command == "topic")
            if command or (context.settings.agent_tools_enabled and model_key not in unsupported and not system_prompt):
                tools.update((tool.name, tool) for tool in specs)
                if not command and context.settings.mcp_config:
                    registry = context.bot_data.get("mcp_registry")
                    if registry is None:
                        registry = MCPRegistry.from_file(context.settings.mcp_config)
                        context.bot_data["mcp_registry"] = registry
                    for tool in await registry.tools_for_chat(update.effective_chat.id):
                        if tool.name in tools:
                            raise ValueError(f"Duplicate tool name: {tool.name}")
                        tools[tool.name] = tool
                if not command:
                    try:
                        schemas = [tool.schema for tool in sorted(tools.values(), key=lambda item: item.name)]
                        bound_model = context.bind_cache(model.bind_tools(schemas), "agent")
                    except Exception as error:
                        if not unsupported_tools(error):
                            raise
                        unsupported.add(model_key)
                        tools.clear()
            context.conversation_data["tool_schemas"] = (
                [tool.schema for tool in sorted(tools.values(), key=lambda item: item.name)]
                if bound_model is not None
                else []
            )
            if command:
                await context.maybe_compact()
                name = {"svg": "generate_svg", "html": "generate_html", "poll": "create_poll", "topic": "create_poll"}[
                    command
                ]
                call = {
                    "name": name,
                    "args": {"prompt": prompt or "Create a poll about the recent conversation."},
                    "id": uuid4().hex,
                    "type": "tool_call",
                }
                pending_message = AIMessage(
                    content="",
                    tool_calls=[call],
                    additional_kwargs={"origin": f"/{command}", "telegram": {"kind": "tool_call", "at": utc_now()}},
                )
                context.chat_message_history.append(pending_message)
                return {"pending": [call]}
            return {}

        async def model_node(state):
            nonlocal bound_model, placeholder, pending_message
            if state.get("iterations", 0) >= context.settings.agent_max_iterations:
                text = "Stopped after reaching the configured agent iteration limit."
                await placeholder.write(text)
                context.save_assistant_message(text, placeholder.message)
                return {"finished": True}
            if state.get("tools_blocked") or state.get("calls", 0) >= context.settings.agent_max_tool_calls:
                bound_model = None
                context.conversation_data["tool_schemas"] = []
            if bound_model is None:
                # Keep the existing streaming and network-retry behavior for
                # plain chat. Tools are projected to attributed assistant text.
                from .chat import stream_chat

                final_prompt = system_prompt
                if state.get("tools_blocked"):
                    final_prompt = context.chat_system_prompt + (
                        "\nA tool's side effect is uncertain. Do not claim success or retry. "
                        "Explain the uncertainty to the user. No more tools are available this turn."
                    )
                await stream_chat(update, context, final_prompt, save_ai_message, response_message=placeholder)
                return {"finished": True}

            await context.maybe_compact()
            # Compaction cannot make an oversized tool schema or the newest
            # atomic tool transaction smaller. Fail clearly rather than send
            # an invalid request or let the backend silently truncate it.
            budget = await context.get_context_budget()
            if (
                context._estimate_token_count(await context._current_context_messages(), include_tools=True)
                > budget * 0.9
            ):
                raise ValueError(
                    "Conversation and enabled tools exceed the model's context window. Reduce the tool allowlist or clear context."
                )
            messages = context.get_templated_messages(tool_history=True).format_messages(
                **await context.get_system_template_dict()
            )
            messages.insert(1, SystemMessage(content=AGENT_SYSTEM_PROMPT))
            aggregate = None
            complete = False
            stream_error = None
            termination_error = None

            async def stream():
                nonlocal aggregate, complete, stream_error, termination_error
                try:
                    async for chunk in bound_model.astream(messages):
                        termination_error = termination_error or _tool_termination_error(chunk)
                        aggregate = chunk if aggregate is None else aggregate + chunk
                        yield chunk
                    complete = True
                except Exception as error:
                    stream_error = error
                    if not unsupported_tools(error):
                        raise

            # A tool-only completion is not an empty-answer error. Tool
            # arguments are never passed to Telegram's text renderer.
            delivered_text = await handle_streaming_response(
                context,
                placeholder,
                stream(),
                "Agent",
                allow_empty=True,
            )
            if stream_error is not None and unsupported_tools(stream_error) and not state.get("calls"):
                unsupported.add(model_key)
                bound_model = None
                context.conversation_data["tool_schemas"] = []
                return await model_node(state)
            if not complete:
                return {"finished": True}
            response = message_chunk_to_message(aggregate) if aggregate is not None else AIMessage(content="")
            if response.tool_calls or response.invalid_tool_calls or getattr(aggregate, "tool_call_chunks", None):
                if termination_error:
                    raise ValueError(termination_error)
                response.tool_calls = _strict_tool_calls(aggregate)
                # Call IDs supplied by a model are not necessarily unique
                # across turns, or even across calls in the same batch.
                used_ids = {
                    call["id"]
                    for item in context.chat_message_history
                    if isinstance(item, AIMessage)
                    for call in item.tool_calls
                }
                for call in response.tool_calls:
                    if not call.get("id") or call["id"] in used_ids:
                        call["id"] = uuid4().hex
                    used_ids.add(call["id"])
                response.additional_kwargs["telegram"] = {"kind": "tool_call", "at": utc_now()}
                if placeholder is not None and placeholder.message is not None:
                    # This response's visible text already has its own Telegram
                    # message. Record its ID so later target checks can tell it
                    # apart from user messages, then start a fresh reply object
                    # for the next model step. Reusing the old one would edit
                    # text the user has already seen.
                    sent = message_metadata(placeholder.message)
                    if sent.get("message_id") != getattr(update.effective_message, "message_id", None):
                        response.additional_kwargs["telegram"].update(sent)
                    response.additional_kwargs["telegram"]["sender"] = {"is_bot": True}
                    placeholder = LazyReply(context, update.effective_message)
                context.chat_message_history.append(response)
                pending_message = response
                return {"pending": response.tool_calls, "iterations": state.get("iterations", 0) + 1}
            if delivered_text and save_ai_message:
                context.save_assistant_message(delivered_text, placeholder.message)
            elif (
                not state.get("delivered")
                and not delivered_text
                and not placeholder.failed
                and placeholder.message is None
            ):
                # Only when nothing reached the chat: never over delivered or
                # partly delivered text, and never for a failed send.
                await placeholder.notify("The model returned an empty response.")
            return {"finished": True}

        async def tool_node(state):
            nonlocal pending_message
            calls = state.get("calls", 0)
            delivered = state.get("delivered", False)
            blocked = state.get("tools_blocked", False)
            for call in state.get("pending", []):
                signature = json.dumps([call["name"], call["args"]], sort_keys=True, ensure_ascii=False)
                if signature in executions:
                    result = executions[signature]
                elif blocked:
                    result = {"status": "error", "error": "Further tools are blocked after an uncertain side effect."}
                elif calls >= context.settings.agent_max_tool_calls:
                    result = {"status": "error", "error": "Tool call limit reached; do not request more tools."}
                else:
                    calls += 1
                    tool = tools.get(call["name"])
                    try:
                        if tool is None:
                            raise ValueError("This tool is not permitted in this conversation.")
                        # Disable JSON Schema's legacy remote-ref retrieval.
                        validator_for(tool.parameters)(tool.parameters, registry=Registry()).validate(call["args"])
                        # A tool may have side effects; from here the turn is no
                        # longer cancelled by a newer message.
                        debounce_turns.commit_or_stop(context)
                        context.conversation_data["active_tool_call_id"] = call["id"]
                        started = context.conversation_data.setdefault("started_tool_call_ids", [])
                        if call["id"] not in started:
                            started.append(call["id"])
                        try:
                            async with asyncio.timeout(tool.timeout):
                                result = await tool.invoke(call["args"])
                        finally:
                            if context.conversation_data.get("active_tool_call_id") == call["id"]:
                                context.conversation_data.pop("active_tool_call_id", None)
                    except SchemaError as error:
                        result = {"status": "error", "error": f"Invalid tool arguments: {error.message}"}
                    except TimeoutError:
                        interrupted = context.conversation_data.pop("interrupted_delivery", None)
                        if isinstance(interrupted, dict) and interrupted.get("tool_call_id") == call["id"]:
                            result = interrupted["result"]
                        else:
                            result = {
                                "status": "execution_uncertain",
                                "error": "Tool execution timed out; a side effect may have occurred. Do not automatically retry.",
                            }
                    except Exception as error:
                        result = {"status": "error", "error": redact_for_user(str(error), context.settings)}
                    executions[signature] = result
                delivered = delivered or result.get("status") == "delivered"
                blocked = blocked or result.get("status") in {"execution_uncertain", "delivery_uncertain"}
                context.chat_message_history.append(
                    ToolMessage(
                        name=call["name"],
                        tool_call_id=call["id"],
                        content=result_content(result, context.settings),
                        status="success" if result.get("status") in {"success", "delivered"} else "error",
                        additional_kwargs={"telegram": {"kind": "tool_result", "at": utc_now()}},
                    )
                )
            pending_message = None
            if command:
                last = result
                if last.get("status") != "delivered":
                    await update.effective_message.reply_text(
                        last.get("error", "The tool could not complete the request.")
                    )
            return {"calls": calls, "delivered": delivered, "pending": [], "tools_blocked": blocked}

        async def finish_node(state):
            context.conversation_data.pop("started_tool_call_ids", None)
            context._enforce_history_cap()
            return {}

        # Ingest runs outside the cancellable graph task: a turn superseded by a
        # newer message still leaves its own message in history.
        if not (await ingest_node({}))["accepted"]:
            return

        builder = StateGraph(AgentState)
        builder.add_node("should_respond", engagement_node)
        builder.add_node("prepare", prepare_node)
        builder.add_node("agent", model_node)
        builder.add_node("tools", tool_node)
        builder.add_node("finish", finish_node)
        builder.add_edge(START, "should_respond")
        builder.add_conditional_edges("should_respond", lambda state: "prepare" if state["engage"] else END)
        builder.add_conditional_edges(
            "prepare", lambda state: "finish" if state.get("finished") else "tools" if command else "agent"
        )
        builder.add_conditional_edges("agent", lambda state: "finish" if state.get("finished") else "tools")
        builder.add_conditional_edges("tools", lambda state: "finish" if command else "agent")
        builder.add_edge("finish", END)
        graph = builder.compile()

        async def run_graph():
            async with asyncio.timeout(context.settings.agent_timeout):
                await graph.ainvoke(
                    {"conversation": context.conversation_data},
                    {"recursion_limit": context.settings.agent_max_iterations * 3 + 12},
                )

        # A separate task, so a newer relevant message can cancel the work
        # without cancelling the handler that owns this turn.
        context._turn = turn
        graph_task = asyncio.ensure_future(run_graph())
        if turn is not None:
            turn.task = graph_task
        try:
            try:
                await graph_task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if (
                    turn is not None
                    and turn.superseded
                    and graph_task.cancelled()
                    and not (current is not None and current.cancelling())
                ):
                    # Nothing was sent or executed: drop the unanswered tool
                    # request instead of recording interrupted results.
                    if pending_message is not None:
                        _discard_pending(context, pending_message)
                    logger.info("Turn for message(s) %s was superseded before replying", sorted(turn.message_ids))
                    return
                raise
        except Exception as error:
            # A timed-out/cancelled batch must still have a result per call.
            if turn is not None and turn.superseded and not turn.committed:
                # It failed while being replaced; the newer turn answers instead.
                if pending_message is not None:
                    _discard_pending(context, pending_message)
                logger.info("Superseded turn ended with an error: %s", error)
                return
            if pending_message is not None:
                _close_pending(context, pending_message)
            if isinstance(error, NetworkError) and placeholder is not None:
                raise
            text = (
                "Agent execution timed out."
                if isinstance(error, TimeoutError)
                else redact_for_user(str(error), context.settings)
            )
            if placeholder is not None:
                # Posted separately when an answer is already on screen.
                await placeholder.notify(f"Error processing chat response: {text}")
            else:
                try:
                    await update.effective_message.reply_text(text)
                except TelegramError as reply_error:
                    logger.error("Could not report a failed turn: %s", reply_error)
        except asyncio.CancelledError:
            if pending_message is not None:
                _close_pending(context, pending_message)
            raise
        finally:
            if not graph_task.done():
                graph_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await graph_task
            # Forget every delivered reply, on every exit path; only a message
            # whose final edit failed stays tracked for cleanup.
            context.finish_placeholder()
            await indicator.aclose()
            context._turn = None
            context._request_model = None
            context._request_model_name = None


class _WithMessage:
    """An update whose message was edited before its turn stored it."""

    def __init__(self, update, message):
        self._update = update
        self.effective_message = message
        self.message = message

    def __getattr__(self, name):
        return getattr(self._update, name)


def _discard_pending(context, message) -> None:
    """Remove a tool request whose tools never started; it had no effect."""
    history = context.chat_message_history
    for index, item in enumerate(history):
        if item is message:
            del history[index]
            break
    for key in ("active_tool_call_id", "started_tool_call_ids", "interrupted_delivery"):
        context.conversation_data.pop(key, None)


def _tool_termination_error(message) -> str | None:
    metadata = getattr(message, "response_metadata", None) or {}
    successful = {
        "stop",
        "tool_call",
        "tool_calls",
        "function_call",
        "end_turn",
        "stop_sequence",
        "completed",
        "eos_token",
    }
    for key in ("normalized_finish_reason", "finish_reason", "done_reason"):
        reason = metadata.get(key)
        if reason and reason not in successful:
            return "Tool call completion was interrupted or filtered; no tools were executed."
    return None


def _strict_tool_calls(message) -> list[dict]:
    """Never execute LangChain's best-effort repairs of streamed argument JSON."""
    if message.invalid_tool_calls:
        raise ValueError("The model returned invalid tool arguments; no tools were executed.")
    raw_calls = getattr(message, "tool_call_chunks", None)
    if raw_calls:
        candidates = [{"name": call["name"], "arguments": call["args"], "id": call.get("id")} for call in raw_calls]
    elif message.additional_kwargs.get("tool_calls"):
        candidates = [
            {"name": call["function"]["name"], "arguments": call["function"]["arguments"], "id": call.get("id")}
            for call in message.additional_kwargs["tool_calls"]
        ]
    else:
        # Some providers deliver structured arguments rather than JSON deltas.
        candidates = [
            {"name": call["name"], "arguments": call["args"], "id": call.get("id")} for call in message.tool_calls
        ]

    def reject_constant(value):
        raise ValueError("Non-JSON numeric constant")

    calls = []
    for call in candidates:
        try:
            arguments = call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments, parse_constant=reject_constant)
            if not isinstance(arguments, dict) or not isinstance(call["name"], str) or not call["name"]:
                raise ValueError("Tool arguments must be an object with a tool name")
        except (TypeError, ValueError):
            raise ValueError(
                "The model returned incomplete or invalid tool arguments; no tools were executed."
            ) from None
        calls.append({"name": call["name"], "args": arguments, "id": call["id"], "type": "tool_call"})
    return calls


def _close_pending(context, message):
    history = context.chat_message_history
    start = next((index for index, item in enumerate(history) if item is message), None)
    if start is None:
        for key in ("active_tool_call_id", "started_tool_call_ids", "interrupted_delivery"):
            context.conversation_data.pop(key, None)
        logger.error("Could not find the interrupted tool transaction in history")
        return
    completed = set()
    for item in history[start + 1 :]:
        if not isinstance(item, ToolMessage):
            break
        completed.add(item.tool_call_id)
    interrupted = context.conversation_data.pop("interrupted_delivery", None)
    if isinstance(interrupted, dict) and not isinstance(interrupted.get("tool_call_id"), str):
        interrupted = None
    started = set(context.conversation_data.pop("started_tool_call_ids", []))
    context.conversation_data.pop("active_tool_call_id", None)
    for call in message.tool_calls:
        if call["id"] not in completed:
            if interrupted is not None and interrupted["tool_call_id"] == call["id"]:
                result = interrupted["result"]
                interrupted = None
            elif call["id"] in started:
                result = {
                    "status": "execution_uncertain",
                    "error": "Run interrupted after this tool started; do not automatically retry.",
                }
            else:
                result = {
                    "status": "not_executed",
                    "error": "Run interrupted before this tool started; no side effect occurred. Do not automatically retry.",
                }
            context.chat_message_history.append(
                ToolMessage(
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                    content=result_content(result, context.settings),
                    additional_kwargs={"telegram": {"kind": "tool_result", "at": utc_now()}},
                )
            )
