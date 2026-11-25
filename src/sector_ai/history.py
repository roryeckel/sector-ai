"""Protocol-safe conversation history shared by commands and the graph.

Tool calls and their results are indivisible when trimming history. Text-only
models and summarizers receive an attributed projection, not orphan tool roles.
"""

import json
from functools import wraps

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .chat_metadata import is_message_edit, render_metadata, render_prompt_message
from .message_text import message_text


def _history_text(message: BaseMessage, render) -> str:
    text = message_text(render(message))
    if isinstance(message, AIMessage) and message.tool_calls:
        text += "\nTool calls: " + json.dumps(message.tool_calls, ensure_ascii=False)
    if isinstance(message, ToolMessage):
        text = f"Tool {message.name or 'unknown'} ({message.status}): {text}"
    return text


def history_text(message: BaseMessage) -> str:
    """Projection used for compaction and token accounting."""
    return _history_text(message, render_metadata)


def prompt_history_text(message: BaseMessage) -> str:
    """Projection used in prompts; assistant envelopes are never repeated."""
    return _history_text(message, render_prompt_message)


def plain_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Project tool events into assistant text for non-tool-calling requests."""
    result = []
    for message in messages:
        if isinstance(message, ToolMessage) or (isinstance(message, AIMessage) and message.tool_calls):
            result.append(AIMessage(content=prompt_history_text(message)))
        else:
            result.append(message)
    return result


def history_groups(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    groups = []
    index = 0
    while index < len(messages):
        group = [messages[index]]
        index += 1
        if isinstance(group[0], AIMessage) and group[0].tool_calls:
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                group.append(messages[index])
                index += 1
        groups.append(group)
    return groups


def safe_cut(messages: list[BaseMessage], index: int) -> int:
    """Move a cut backwards so the retained tail cannot begin with a result."""
    while 0 < index < len(messages) and isinstance(messages[index], ToolMessage):
        index -= 1
    return index


def record_content_command(function):
    """Give legacy content commands the same ingestion/serialization boundary."""

    @wraps(function)
    async def wrapped(update, context, *args, **kwargs):
        if is_message_edit(update):
            from .chat_events import handle_message_edit

            await handle_message_edit(update, context)
            return
        context.bind_update(update)
        async with context.conversation_lock:
            if not context.accept_update(update):
                return
            context.save_user_message(update.effective_message, include_command=True)
            return await function(update, context, *args, **kwargs)

    return wrapped


def scoped_control(function):
    """Serialize conversation controls without adding them to model history."""

    @wraps(function)
    async def wrapped(update, context, *args, **kwargs):
        context.bind_update(update)
        async with context.conversation_lock:
            return await function(update, context, *args, **kwargs)

    return wrapped
