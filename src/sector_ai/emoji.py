"""`/emoji`: turn a prompt, or the recent conversation, into emojis."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from telegram import Update
from telegram.constants import MessageLimit

from .chat_metadata import (
    instruction_messages,
    metadata_pair,
    render_prompt_message,
    strip_internal_envelope,
    with_context,
)
from .errors import UserFacingError
from .history import plain_history, record_content_command
from .message_text import message_text
from .sector_context import SectorContext
from .telegram_format import clip_utf16
from .typing_indicator import typing
from .usage import record_usage

logger = logging.getLogger(__name__)

# With no prompt, the retained conversation above the request is the input.
CONVERSATION_INPUT = "the recent conversation above"


def conversation_history(context, current_message) -> list:
    """Rendered history without the ``/emoji`` command itself."""
    history = [render_prompt_message(message) for message in plain_history(context.chat_message_history)]
    if not history:
        return history
    tail = history[-1]
    envelope = tail.additional_kwargs.get("telegram") or {}
    identifier = getattr(current_message, "message_id", None)
    if (type(identifier) is int and envelope.get("message_id") == identifier) or (
        identifier is None and isinstance(tail.content, str) and tail.content.rstrip().endswith("/emoji")
    ):
        return history[:-1]
    return history


async def conversation_messages(context, template, variables, current_message):
    """Instructions, summary and bounded history, then the task.

    Returns ``None`` when nothing was said before the command.
    """
    history = conversation_history(context, current_message)
    if not history:
        return None
    task_messages = instruction_messages(
        template,
        {**variables, "input": CONVERSATION_INPUT},
        input_key="input",
        input_text=CONVERSATION_INPUT,
    )
    if not any(isinstance(message, HumanMessage) for message in task_messages):
        # A custom template may not interpolate the input placeholder.
        task_messages.append(HumanMessage(content=f"Convert {CONVERSATION_INPUT} into emojis."))
    prefix = []
    remainder = task_messages
    if task_messages and isinstance(task_messages[0], SystemMessage):
        prefix.append(task_messages[0])
        remainder = task_messages[1:]
    static, dynamic = metadata_pair(context)
    if static is not None:
        prefix.append(static)
    if context.context_summary:
        prefix.append(SystemMessage(content=f"Summary of the earlier conversation:\n{context.context_summary}"))
    shared = [render_prompt_message(message) for message in context.shared_chat_events()]
    suffix = list(remainder)
    if dynamic is not None:
        suffix.insert(0, dynamic)
    budget = await context.get_context_budget()
    # Drop the oldest conversation first; the instructions and task stay.
    while len(history) > 1 and context._estimate_token_count([*prefix, *history, *shared, *suffix]) > budget * 0.65:
        history.pop(0)
    if context._estimate_token_count([*prefix, *history, *shared, *suffix]) > budget * 0.8:
        raise UserFacingError("The recent conversation is too large to turn into emojis.")
    return [*prefix, *history, *shared, *suffix]


@record_content_command
async def emoji_cmd(update: Update, context: SectorContext) -> None:
    parts = (update.message.text or "").split(maxsplit=1)
    prompt = parts[1].strip() if len(parts) > 1 else ""
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.emoji, input_variables=["input", *system_template_dict.keys()]
    )
    if prompt:
        messages = with_context(
            instruction_messages(
                template,
                {"input": prompt, **system_template_dict},
                input_key="input",
                input_text=prompt,
            ),
            context,
        )
    else:
        try:
            messages = await conversation_messages(context, template, system_template_dict, update.effective_message)
        except ValueError as error:
            await update.message.reply_text(str(error))
            return
        if messages is None:
            await update.message.reply_text("Please enter some text to turn into emojis")
            return
    logger.info("Emoji Prompt: %s", prompt or "[recent conversation]")
    async with typing(update.message):
        await _reply_with_emojis(update, context, messages)


async def _reply_with_emojis(update: Update, context: SectorContext, messages) -> None:
    message = await context.bind_cache(context.bot_llm, "emoji").ainvoke(messages)
    record_usage(context, "emoji", message)
    # Telegram counts UTF-16 units, so an emoji usually counts twice.
    emoji_response = clip_utf16(strip_internal_envelope(message_text(message)), MessageLimit.MAX_TEXT_LENGTH)
    logger.info(f"Response: {emoji_response}")
    if emoji_response.strip():
        sent = await update.message.reply_text(emoji_response)
        context.save_assistant_message(emoji_response, sent)
    else:
        await update.message.reply_text("The model returned an empty response.")
