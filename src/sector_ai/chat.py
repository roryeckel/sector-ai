import asyncio
import logging

from telegram import Update
from telegram.error import NetworkError

from .logging_utils import redact_for_user
from .sector_context import SectorContext
from .streaming_handler import LazyReply, handle_streaming_response
from .typing_indicator import typing

logger = logging.getLogger(__name__)


async def stream_chat(
    update: Update, context: SectorContext, system_prompt: str = None, save_ai_message=True, *, response_message=None
) -> None:
    """Stream a plain reply into ``response_message``, a ``LazyReply`` the caller owns.

    Without one, the reply is posted when the first text arrives and the chat
    shows "typing" until then (and while it streams).
    """
    if response_message is None:
        await context.remove_pending_placeholder()
        async with typing(update.effective_message):
            await stream_chat(
                update,
                context,
                system_prompt,
                save_ai_message,
                response_message=LazyReply(context, update.effective_message),
            )
        return

    response = ""
    try:
        for attempt in range(3):
            try:
                # This path is also used after tools exhaust their budget or
                # have an uncertain side effect. Their new results must be
                # compacted and checked before sending another model request.
                await context.maybe_compact()
                messages = context.get_templated_messages(system_prompt=system_prompt).format_messages(
                    **await context.get_system_template_dict()
                )
                budget = await context.get_context_budget()
                if context._estimate_token_count(messages) > budget * 0.9:
                    raise ValueError("Conversation exceeds the model's context window. Compact or clear context.")
                logger.info("Respond Prompt: %s", messages)
                stream_generator = context.bind_cache(context.bot_llm, "chat").astream(messages)
                response = await handle_streaming_response(context, response_message, stream_generator, "Chat")
                break
            except NetworkError as e:
                response = ""
                if attempt == 2:
                    raise
                logger.warning("Telegram network error while chatting (%s); retrying", e)
                await asyncio.sleep(5.0 * (attempt + 1))

        if response.startswith("AI:"):
            logger.warning("Removing AI Prefix from response")
            response = response[3:].strip()

        if save_ai_message and response:
            context.save_assistant_message(response, response_message.message)
        context.finish_placeholder()

    except NetworkError:
        raise
    except Exception as e:
        await response_message.write(f"Error processing chat response: {redact_for_user(str(e), context.settings)}")
        context.finish_placeholder()


async def chat_cmd(update: Update, context: SectorContext) -> None:
    if len((update.effective_message.text or "").split(maxsplit=1)) <= 1:
        return await update.message.reply_text("Please enter your message to me.")
    from .agent import run_agent

    await run_agent(update, context, force=True)


async def handle_chat(update: Update, context: SectorContext, system_prompt: str = None, save_ai_message=True) -> None:
    from .agent import run_agent

    await run_agent(
        update, context, force=True, ingest=False, system_prompt=system_prompt, save_ai_message=save_ai_message
    )
