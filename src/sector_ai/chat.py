import asyncio
import logging

from langchain_core.messages import AIMessage
from telegram import Update
from telegram.error import NetworkError

from .logging_utils import redact_for_user
from .retry import retry_on_network_error
from .sector_context import SectorContext
from .streaming_handler import handle_streaming_response

logger = logging.getLogger(__name__)


async def handle_chat(update: Update, context: SectorContext, system_prompt: str = None, save_ai_message=True) -> None:
    response = ""
    response_message = None
    try:
        # Post this before compaction: folding a long transcript is another
        # model call, and the chat should not look silent while it runs.
        response_message = await retry_on_network_error(
            lambda: update.message.reply_text("Processing..."), what="Processing placeholder"
        )
        if response_message is None:
            logger.error("Could not post the placeholder; dropping this reply")
            return
        await context.maybe_compact()

        prompt_template = context.get_templated_messages(system_prompt=system_prompt)
        logger.info(f"Respond Prompt: {prompt_template}")
        chain = prompt_template | context.bot_llm

        for attempt in range(3):
            try:
                stream_generator = chain.astream(await context.get_system_template_dict())
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
            context.chat_message_history.append(AIMessage(content=response))

    except NetworkError:
        raise
    except Exception as e:
        if response_message is None:
            raise
        await response_message.edit_text(f"Error processing chat response: {redact_for_user(str(e), context.settings)}")


async def chat_cmd(update: Update, context: SectorContext) -> None:
    if context.save_user_message(update.message):
        await handle_chat(update, context)
    else:
        return await update.message.reply_text("Please enter your message to me.")
