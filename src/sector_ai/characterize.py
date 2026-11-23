import logging

from langchain_core.prompts import PromptTemplate
from telegram import Update

from .admin import admin_only
from .chat_metadata import instruction_messages, strip_internal_envelope, with_context
from .history import scoped_control
from .message_text import message_refused, message_text
from .sector_context import SectorContext
from .typing_indicator import typing
from .usage import record_usage

logger = logging.getLogger(__name__)


@admin_only
@scoped_control
async def characterize_cmd(update: Update, context: SectorContext) -> None:
    """Replace this conversation's system prompt. Admin-only: it rewrites the bot's behaviour."""
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.characterizer, input_variables=["input", *system_template_dict.keys()]
    )
    characterization_prompt = update.message.text.split(maxsplit=1)
    if len(characterization_prompt) <= 1:
        await update.message.reply_text("No prompt provided.")
        return
    characterization_prompt = characterization_prompt[-1]
    logger.info(f"Characterization Prompt: {characterization_prompt}")
    messages = with_context(
        instruction_messages(
            template,
            {"input": characterization_prompt, **system_template_dict},
            input_key="input",
            input_text=characterization_prompt,
        ),
        context,
    )
    async with typing(update.message):
        message = await context.bind_cache(context.bot_llm, "characterize").ainvoke(messages)
        record_usage(context, "characterize", message)
        if message_refused(message):
            await update.message.reply_text(message_text(message) or "The model refused the request.")
            return
        characterization = strip_internal_envelope(message_text(message))
        if not characterization.strip():
            await update.message.reply_text("The model returned an empty response; the system prompt was not changed.")
            return
        if "SECTOR_AI_INPUT" in characterization:
            # It echoed the internal placeholder instead of writing a prompt.
            await update.message.reply_text(
                "The model echoed an internal placeholder; the system prompt was not changed. Try again."
            )
            return
        await update.message.reply_text(f"Update the system prompt to generated characterization:\n{characterization}")
    logger.info(f"Characterization: {characterization}")
    context.clear_history()
    context.chat_system_prompt = characterization
