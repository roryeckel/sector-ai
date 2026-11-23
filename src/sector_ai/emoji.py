import logging

from langchain_core.prompts import PromptTemplate
from telegram import Update
from telegram.constants import MessageLimit

from .chat_metadata import instruction_messages, strip_internal_envelope, with_context
from .history import record_content_command
from .message_text import message_text
from .sector_context import SectorContext
from .usage import record_usage

logger = logging.getLogger(__name__)


@record_content_command
async def emoji_cmd(update: Update, context: SectorContext) -> None:
    prompt = update.message.text[6:]
    if not prompt:
        return await update.message.reply_text("Please enter some text to turn into emojis")
    prompt = prompt.strip()
    logger.info(f"Emoji Prompt: {prompt}")
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.emoji, input_variables=["input", *system_template_dict.keys()]
    )
    messages = with_context(
        instruction_messages(
            template,
            {"input": prompt, **system_template_dict},
            input_key="input",
            input_text=prompt,
        ),
        context,
    )
    message = await context.bind_cache(context.bot_llm, "emoji").ainvoke(messages)
    record_usage(context, "emoji", message)
    emoji_response = strip_internal_envelope(message_text(message))
    if len(emoji_response) > MessageLimit.MAX_TEXT_LENGTH:
        emoji_response = emoji_response[: MessageLimit.MAX_TEXT_LENGTH]
    logger.info(f"Response: {emoji_response}")
    if emoji_response.strip():
        sent = await update.message.reply_text(emoji_response)
        context.save_assistant_message(emoji_response, sent)
    else:
        await update.message.reply_text("The model returned an empty response.")
