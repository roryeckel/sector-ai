import logging

from langchain_core.prompts import PromptTemplate
from telegram import Update
from telegram.constants import MessageLimit

from .message_text import message_text
from .sector_context import SectorContext

logger = logging.getLogger(__name__)


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
    chain = template | context.bot_llm
    emoji_response = message_text(await chain.ainvoke({"input": prompt, **system_template_dict}))
    if len(emoji_response) > MessageLimit.MAX_TEXT_LENGTH:
        emoji_response = emoji_response[: MessageLimit.MAX_TEXT_LENGTH]
    logger.info(f"Response: {emoji_response}")
    if emoji_response.strip():
        await update.message.reply_text(emoji_response)
    else:
        await update.message.reply_text("The model returned an empty response.")
