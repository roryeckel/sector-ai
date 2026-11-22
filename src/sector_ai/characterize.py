import logging

from langchain_core.prompts import PromptTemplate
from telegram import Update

from .message_text import message_refused, message_text
from .sector_context import SectorContext

logger = logging.getLogger(__name__)


async def characterize_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.characterizer, input_variables=["input", *system_template_dict.keys()]
    )
    chain = template | context.bot_llm
    characterization_prompt = update.message.text.split(maxsplit=1)
    if len(characterization_prompt) <= 1:
        await update.message.reply_text("No prompt provided.")
        return
    characterization_prompt = characterization_prompt[-1]
    logger.info(f"Characterization Prompt: {characterization_prompt}")
    message = await chain.ainvoke({"input": characterization_prompt, **system_template_dict})
    if message_refused(message):
        await update.message.reply_text(message_text(message) or "The model refused the request.")
        return
    characterization = message_text(message)
    if not characterization.strip():
        await update.message.reply_text("The model returned an empty response; the system prompt was not changed.")
        return
    await update.message.reply_text(f"Update the system prompt to generated characterization:\n{characterization}")
    logger.info(f"Characterization: {characterization}")
    context.clear_history()
    context.chat_system_prompt = characterization
