import logging

from langchain_core.prompts import PromptTemplate
from telegram import Update

from .sector_context import SectorContext

logger = logging.getLogger(__name__)

async def characterize_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.config_characterizer_system_prompt,
        input_variables=['input', *system_template_dict.keys()])
    chain = template | context.bot_ollama
    characterization_prompt = update.message.text.split(maxsplit=1)
    if len(characterization_prompt) <= 1:
        await update.message.reply_text('No prompt provided.')
        return
    characterization_prompt = characterization_prompt[-1]
    logger.info(f'Characterization Prompt: {characterization_prompt}')
    characterization = chain.invoke({'input': characterization_prompt, **system_template_dict}).content
    await update.message.reply_text(f'Update the system prompt to generated characterization:\n{characterization}')
    logger.info(f'Characterization: {characterization}')
    context.chat_message_history.clear()
    context.chat_system_prompt = characterization
