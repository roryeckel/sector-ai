import logging
from typing import List
from .sector_context import SectorContext
from pydantic import BaseModel, Field
from telegram import Update
from telegram.constants import PollLimit
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class BasicPoll(BaseModel):
    question: str = Field(description="A question related to the chat")
    options: List[str] = Field(description=f"The {PollLimit.MIN_OPTION_NUMBER} or more options to choose from, must be less than {PollLimit.MAX_OPTION_NUMBER + 1} options."
                               f"Preferrably 5 options unless otherwise specified. The length of each option must be less than {PollLimit.MAX_OPTION_LENGTH} characters.")

async def poll_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    parser = PydanticOutputParser(pydantic_object=BasicPoll)
    template = PromptTemplate(
        template=context.config_basic_poll_system_prompt,
        input_variables=['query', *system_template_dict.keys()],
        partial_variables={'format_instructions': parser.get_format_instructions()})
    chain = template | context.bot_ollama | parser
    poll_prompt = update.message.text.split(maxsplit=1)
    if len(poll_prompt) <= 1:
        await update.message.reply_text('No prompt provided.')
        return
    poll_prompt = poll_prompt[-1]
    logger.info(f'Poll Prompt: {poll_prompt}')
    poll_response = chain.invoke({'query': poll_prompt, **system_template_dict})
    logger.info(f'Poll Response: {poll_response}')
    
    if not (PollLimit.MIN_OPTION_NUMBER <= len(poll_response.options) <= PollLimit.MAX_OPTION_NUMBER):
        logger.error(f'Poll Error: Invalid number of options: {len(poll_response.options)}')
        raise OutputParserException(f"Poll must have between {PollLimit.MIN_OPTION_NUMBER} and {PollLimit.MAX_OPTION_NUMBER} options.")
    
    for option in poll_response.options:
        if len(option) > PollLimit.MAX_OPTION_LENGTH:
            logger.error(f'Poll Error: Option length exceeds limit: {option}')
            raise OutputParserException(f"Each option must be less than {PollLimit.MAX_OPTION_LENGTH} characters.")
        
    await update.message.reply_poll(poll_response.question, poll_response.options, is_anonymous=False)