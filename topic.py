import logging
from typing import List
from .sector_context import SectorContext
from pydantic import BaseModel, Field
from telegram import Update
from telegram.constants import PollLimit
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

logger = logging.getLogger(__name__)

class TopicPoll(BaseModel):
    title: str = Field(description="A short title about the chat topic summarization")
    topics: List[str] = Field(description=f"The {PollLimit.MIN_OPTION_NUMBER} or more relevant topics in the chat, must be less than {PollLimit.MAX_OPTION_NUMBER + 1} topics."
                              f"Preferrably 5 topics unless otherwise specified. The length of each topic must be less than {PollLimit.MAX_OPTION_LENGTH} characters.")

async def topic_poll_cmd(update: Update, context: SectorContext) -> None:
    context.bot_ollama.format = 'json'
    system_template_dict = await context.get_system_template_dict()
    parser = PydanticOutputParser(pydantic_object=TopicPoll)
    prompt_template = context.get_templated_messages(system_prompt=context.config_topic_poll_system_prompt)
    chain = prompt_template | context.bot_ollama | parser
    poll_response = chain.invoke({'format_instructions': parser.get_format_instructions(), **system_template_dict})
    logger.info(f'Topic Poll Response: {poll_response}')
    context.bot_ollama.format = ''

    if not (PollLimit.MIN_OPTION_NUMBER <= len(poll_response.topics) <= PollLimit.MAX_OPTION_NUMBER):
        logger.error(f'Poll Error: Invalid number of topics: {len(poll_response.topics)}')
        raise OutputParserException(f"Poll must have between {PollLimit.MIN_OPTION_NUMBER} and {PollLimit.MAX_OPTION_NUMBER} topics.")
    
    for topic in poll_response.topics:
        if len(topic) > PollLimit.MAX_OPTION_LENGTH:
            logger.error(f'Poll Error: Topic length exceeds limit: {topic}')
            raise OutputParserException(f"Each topic must be less than {PollLimit.MAX_OPTION_LENGTH} characters.")
    
    await update.message.reply_poll(poll_response.title or "Topics", poll_response.topics, is_anonymous=False)