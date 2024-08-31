import logging
from .sector_context import SectorContext
from pydantic import BaseModel, Field
from telegram import Update
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class Decision(BaseModel):
    result: bool = Field(description="True if the decision is in agreeance or opinion is positive/good, otherwise False")

async def make_decision(prompt: str, context: SectorContext, partial_variables=None) -> bool | str:
    partial_variables = partial_variables or {}
    system_template_dict = await context.get_system_template_dict()
    parser = PydanticOutputParser(pydantic_object=Decision)
    template = PromptTemplate(
        template=context.config_decide_system_prompt,
        input_variables=['input', *system_template_dict.keys(), *partial_variables.keys()],
        partial_variables={'format_instructions': parser.get_format_instructions(), **partial_variables})
    chain = template | context.bot_ollama | parser
    logger.info(f'Decide Prompt: {prompt}')
    try:
        decision = chain.invoke({'input': prompt, **system_template_dict})
        logger.info(f'Decision: {decision.result}')
        return decision.result
    except OutputParserException as e:
        logger.exception(f'Decision Error: {e.llm_output}', exc_info=e)
        return e.llm_output

async def decide_cmd(update: Update, context: SectorContext) -> None:
    decide_prompt = update.message.text.split(maxsplit=1)
    if len(decide_prompt) <= 1:
        await update.message.reply_text('No prompt provided.')
        return
    decide_prompt = decide_prompt[-1]
    decision = await make_decision(decide_prompt, context)
    await update.message.reply_text(str(decision))
