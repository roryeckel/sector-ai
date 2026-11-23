import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from telegram import Update

from .chat_metadata import instruction_messages, strip_internal_envelope, with_context
from .history import record_content_command
from .message_text import message_text
from .sector_context import SectorContext
from .typing_indicator import typing
from .usage import record_usage

logger = logging.getLogger(__name__)


class Decision(BaseModel):
    result: bool = Field(
        description="True if the decision is in agreeance or opinion is positive/good, otherwise False"
    )


class EngagementDecision(BaseModel):
    result: bool = Field(
        description="True if the bot should respond to the final incoming message; False to stay silent."
    )


async def make_decision(prompt: str, context: SectorContext, partial_variables=None, *, update=None) -> bool | str:
    partial_variables = partial_variables or {}
    system_template_dict = await context.get_system_template_dict()
    parser = PydanticOutputParser(pydantic_object=EngagementDecision if update is not None else Decision)
    template = PromptTemplate(
        template=(prompt + "\n{format_instructions}") if partial_variables else context.settings.system_prompts.decide,
        input_variables=["input", *system_template_dict.keys(), *partial_variables.keys()],
        partial_variables={"format_instructions": parser.get_format_instructions(), **partial_variables},
    )
    logger.info(f"Decide Prompt: {prompt}")
    try:
        values = {"input": prompt, **system_template_dict, **partial_variables}
        if update is not None:
            from .engagement import engagement_messages

            messages = await engagement_messages(update, context, template.format(**values))
            response = await context.bind_cache(context.bot_llm, "engage").ainvoke(messages)
            record_usage(context, "engage", response)
            response = response.model_copy(update={"content": strip_internal_envelope(message_text(response))})
            decision = parser.invoke(response)
        else:
            messages = with_context(
                instruction_messages(template, values, input_key="input", input_text=prompt),
                context,
            )
            response = await context.bind_cache(context.bot_llm, "decide").ainvoke(messages)
            record_usage(context, "decide", response)
            response = response.model_copy(update={"content": strip_internal_envelope(message_text(response))})
            decision = parser.invoke(response)
        logger.info(f"Decision: {decision.result}")
        return decision.result
    except OutputParserException as e:
        logger.exception(f"Decision Error: {e.llm_output}", exc_info=e)
        return e.llm_output


@record_content_command
async def decide_cmd(update: Update, context: SectorContext) -> None:
    decide_prompt = update.message.text.split(maxsplit=1)
    if len(decide_prompt) <= 1:
        await update.message.reply_text("No prompt provided.")
        return
    decide_prompt = decide_prompt[-1]
    async with typing(update.message):
        decision = await make_decision(decide_prompt, context)
        sent = await update.message.reply_text(str(decision))
    context.save_assistant_message(str(decision), sent)
