import logging
from .sector_context import SectorContext
from telegram import Update
from .streaming_handler import handle_streaming_response
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

async def handle_chat(update: Update, context: SectorContext, system_prompt: str = None, save_ai_message = True) -> None:
    prompt_template = context.get_templated_messages(system_prompt=system_prompt)
    logger.info(f'Respond Prompt: {prompt_template}')
    chain = prompt_template | context.bot_ollama

    response_message = await update.message.reply_text("Processing...")

    try:
        stream_generator = chain.stream(await context.get_system_template_dict())
        response = await handle_streaming_response(
            context,
            response_message,
            stream_generator,
            "Chat"
        )

        if response.startswith('AI:'):
            logger.warning(f'Removing AI Prefix from response')
            response = response[3:].strip()

        if save_ai_message and response:
            context.chat_message_history.append(AIMessage(content=response))

    except Exception as e:
        await response_message.edit_text(f'Error processing chat response: {e}')

async def chat_cmd(update: Update, context: SectorContext) -> None:
    if context.save_user_message(update.message):
        await handle_chat(update, context)
    else:
        return await update.message.reply_text('Please enter your message to me.')