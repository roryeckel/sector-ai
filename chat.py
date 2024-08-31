import asyncio
import logging
from .sector_context import SectorContext
from telegram import Update
from telegram.constants import MessageLimit
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

async def respond(update: Update, context: SectorContext, system_prompt: str = None, save_ai_message = True) -> None:
    prompt_template = context.get_templated_messages(system_prompt=system_prompt)
    logger.info(f'Respond Prompt: {prompt_template}')
    chain = prompt_template | context.bot_ollama

    response_message = await update.message.reply_text("Processing...")

    try:
        response = ''
        buffer = ''
        last_update = asyncio.get_event_loop().time()
        update_task = None

        async def update_message(message_postfix: str = '') -> None:
            nonlocal buffer, last_update, response
            if buffer:
                response += buffer
            await response_message.edit_text(response + message_postfix)
            buffer = ''
            last_update = asyncio.get_event_loop().time()
        

        for chunk in chain.stream(await context.get_system_template_dict()):
            if not chunk:
                break

            buffer += chunk

            current_time = asyncio.get_event_loop().time()
            if current_time - last_update >= 0.7 or len(buffer) >= 200:
                if update_task:
                    await update_task
                update_task = asyncio.create_task(update_message(context.config_cursor or ''))

        if update_task:
            await update_task

        if response.startswith('AI:'):
            logger.warning(f'Removing AI Prefix from response')
            response = response[3:].strip()

        await update_message()

        logger.info(f'Respond Result: {response}')
        if save_ai_message:
            context.chat_message_history.append(AIMessage(content=response))
    except Exception as e:
        await response_message.edit_text(f'Error processing chat response: {e}')
    finally:
        if update_task and not update_task.done():
            await update_task

async def chat_cmd(update: Update, context: SectorContext) -> None:
    if context.save_user_message(update.message):
        await respond(update, context)
    else:
        return await update.message.reply_text('Please enter your message to me.')