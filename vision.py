import asyncio
import logging
import base64
from telegram import Update
from langchain_core.messages import HumanMessage

from .sector_context import SectorContext

logger = logging.getLogger(__name__)

async def handle_vision(update: Update, context: SectorContext) -> None:
    if not context.chat_autoreply_mode:
        return
    old_model = context.bot_ollama.model

    vision_prompt = update.message.caption or context.config_vision_system_prompt

    logger.info(f'Vision Prompt: {vision_prompt}')

    total_size = 0
    for photo in update.message.photo:
        total_size += photo.file_size
        if total_size > 20 * 1024 * 1024:  # 20 megabytes
            await update.message.reply_text('The combined size of the photos exceeds the 20MB limit.')
            return
        if photo.file_size > 10 * 1024 * 1024:  # 10 megabytes
            await update.message.reply_text('One of the photos exceeds the 10MB size limit.')
            return
        
    response_message = await update.message.reply_text("Processing...")

    photo_files = [await photo.get_file(read_timeout=30, connect_timeout=30) for photo in update.message.photo]
    photo_bytes = [await photo.download_as_bytearray() for photo in photo_files]
    photo_base64s = [base64.b64encode(pb).decode('utf-8') for pb in photo_bytes]

    try:
        context.load_model(context.config_vision_model)

        messages = []
        messages.append(HumanMessage(content=vision_prompt))

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

        for chunk in context.bot_ollama.stream(messages, images=photo_base64s):
            if not chunk:
                break

            buffer += chunk

            current_time = asyncio.get_event_loop().time()
            if current_time - last_update >= context.config_streaming_interval_sec or len(buffer) >= context.config_streaming_chunk_size:
                if update_task:
                    await update_task
                update_task = asyncio.create_task(update_message(context.config_streaming_cursor or ''))

        if update_task:
            await update_task

        await update_message()

        logger.info(f'Vision Result: {response}')
    except TimeoutError as e:
        if not response:
            await response_message.edit_text(f'Error processing vision prompt: {e}')
    except Exception as e:
        await response_message.edit_text(f'Error processing vision prompt: {e}')
    finally:
        context.load_model(old_model)
        if update_task and not update_task.done():
            await update_task