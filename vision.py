import logging
import base64
from telegram import Update
from langchain_core.messages import HumanMessage
from .streaming_handler import handle_streaming_response
from langchain_core.messages import AIMessage

from .sector_context import SectorContext

logger = logging.getLogger(__name__)

async def handle_vision(update: Update, context: SectorContext) -> None:
    if not context.chat_autoreply_mode:
        return
    old_model = context.get_model()

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

    photo_files = [await photo.get_file(read_timeout=30, connect_timeout=30) for photo in [update.message.photo[-1]]]
    photo_bytes = [await photo.download_as_bytearray() for photo in photo_files]
    photo_base64s = [base64.b64encode(pb).decode('utf-8') for pb in photo_bytes]

    try:
        context.load_model(context.config_vision_model)

        messages = [
            HumanMessage(content=[
                { "type": "text", "text": vision_prompt },
                *({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{photo_base64}"
                } for photo_base64 in photo_base64s)
            ])
        ]

        stream_generator = context.bot_ollama.stream(messages)
        response = await handle_streaming_response(
            context,
            response_message,
            stream_generator,
            "Vision"
        )

        if response:
            context.chat_message_history.append(AIMessage(content=response))

    except Exception as e:
        await response_message.edit_text(f'Error processing vision prompt: {e}')
    finally:
        context.load_model(old_model)