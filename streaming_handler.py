import asyncio
import logging
from telegram import Message
from .sector_context import SectorContext

logger = logging.getLogger(__name__)

async def handle_streaming_response(
    context: SectorContext,
    response_message: Message,
    stream_generator,
    log_prefix: str
) -> str:
    response = ''
    buffer = ''
    last_update = asyncio.get_running_loop().time()
    update_task = None
    buffer_lock = asyncio.Lock()

    async def update_message(message_postfix: str = '') -> None:
        nonlocal buffer, last_update, response
        async with buffer_lock:
            if buffer:
                response += buffer
            buffer = ''
        try:
            await response_message.edit_text((response + message_postfix) or " ")
        except Exception as e:
            logger.error(f"Error updating message: {e}")
        last_update = asyncio.get_running_loop().time()

    try:
        for chunk in stream_generator:
            if not chunk:
                break

            async with buffer_lock:
                buffer += chunk.content

            current_time = asyncio.get_running_loop().time()
            if current_time - last_update >= context.config_streaming_interval_sec or len(buffer) >= context.config_streaming_chunk_size:
                if update_task:
                    await update_task
                update_task = asyncio.create_task(update_message(context.config_streaming_cursor or ''))

        if update_task:
            await update_task

        await update_message()

        logger.info(f'{log_prefix} Result: {response}')
        return response

    except Exception as e:
        error_message = f'Error processing {log_prefix.lower()}: {e}'
        await response_message.edit_text(error_message)
        logger.error(error_message)
        return ''

    finally:
        if update_task and not update_task.done():
            await update_task