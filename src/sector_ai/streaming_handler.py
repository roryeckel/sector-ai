import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from telegram import Message
from telegram.error import BadRequest, NetworkError

from .logging_utils import redact_for_user
from .message_text import message_text
from .retry import retry_on_network_error
from .sector_context import SectorContext

logger = logging.getLogger(__name__)


async def handle_streaming_response(
    context: SectorContext, response_message: Message, stream_generator: AsyncIterator, log_prefix: str
) -> str:
    response = ""
    buffer = ""
    last_sent = None
    last_update = asyncio.get_running_loop().time()
    update_task = None
    saw_chunk = False
    ended_with_falsy = False
    finish_reason = None
    buffer_lock = asyncio.Lock()

    async def update_message(message_postfix: str = "") -> bool:
        nonlocal buffer, last_update, response, last_sent
        async with buffer_lock:
            if buffer:
                response += buffer
            buffer = ""
        try:
            new_text = response + message_postfix
            if new_text.strip() and new_text != last_sent:
                delivered = await retry_on_network_error(
                    lambda: response_message.edit_text(new_text),
                    what="Telegram edit",
                    delays=(2.0, 5.0),
                )
                if delivered is None:
                    return False
                last_sent = new_text
        except BadRequest as e:
            if "not modified" in str(e).lower():
                last_sent = new_text
            else:
                logger.error(f"Error updating message: {e}")
                return False
        except Exception as e:
            logger.error(f"Error updating message: {e}")
            return False
        last_update = asyncio.get_running_loop().time()
        return True

    try:
        async for chunk in stream_generator:
            if not chunk:
                ended_with_falsy = True
                break

            saw_chunk = True
            metadata = getattr(chunk, "response_metadata", None) or {}
            finish_reason = metadata.get("normalized_finish_reason") or metadata.get("finish_reason") or finish_reason
            async with buffer_lock:
                buffer += message_text(chunk)

            current_time = asyncio.get_running_loop().time()
            if (
                current_time - last_update >= context.settings.streaming_interval_sec
                or len(buffer) >= context.settings.streaming_chunk_size
            ):
                if update_task:
                    if not await update_task:
                        break
                update_task = asyncio.create_task(update_message(context.settings.streaming_cursor or ""))

        if update_task:
            await update_task

        delivered = await update_message()

        if not response.strip() and (saw_chunk or not ended_with_falsy):
            if finish_reason == "refusal":
                await response_message.edit_text("The model refused the request.")
            else:
                await response_message.edit_text("The model returned an empty response.")
            logger.warning("%s returned no text", log_prefix)

        if not delivered and response.strip():
            logger.error("%s could not be delivered; not saving it", log_prefix)
            return ""

        logger.info(f"{log_prefix} Result: {response}")
        return response

    except NetworkError as e:
        # A delivery failure must not turn into another model call.
        logger.error("%s delivery failed: %s", log_prefix, e)
        return ""
    except Exception as e:
        error_message = f"Error processing {log_prefix.lower()}: {redact_for_user(str(e), context.settings)}"
        await response_message.edit_text(error_message)
        logger.error(error_message)
        return ""

    finally:
        # A pending task here means the generator failed mid-stream; awaiting it
        # would paint stale text over the error message, so cancel it instead.
        if update_task and not update_task.done():
            update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await update_task
