import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from telegram import Message, MessageEntity
from telegram.error import BadRequest, NetworkError

from .logging_utils import redact_for_user
from .message_text import message_text
from .retry import retry_on_network_error
from .sector_context import SectorContext
from .telegram_format import render_chunks, render_preview, telegram_entities

logger = logging.getLogger(__name__)


def _payload_key(text: str, entities: list[MessageEntity] | None) -> tuple:
    """What Telegram would show. A formatting-only change still needs an edit."""
    return (
        text,
        tuple(
            (entity.type, entity.offset, entity.length, entity.url, entity.language, entity.custom_emoji_id)
            for entity in entities or ()
        ),
    )


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
    paused = False
    entities_ok = True
    markdown_enabled = context.settings.markdown
    cursor = context.settings.streaming_cursor or ""
    buffer_lock = asyncio.Lock()

    async def edit(text: str, entities: list[MessageEntity] | None) -> bool:
        nonlocal last_sent, last_update, entities_ok
        key = _payload_key(text, entities)
        if not text.strip() or key == last_sent:
            return True
        try:
            delivered = await retry_on_network_error(
                lambda: response_message.edit_text(text, entities=entities or None),
                what="Telegram edit",
                delays=(2.0, 5.0),
            )
        except BadRequest as error:
            if "not modified" in str(error).lower():
                last_sent = key
                return True
            if entities:
                # The server refused the formatting, not the text. Deliver the
                # same text plainly and stop attaching entities to this reply.
                entities_ok = False
                logger.warning("Telegram rejected formatted text (%s); falling back to plain", error)
                return await edit(text, None)
            logger.error("Error updating message: %s", error)
            return False
        except Exception as error:
            logger.error("Error updating message: %s", error)
            return False
        if delivered is None:
            return False
        last_sent = key
        last_update = asyncio.get_running_loop().time()
        return True

    async def update_message() -> bool:
        nonlocal buffer, response
        async with buffer_lock:
            if buffer:
                response += buffer
            buffer = ""
        text, entities = render_preview(response, enabled=markdown_enabled, cursor=cursor)
        return await edit(text, telegram_entities(entities) if entities_ok else None)

    async def deliver_final() -> bool:
        """Edit the placeholder with the first message, then send the rest."""
        nonlocal buffer, response
        async with buffer_lock:
            if buffer:
                response += buffer
            buffer = ""

        chunks = render_chunks(response, enabled=markdown_enabled)
        if not chunks:
            return True

        text, entities = chunks[0]
        if not await edit(text, telegram_entities(entities) if entities_ok else None):
            return False

        for text, entities in chunks[1:]:
            payload = telegram_entities(entities) if entities_ok else None
            sent = await retry_on_network_error(
                lambda text=text, payload=payload: context.bot.send_message(
                    chat_id=response_message.chat_id, text=text, entities=payload
                ),
                what="Telegram message",
                delays=(2.0, 5.0),
            )
            if sent is None:
                logger.error("%s could not deliver every message part", log_prefix)
                return False
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
            if not paused and (
                current_time - last_update >= context.settings.streaming_interval_sec
                or len(buffer) >= context.settings.streaming_chunk_size
            ):
                if update_task and not await update_task:
                    # Keep consuming the model stream so the final flush can
                    # still deliver the whole answer.
                    paused = True
                if not paused:
                    update_task = asyncio.create_task(update_message())

        if update_task:
            await update_task

        delivered = await deliver_final()

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
