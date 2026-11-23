import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from telegram import Message, MessageEntity
from telegram.error import BadRequest, NetworkError

from .chat_metadata import strip_internal_envelope
from .debounce import commit_or_stop
from .logging_utils import redact_for_user
from .message_text import message_text
from .retry import retry_on_network_error
from .sector_context import SectorContext
from .telegram_format import render_chunks, render_preview, telegram_entities
from .usage import record_usage, usage_details

logger = logging.getLogger(__name__)


class LazyReply:
    """The bot's reply to ``source``, posted only once there is text to show.

    The first write sends a reply; later writes edit it. A turn that never
    produces visible text, such as one that only sends a poll, leaves nothing
    behind in the chat.
    """

    def __init__(self, context: SectorContext, source=None, *, message=None):
        self.context = context
        self.source = source
        self.message = message

    @classmethod
    def wrap(cls, context: SectorContext, target) -> "LazyReply":
        """Accept a ``LazyReply`` or an already posted message."""
        return target if isinstance(target, LazyReply) else cls(context, message=target)

    @property
    def chat_id(self):
        return getattr(self.message or self.source, "chat_id", None)

    @property
    def message_thread_id(self):
        return getattr(self.message or self.source, "message_thread_id", None)

    async def write(self, text: str, entities: list[MessageEntity] | None = None):
        # Claimed before the request goes out: a newer message can never cancel a
        # turn while its first send is in flight, and a write queued by a turn
        # that was just superseded never goes out.
        commit_or_stop(self.context)
        if self.message is not None:
            return await self.message.edit_text(text, entities=entities)
        sent = await self.source.reply_text(text, entities=entities)
        if sent is not None:
            self.message = sent
            self.context.remember_sent(sent)
            # Tracked so a cursor left behind by a failed edit can be removed.
            self.context.remember_placeholder(sent)
        return sent


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
    context: SectorContext,
    response_message: "LazyReply | Message",
    stream_generator: AsyncIterator,
    log_prefix: str,
    *,
    allow_empty: bool = False,
) -> str:
    reply = LazyReply.wrap(context, response_message)
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
    usage_chunk = None
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
                lambda: reply.write(text, entities or None),
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
        text, entities = render_preview(strip_internal_envelope(response), enabled=markdown_enabled, cursor=cursor)
        return await edit(text, telegram_entities(entities) if entities_ok else None)

    async def deliver_final() -> bool:
        """Post or edit the reply with the first message, then send the rest."""
        nonlocal buffer, response
        async with buffer_lock:
            if buffer:
                response += buffer
            buffer = ""

        chunks = render_chunks(strip_internal_envelope(response), enabled=markdown_enabled)
        if not chunks:
            return True

        text, entities = chunks[0]
        if not await edit(text, telegram_entities(entities) if entities_ok else None):
            return False

        thread_id = reply.message_thread_id
        for text, entities in chunks[1:]:
            payload = telegram_entities(entities) if entities_ok else None
            kwargs = {"chat_id": reply.chat_id, "text": text, "entities": payload}
            if type(thread_id) is int:
                kwargs["message_thread_id"] = thread_id
            sent = await retry_on_network_error(
                lambda kwargs=kwargs: context.bot.send_message(**kwargs),
                what="Telegram message",
                delays=(2.0, 5.0),
            )
            if sent is None:
                logger.error("%s could not deliver every message part", log_prefix)
                return False
            context.remember_sent(sent)
        return True

    try:
        async for chunk in stream_generator:
            if not chunk:
                ended_with_falsy = True
                break

            saw_chunk = True
            metadata = getattr(chunk, "response_metadata", None) or {}
            finish_reason = metadata.get("normalized_finish_reason") or metadata.get("finish_reason") or finish_reason
            if usage_details(chunk):
                usage_chunk = chunk
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
        visible = strip_internal_envelope(response)

        if not allow_empty and not visible.strip() and (saw_chunk or not ended_with_falsy):
            if finish_reason == "refusal":
                await reply.write("The model refused the request.")
            else:
                await reply.write("The model returned an empty response.")
            logger.warning("%s returned no text", log_prefix)

        if delivered:
            # A successful edit replaced the cursor. The turn owner decides
            # whether to forget the placeholder when the turn itself ends.
            context.settle_placeholder()
        elif visible.strip() or not allow_empty:
            context.mark_placeholder_unsettled()

        if usage_chunk is not None:
            record_usage(context, log_prefix, usage_chunk)

        if not delivered and visible.strip():
            logger.error("%s could not be delivered; not saving it", log_prefix)
            return ""

        logger.info(f"{log_prefix} Result: {visible}")
        return visible

    except NetworkError as e:
        # A delivery failure must not turn into another model call.
        context.mark_placeholder_unsettled()
        logger.error("%s delivery failed: %s", log_prefix, e)
        return ""
    except Exception as e:
        error_message = f"Error processing {log_prefix.lower()}: {redact_for_user(str(e), context.settings)}"
        await reply.write(error_message)
        context.finish_placeholder()
        logger.error(error_message)
        return ""

    finally:
        # A pending task here means the generator failed mid-stream; awaiting it
        # would paint stale text over the error message, so cancel it instead.
        if update_task and not update_task.done():
            update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await update_task
