import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from telegram import Message, MessageEntity
from telegram.error import BadRequest, NetworkError, TelegramError

from .chat_metadata import strip_internal_envelope
from .debounce import commit_or_stop
from .logging_utils import user_error_text
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
        #: The last streamed response was not fully delivered. Its text must not
        #: be replaced by a fallback such as "empty response".
        self.failed = False

    @classmethod
    def wrap(cls, context: SectorContext, target) -> "LazyReply":
        """Accept a ``LazyReply`` or an already posted message."""
        return target if isinstance(target, LazyReply) else cls(context, message=target)

    @property
    def chat_id(self):
        return getattr(self.message or self.source, "chat_id", None)

    @property
    def message_thread_id(self):
        """The forum topic to post into; ``None`` outside topics.

        Replies in ordinary supergroups also carry a thread ID, but it names a
        reply chain, not a topic, and must not be sent back (as PTB does).
        """
        for message in (self.message, self.source):
            if message is not None:
                if getattr(message, "is_topic_message", False) is True:
                    thread = getattr(message, "message_thread_id", None)
                    return thread if type(thread) is int else None
        return None

    async def notify(self, text: str) -> bool:
        """Best-effort status or error text; never raises a Telegram error.

        If this reply already shows text, the notice is posted as a new message
        instead of replacing what the user is reading.
        """
        if self.message is None and self.source is None:
            # No chat to post into (a reply built without a source).
            return False
        try:
            if self.message is not None and self.source is not None:
                commit_or_stop(self.context)
                sent = await self.source.reply_text(text)
                self.context.remember_sent(sent)
            else:
                await self.write(text)
            return True
        except TelegramError as error:
            logger.error("Could not post a notice to the chat: %s", error)
            return False

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
    # The final edit of reply.message succeeded, so no cursor is left in it.
    final_edit_ok = False
    # Texts of the messages the final delivery got onto the screen.
    delivered_parts: list[str] = []
    reply.failed = False

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

    async def send_part(text: str, entities) -> bool:
        """Send one continuation message; formatting is dropped if Telegram rejects it."""
        kwargs = {"chat_id": reply.chat_id, "text": text, "entities": entities}
        thread_id = reply.message_thread_id
        if type(thread_id) is int:
            kwargs["message_thread_id"] = thread_id
        try:
            sent = await retry_on_network_error(
                lambda: context.bot.send_message(**kwargs),
                what="Telegram message",
                delays=(2.0, 5.0),
            )
        except BadRequest as error:
            if not entities:
                logger.error("%s message part was rejected: %s", log_prefix, error)
                return False
            logger.warning("Telegram rejected formatted text (%s); falling back to plain", error)
            return await send_part(text, None)
        if sent is None:
            return False
        context.remember_sent(sent)
        return True

    async def deliver_final() -> bool:
        """Post or edit the reply with the first message, then send the rest."""
        nonlocal buffer, response, final_edit_ok
        async with buffer_lock:
            if buffer:
                response += buffer
            buffer = ""

        chunks = render_chunks(strip_internal_envelope(response), enabled=markdown_enabled)
        if not chunks:
            final_edit_ok = True
            return True

        text, entities = chunks[0]
        if not await edit(text, telegram_entities(entities) if entities_ok else None):
            return False
        # The cursor is gone from the first message; a later part failing does
        # not put it back.
        final_edit_ok = True
        delivered_parts.append(text)

        for text, entities in chunks[1:]:
            if not await send_part(text, telegram_entities(entities) if entities_ok else None):
                logger.error("%s could not deliver every message part", log_prefix)
                return False
            delivered_parts.append(text)
        return True

    settled = False

    def settle() -> None:
        """Track a cursor only on the message whose final edit failed."""
        nonlocal settled
        settled = True
        if reply.message is None:
            return
        if final_edit_ok:
            context.settle_placeholder(reply.message)
        else:
            context.mark_placeholder_unsettled(reply.message)

    async def strip_cursor() -> None:
        """Take the cursor out of the preview with the text received so far."""
        nonlocal final_edit_ok
        if reply.message is None or final_edit_ok:
            return
        chunks = render_chunks(strip_internal_envelope(response + buffer), enabled=markdown_enabled)
        if not chunks:
            return
        text, entities = chunks[0]
        if await edit(text, telegram_entities(entities) if entities_ok else None):
            final_edit_ok = True

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
        settle()

        if usage_chunk is not None:
            record_usage(context, log_prefix, usage_chunk)

        if not delivered and visible.strip():
            reply.failed = True
            if delivered_parts:
                # Record what the user actually sees, so the next turn neither
                # repeats it nor claims the parts that never arrived.
                shown = "\n".join(delivered_parts)
                logger.error("%s was only partly delivered; saving the delivered part", log_prefix)
                return shown
            logger.error("%s could not be delivered; not saving it", log_prefix)
            return ""

        if not allow_empty and not visible.strip() and (saw_chunk or not ended_with_falsy):
            if finish_reason == "refusal":
                await reply.notify("The model refused the request.")
            else:
                await reply.notify("The model returned an empty response.")
            logger.warning("%s returned no text", log_prefix)

        logger.info(f"{log_prefix} Result: {visible}")
        return visible

    except NetworkError as e:
        # A delivery failure must not turn into another model call.
        reply.failed = True
        settle()
        logger.error("%s delivery failed: %s", log_prefix, e)
        return "\n".join(delivered_parts)
    except Exception as e:
        reply.failed = True
        error_message = f"Error processing {log_prefix.lower()}: {user_error_text(e, context.settings)}"
        logger.error(error_message)
        if update_task and not update_task.done():
            # A stale preview edit must not land after the final text.
            update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await update_task
        if reply.message is not None and reply.source is not None:
            if not final_edit_ok:
                # Nothing complete is on screen yet: finish the partial text.
                # A reply that was already delivered must never be sent again.
                await deliver_final()
            # Keep what the user was reading; the error follows as its own message.
            await reply.notify(error_message)
        elif await reply.notify(error_message) and reply.message is not None:
            # An already posted message: the notice replaced its preview.
            final_edit_ok = True
        settle()
        # Save what is on screen, so the next turn neither repeats it nor
        # claims the parts that never arrived.
        return "\n".join(delivered_parts)

    finally:
        # A pending task here means the generator failed mid-stream; awaiting it
        # would paint stale text over the error message, so cancel it instead.
        if update_task and not update_task.done():
            update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await update_task
        if not settled:
            # Cancelled or interrupted before the final delivery (a run
            # timeout lands here). Without this the preview keeps its cursor
            # and is never cleaned up.
            try:
                await asyncio.wait_for(strip_cursor(), timeout=5)
            except asyncio.CancelledError:
                settle()
                raise
            except Exception:
                logger.warning("%s could not remove the cursor from its preview", log_prefix)
            settle()
