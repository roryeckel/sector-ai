"""Telegram's "typing..." indicator, kept alive while the bot works on a reply.

Show it only once the bot has committed to posting something. Telegram clears
the indicator after about five seconds, or as soon as the bot sends a message,
so it is resent on a short interval until the work ends.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from telegram.constants import ChatAction

logger = logging.getLogger(__name__)

REFRESH_SEC = 4.0


@contextlib.asynccontextmanager
async def typing(message) -> AsyncIterator[None]:
    """Show "typing" in ``message``'s chat (and topic) until the block exits.

    A failed chat action is cosmetic: it is logged and never fails the work.
    """
    if message is None:
        yield
        return

    async def send() -> None:
        try:
            # Never let a slow chat action hold up the work it decorates.
            async with asyncio.timeout(REFRESH_SEC):
                await message.reply_chat_action(ChatAction.TYPING)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.debug("Could not send the typing indicator: %s", error)

    stop = asyncio.Event()

    async def refresh() -> None:
        # Waiting on an event (not asyncio.sleep) always yields to the loop,
        # even where asyncio.sleep is patched out.
        while True:
            try:
                await asyncio.wait_for(stop.wait(), REFRESH_SEC)
                return
            except TimeoutError:
                await send()

    # The first action goes out before the work starts.
    await send()
    task = asyncio.create_task(refresh())
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
