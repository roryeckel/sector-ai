"""Retry helpers for transient Telegram network errors.

Long-polling survives outages on its own, but a handler that dies mid-run
loses its update: Telegram has already marked it as delivered. These helpers
give a handler a bounded window (default: three attempts over ~20 seconds) to
get through a short network blip instead of silently dropping the message.
"""

import asyncio
import logging

from telegram.error import NetworkError

logger = logging.getLogger(__name__)


async def retry_on_network_error(call, *, what: str, attempts: int = 3, delays: tuple[float, ...] = (5.0, 15.0)):
    """Await ``call()``, retrying while it raises a Telegram network error.

    Returns the call's result, or ``None`` when every attempt failed. Other
    exceptions propagate to the caller.
    """
    for attempt in range(attempts):
        try:
            return await call()
        except NetworkError as error:
            if attempt == attempts - 1:
                logger.error("%s failed after %d attempts: %s", what, attempts, error)
                return None
            delay = delays[min(attempt, len(delays) - 1)]
            logger.warning("%s hit a network error (%s); retrying in %.0fs", what, error, delay)
            await asyncio.sleep(delay)
