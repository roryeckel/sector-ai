"""Retry helpers for transient Telegram network errors.

Long-polling survives outages on its own, but a handler that dies mid-run
loses its update: Telegram has already marked it as delivered. These helpers
give a handler a bounded window (default: three attempts over ~20 seconds) to
get through a short network blip instead of silently dropping the message.
"""

import asyncio
import logging
from datetime import timedelta

from telegram.error import BadRequest, NetworkError, RetryAfter

logger = logging.getLogger(__name__)

#: Never sleep longer than this for a flood-control retry. A longer server
#: window is better spent failing this update than blocking the whole bot.
MAX_RETRY_AFTER = 30.0


def _retry_after_seconds(error: RetryAfter) -> float:
    """PTB may expose ``retry_after`` as seconds or as a timedelta."""
    value = error.retry_after
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)


async def retry_on_network_error(call, *, what: str, attempts: int = 3, delays: tuple[float, ...] = (5.0, 15.0)):
    """Await ``call()``, retrying while it raises a Telegram network error.

    Returns the call's result, or ``None`` when every attempt failed. Other
    exceptions propagate to the caller. ``BadRequest`` subclasses
    ``NetworkError``, but a rejected request will not succeed on a retry.
    ``RetryAfter`` is Telegram flood control: wait the requested window instead
    of hammering the same chat with edits that will only fail again.
    """
    for attempt in range(attempts):
        try:
            return await call()
        except BadRequest:
            raise
        except RetryAfter as error:
            if attempt == attempts - 1:
                logger.error("%s hit flood control after %d attempts: %s", what, attempts, error)
                return None
            delay = max(_retry_after_seconds(error), 0.0)
            if delay > MAX_RETRY_AFTER:
                # Retrying sooner than Telegram allows only fails again.
                logger.error("%s hit flood control for %.0fs; giving up", what, delay)
                return None
            logger.warning("%s hit flood control; retrying in %.0fs", what, delay)
            await asyncio.sleep(delay)
        except NetworkError as error:
            if attempt == attempts - 1:
                logger.error("%s failed after %d attempts: %s", what, attempts, error)
                return None
            delay = delays[min(attempt, len(delays) - 1)]
            logger.warning("%s hit a network error (%s); retrying in %.0fs", what, error, delay)
            await asyncio.sleep(delay)
