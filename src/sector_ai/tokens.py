import logging

from telegram import Update

from .admin import admin_only
from .history import scoped_control
from .sector_context import SectorContext

logger = logging.getLogger(__name__)


@admin_only
@scoped_control
async def tokens_cmd(update: Update, context: SectorContext) -> None:
    try:
        tokens = await context.count_context_tokens()
    except Exception:
        logger.exception("Token count failed")
        await update.message.reply_text("Could not count tokens. Check the logs for details.")
        return

    suffix = "" if context.token_count_is_exact else " (estimated)"
    length = await context.get_context_length()
    usage = f" of {length} ({100 * tokens / length:.1f}%)" if length else ""
    await update.message.reply_text(f"The current context is {tokens} tokens{suffix}{usage}.")
