import logging

from telegram import Update

from .decision import make_decision
from .history import scoped_control
from .retry import retry_on_network_error
from .sector_context import SectorContext

logger = logging.getLogger(__name__)


async def should_respond(update: Update, context: SectorContext) -> bool | str:
    """Engagement only: ingestion belongs to the graph, not this decision."""
    message = update.effective_message
    text = message.text or getattr(message, "caption", None) or ""
    if not isinstance(text, str):
        text = ""
    photos = getattr(message, "photo", None)
    has_photo = isinstance(photos, (list, tuple)) and bool(photos)
    if context.chat_autoreply_mode and (text or has_photo) and not text.startswith("/"):
        incoming = text or "[photo without a caption]"
        try:
            context.bind_update(update)
            should_respond = await retry_on_network_error(
                lambda: make_decision(
                    context.settings.system_prompts.decide_autoreply,
                    context,
                    partial_variables={"input": incoming},
                    update=update,
                ),
                what="Autoreply decision",
            )
        except Exception as e:
            logger.exception("Decision Error", exc_info=e)
            should_respond = False
        if should_respond is None:
            return False
        if isinstance(should_respond, str):
            logger.error(f"Autorespond Decision Error: {should_respond}")
            return False
        return should_respond


@scoped_control
async def autoreply_cmd(update: Update, context: SectorContext) -> None:
    """Toggle autoreply for this conversation (this forum topic only)."""
    current_mode = context.chat_autoreply_mode
    context.chat_autoreply_mode = not current_mode
    await update.message.reply_text(
        f"Autoreply mode is now {'enabled' if context.chat_autoreply_mode else 'disabled'}."
    )
