import logging

from telegram import Update

from .decision import make_decision
from .retry import retry_on_network_error
from .sector_context import SectorContext

logger = logging.getLogger(__name__)


async def should_respond(update: Update, context: SectorContext) -> bool | str:
    if context.save_user_message(update.message) and context.chat_autoreply_mode:
        try:
            should_respond = await retry_on_network_error(
                lambda: make_decision(
                    context.settings.system_prompts.decide,
                    context,
                    partial_variables={"input": update.effective_message.text},
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


async def autoreply_cmd(update: Update, context: SectorContext) -> None:
    current_mode = context.chat_autoreply_mode
    context.chat_autoreply_mode = not current_mode
    await update.message.reply_text(
        f"Autoreply mode is now {'enabled' if context.chat_autoreply_mode else 'disabled'}."
    )
