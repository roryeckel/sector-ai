import logging
from telegram import Update
from .chat import handle_chat
from .decision import make_decision
from .sector_context import SectorContext

logger = logging.getLogger(__name__)

async def should_respond(update: Update, context: SectorContext) -> bool | str:
    if context.save_user_message(update.message) and context.chat_autoreply_mode:
        try:
            should_respond = await make_decision(
                context.config_decide_system_prompt,
                context,
                partial_variables={'input': update.effective_message.text})
        except Exception as e:
            logger.exception('Decision Error', exc_info=e)
            should_respond = False
        if isinstance(should_respond, str):
            logger.error(f'Autorespond Decision Error: {should_respond}')
        return should_respond

async def autoreply_cmd(update: Update, context: SectorContext) -> None:
    current_mode = context.chat_autoreply_mode
    context.chat_autoreply_mode = not current_mode
    await update.message.reply_text(f'Autoreply mode is now {"enabled" if context.chat_autoreply_mode else "disabled"}.')