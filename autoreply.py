import logging
from telegram import Update
from .chat import respond
from .decision import make_decision
from .sector_context import SectorContext

logger = logging.getLogger(__name__)

async def decide_to_respond(update: Update, context: SectorContext) -> None:
    if context.save_user_message(update.message) and context.chat_autoreply_mode:
        try:
            should_respond = await make_decision(
                context.config_decide_system_prompt,
                context,
                partial_variables={'input': update.effective_message.text})
        except Exception as e:
            logger.exception('Decision Error', exc_info=e)
            should_respond = False
        if should_respond is True:
            await respond(update, context)
        elif isinstance(should_respond, str):
            logger.error(f'Autorespond Decision Error: {should_respond}')

async def autoreply_cmd(update: Update, context: SectorContext) -> None:
    current_mode = context.chat_autoreply_mode
    context.chat_autoreply_mode = not current_mode
    await update.message.reply_text(f'Autoreply mode is now {"enabled" if context.chat_autoreply_mode else "disabled"}.')