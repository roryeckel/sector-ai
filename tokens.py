from telegram import Update
from .sector_context import SectorContext


async def tokens_cmd(update: Update, context: SectorContext) -> None:
    await update.message.reply_text(f'The current context is {context.bot_ollama.get_num_tokens_from_messages(context.chat_message_history)} tokens.')
