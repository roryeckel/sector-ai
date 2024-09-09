from telegram import Update

from .chat import handle_chat

from .sector_context import SectorContext


async def summarize_cmd(update: Update, context: SectorContext) -> None:
    await handle_chat(update, context, system_prompt=context.config_summarization_system_prompt)