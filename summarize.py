from telegram import Update

from .chat import respond

from .sector_context import SectorContext


async def summarize_cmd(update: Update, context: SectorContext) -> None:
    await respond(update, context, system_prompt=context.config_summarization_system_prompt)