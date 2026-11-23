from telegram import Update

from .agent import run_agent
from .sector_context import SectorContext


async def summarize_cmd(update: Update, context: SectorContext) -> None:
    await run_agent(update, context, force=True, system_prompt=context.settings.system_prompts.summarization)
