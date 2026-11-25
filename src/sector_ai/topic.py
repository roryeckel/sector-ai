from telegram import Update

from .capabilities import TopicContent as TopicPoll
from .sector_context import SectorContext

__all__ = ["TopicPoll", "topic_poll_cmd"]


async def topic_poll_cmd(update: Update, context: SectorContext) -> None:
    from .agent import run_agent

    await run_agent(update, context, command="topic")
