from pydantic import BaseModel, Field
from telegram import Update

from .coding import artifact_command
from .sector_context import SectorContext


class BasicPoll(BaseModel):
    """Public output shape retained for integrations."""

    question: str
    options: list[str] = Field(description="Telegram poll options")


async def poll_cmd(update: Update, context: SectorContext) -> None:
    await artifact_command(update, context, "poll")
