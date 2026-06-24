import logging
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from telegram import Update
from telegram.helpers import escape_markdown

from .sector_context import SectorContext

logger = logging.getLogger(__name__)


class Emotion(BaseModel):
    name: str = Field(description="The name of the emotion")
    emoji: str = Field(description="A single emoji representing this emotion")
    intensity: int = Field(description="Intensity from 1-10", ge=1, le=10)


class MoodReport(BaseModel):
    overall_mood: str = Field(description="1-2 sentence description of the overall mood")
    energy_level: int = Field(description="Energy level from 1-10", ge=1, le=10)
    emotions: List[Emotion] = Field(description="Top 3-5 emotions detected in the chat")
    vibe_metaphor: str = Field(description="A creative metaphor for the chat vibe")
    top_topics: List[str] = Field(description="Top 3 topics being discussed")


def _render_bar(emoji: str, value: int, max_val: int = 10) -> str:
    filled = emoji * value
    empty = "░" * (max_val - value)
    return f"{filled}{empty} {value}/{max_val}"


def _esc(text: str) -> str:
    return escape_markdown(text, version=1)


def render_mood_report(report: MoodReport) -> str:
    lines = ["🎭 *Vibe Check*\n"]
    lines.append(f"*Overall:* {_esc(report.overall_mood)}\n")
    lines.append(f"*Energy:* {_render_bar('⚡', report.energy_level)}\n")
    lines.append("*Emotions:*")
    for emotion in report.emotions:
        lines.append(f"{emotion.emoji} {_esc(emotion.name)}: {_render_bar(emotion.emoji, emotion.intensity)}")
    lines.append(f'\n*Vibe:* "{_esc(report.vibe_metaphor)}"')
    lines.append(f"\n*Topics:* {_esc(', '.join(report.top_topics))}")
    return "\n".join(lines)


async def mood_cmd(update: Update, context: SectorContext) -> None:
    if not context.chat_message_history:
        await update.message.reply_text("No recent messages to analyze!")
        return

    context.bot_ollama.format = "json"
    try:
        system_template_dict = await context.get_system_template_dict()
        parser = PydanticOutputParser(pydantic_object=MoodReport)
        prompt_template = context.get_templated_messages(system_prompt=context.config_mood_system_prompt)
        chain = prompt_template | context.bot_ollama | parser
        mood_response = await chain.ainvoke({"format_instructions": parser.get_format_instructions(), **system_template_dict})
        logger.info(f"Mood Response: {mood_response}")

        rendered = render_mood_report(mood_response)
        await update.message.reply_text(rendered, parse_mode="Markdown")
    finally:
        context.bot_ollama.format = ""
