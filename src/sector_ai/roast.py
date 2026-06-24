import logging

from langchain_core.messages import HumanMessage
from telegram import Update

from .sector_context import SectorContext
from .streaming_handler import handle_streaming_response

logger = logging.getLogger(__name__)


def _resolve_target(update: Update, context: SectorContext) -> tuple[str, str] | None:
    """Parse target username and gather their messages. Returns (username, messages_text) or None."""
    args = update.message.text.split(maxsplit=1)
    if len(args) > 1:
        target = args[1].strip().lstrip("@")
    else:
        target = update.message.from_user.username
        if not target:
            return None

    target_messages = [
        msg.content
        for msg in context.chat_message_history
        if isinstance(msg, HumanMessage) and msg.name and msg.name.lower() == target.lower()
    ]

    if not target_messages:
        return None

    return (target, "\n".join(target_messages))


async def _social_cmd(update: Update, context: SectorContext, system_prompt: str, label: str) -> None:
    result = _resolve_target(update, context)
    if result is None:
        args = update.message.text.split(maxsplit=1)
        if len(args) > 1:
            target = args[1].strip().lstrip("@")
            await update.message.reply_text(f"I don't have any recent messages from @{target} to work with!")
        else:
            username = update.message.from_user.username
            if username:
                await update.message.reply_text(
                    f"I don't have any recent messages from @{username} to work with!"
                )
            else:
                await update.message.reply_text(
                    "You don't have a Telegram username set! Please specify a target: /roast @someone"
                )
        return

    target, messages_text = result

    system_template_dict = await context.get_system_template_dict()
    system_template_dict["target_username"] = target
    system_template_dict["target_messages"] = messages_text
    prompt_template = context.get_templated_messages(system_prompt=system_prompt)
    chain = prompt_template | context.bot_ollama

    response_message = await update.message.reply_text("Processing...")
    try:
        stream_generator = chain.astream(system_template_dict)
        await handle_streaming_response(context, response_message, stream_generator, label)
    except Exception as e:
        await response_message.edit_text(f"Error processing {label.lower()}: {e}")


async def roast_cmd(update: Update, context: SectorContext) -> None:
    await _social_cmd(update, context, context.config_roast_system_prompt, "Roast")


async def hype_cmd(update: Update, context: SectorContext) -> None:
    await _social_cmd(update, context, context.config_hype_system_prompt, "Hype")
