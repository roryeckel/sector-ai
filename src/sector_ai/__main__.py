import argparse
import copy
import json
import logging
import warnings
from importlib import resources
from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from .admin import clear_cmd, model_callback, models_cmd, system_prompt_cmd, temperature_cmd
from .autoreply import autoreply_cmd, should_respond
from .characterize import characterize_cmd
from .chat import chat_cmd, handle_chat
from .coding import code_cmd, html_cmd, svg_cmd
from .decision import decide_cmd
from .emoji import emoji_cmd
from .mood import mood_cmd
from .poll import poll_cmd
from .quiz import next_cmd, quiz_answer_callback, quiz_cmd
from .roast import hype_cmd, roast_cmd
from .sector_context import SectorContext
from .summarize import summarize_cmd
from .tokens import tokens_cmd
from .topic import topic_poll_cmd
from .vision import handle_vision

warnings.filterwarnings("ignore", category=FutureWarning)

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# Command handler for /start
async def start_cmd(update: Update, context: SectorContext) -> None:
    await update.message.reply_text("Hi! I'm a bot that can chat with you. Use /chat to start a conversation.")


# Message handler for non-command messages
async def handle_message(update: Update, context: SectorContext) -> None:
    if (
        update.effective_message.reply_to_message
        and update.effective_message.reply_to_message.from_user.id == context.bot.id
        and update.effective_message.reply_to_message.text
    ):
        new_chat_message = AIMessage(content=update.effective_message.reply_to_message.text)
        if not context.message_exists(new_chat_message):
            context.chat_message_history.append(new_chat_message)
        context.save_user_message(update.message)
        await handle_chat(update, context)
    elif await should_respond(update, context):
        if datetime.now(UTC) - update.effective_message.date < timedelta(minutes=2):
            await handle_chat(update, context)


# Error handler
async def error_handler(update: Update, context: SectorContext) -> None:
    logger.exception("An error occurred", exc_info=context.error)
    error_str = str(context.error)
    new_chat_message = AIMessage(content=error_str)
    context.chat_message_history.append(new_chat_message)
    try:
        if update.message:
            await update.message.reply_text(error_str)
        else:
            await update.callback_query.answer(error_str)
    except BadRequest:
        pass


# Main function
def main() -> None:
    parser = argparse.ArgumentParser(description="Sector AI Telegram Bot")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load default config and merge user config on top
    default_config_text = resources.files("sector_ai").joinpath("default_config.json").read_text()
    default_config = json.loads(default_config_text)

    with open(args.config) as config_file:
        user_config = json.load(config_file)

    def _deep_merge(base: dict, override: dict) -> dict:
        """Merge override into base, filling in missing keys from base."""
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    config_file = _deep_merge(default_config, user_config)

    # Create the application with custom context type
    context_types = ContextTypes(context=SectorContext)
    application = Application.builder().token(config_file["telegram_bot_token"]).context_types(context_types).build()
    application.bot_data["config"] = config_file

    # Add command handlers
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("chat", chat_cmd))
    application.add_handler(CommandHandler("summarize", summarize_cmd))
    application.add_handler(CommandHandler("poll", poll_cmd))
    application.add_handler(CommandHandler("topic", topic_poll_cmd))
    application.add_handler(CommandHandler("emoji", emoji_cmd))
    application.add_handler(CommandHandler("code", code_cmd))
    application.add_handler(CommandHandler("svg", svg_cmd))
    application.add_handler(CommandHandler("html", html_cmd))
    application.add_handler(CommandHandler("decide", decide_cmd))
    application.add_handler(CommandHandler("clear", clear_cmd))
    application.add_handler(CommandHandler("system", system_prompt_cmd))
    application.add_handler(CommandHandler("characterize", characterize_cmd))
    application.add_handler(CommandHandler("temperature", temperature_cmd))
    application.add_handler(CommandHandler("models", models_cmd))
    application.add_handler(CallbackQueryHandler(model_callback, pattern="^model:"))
    application.add_handler(CommandHandler("tokens", tokens_cmd))
    application.add_handler(CommandHandler("autoreply", autoreply_cmd))
    application.add_handler(CommandHandler("mood", mood_cmd))
    application.add_handler(CommandHandler("roast", roast_cmd))
    application.add_handler(CommandHandler("hype", hype_cmd))
    application.add_handler(CommandHandler("quiz", quiz_cmd))
    application.add_handler(CommandHandler("next", next_cmd))
    application.add_handler(CallbackQueryHandler(quiz_answer_callback, pattern="^quiz:"))

    # Add message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_vision))

    # application.add_handler(PollAnswerHandler(handle_poll_answer))

    application.add_error_handler(error_handler)

    # Start the bot
    application.run_polling()


if __name__ == "__main__":
    main()
