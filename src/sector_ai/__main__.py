import argparse
import logging
import warnings
from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage
from pydantic import ValidationError
from pydantic_settings import SettingsError
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from . import __version__
from .admin import clear_cmd, model_callback, models_cmd, system_prompt_cmd, temperature_cmd
from .autoreply import autoreply_cmd, should_respond
from .characterize import characterize_cmd
from .chat import chat_cmd, handle_chat
from .coding import code_cmd, html_cmd, svg_cmd
from .decision import decide_cmd
from .emoji import emoji_cmd
from .logging_utils import configure_logging
from .poll import poll_cmd
from .sector_context import SectorContext
from .settings import LOG_LEVELS, Settings
from .summarize import summarize_cmd
from .tokens import tokens_cmd
from .topic import topic_poll_cmd
from .vision import handle_vision

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


def build_parser() -> argparse.ArgumentParser:
    """The CLI is deliberately small: only settings you would want to change per run.

    Secrets are excluded on purpose. A bot token on the command line ends up in
    ``ps`` output and shell history, so it is env-only.
    """
    parser = argparse.ArgumentParser(prog="sector-ai", description="Sector AI Telegram Bot")
    parser.add_argument("--version", action="version", version=f"sector-ai {__version__}")
    parser.add_argument("--model", dest="default_model", help="Default Ollama model (SECTOR_AI_DEFAULT_MODEL)")
    parser.add_argument(
        "--vision-model", dest="vision_model", help="Ollama model used for images (SECTOR_AI_VISION_MODEL)"
    )
    parser.add_argument("--ollama-url", dest="ollama_url", help="Ollama base URL (OLLAMA_URL)")
    parser.add_argument(
        "--admin",
        dest="admin_usernames",
        action="append",
        metavar="USERNAME",
        help="Admin username, repeatable; replaces SECTOR_AI_ADMIN_USERNAMES when given",
    )
    parser.add_argument(
        "--log-level", dest="log_level", type=str.upper, choices=LOG_LEVELS, help="Logging level (SECTOR_AI_LOG_LEVEL)"
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build settings with the CLI values taking precedence over the environment."""
    overrides = {key: value for key, value in vars(args).items() if value is not None}
    return Settings(**overrides)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = settings_from_args(args)
    except ValidationError as error:
        # A container should log the reason, not a traceback.
        messages = "; ".join(entry["msg"] for entry in error.errors())
        parser.exit(2, f"{parser.prog}: {messages}\n")
    except SettingsError as error:
        parser.exit(2, f"{parser.prog}: {error}\n")

    warnings.filterwarnings("ignore", category=FutureWarning)
    configure_logging(settings)

    # Create the application with custom context type
    context_types = ContextTypes(context=SectorContext)
    application = (
        Application.builder().token(settings.telegram_bot_token.get_secret_value()).context_types(context_types).build()
    )
    application.bot_data["settings"] = settings

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
    application.add_handler(CallbackQueryHandler(model_callback))
    application.add_handler(CommandHandler("tokens", tokens_cmd))
    application.add_handler(CommandHandler("autoreply", autoreply_cmd))

    # Add message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_vision))

    # application.add_handler(PollAnswerHandler(handle_poll_answer))

    application.add_error_handler(error_handler)

    # Start the bot
    application.run_polling()


if __name__ == "__main__":
    main()
