import argparse
import logging
import os
import warnings
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import ValidationError
from pydantic_settings import SettingsError
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PollAnswerHandler,
    PollHandler,
    filters,
)

from . import __version__
from .admin import clear_cmd, compact_cmd, model_callback, models_cmd, system_prompt_cmd, temperature_cmd, usage_cmd
from .agent import run_agent
from .autoreply import autoreply_cmd
from .characterize import characterize_cmd
from .chat import chat_cmd
from .chat_events import (
    CHAT_EVENT_FILTER,
    MESSAGE_EDIT_FILTER,
    handle_bot_membership,
    handle_chat_event,
    handle_edited_message,
    handle_poll_answer,
    handle_poll_result,
)
from .coding import code_cmd, html_cmd, svg_cmd
from .decision import decide_cmd
from .emoji import emoji_cmd
from .images import handle_photo, image_mode
from .logging_utils import configure_logging, redact_for_user
from .mcp_tools import MCPRegistry
from .poll import poll_cmd
from .sector_context import SectorContext
from .settings import LLM_PROVIDERS, LOG_LEVELS, Settings
from .summarize import summarize_cmd
from .tokens import tokens_cmd
from .topic import topic_poll_cmd

logger = logging.getLogger(__name__)

ALLOWED_UPDATES = [
    "message",
    "edited_message",
    "channel_post",
    "edited_channel_post",
    "callback_query",
    "poll",
    "poll_answer",
    "my_chat_member",
]


# Command handler for /start
async def start_cmd(update: Update, context: SectorContext) -> None:
    await update.message.reply_text("Hi! I'm a bot that can chat with you. Use /chat to start a conversation.")


# Message handler for non-command messages
async def handle_message(update: Update, context: SectorContext) -> None:
    await run_agent(update, context, debounce=True)


# Error handler
async def error_handler(update: Update, context: SectorContext) -> None:
    logger.exception("An error occurred", exc_info=context.error)
    error_str = str(context.error)
    settings = context.bot_data.get("settings")
    if settings is not None:
        # Backend errors can echo an Authorization header back to the chat.
        error_str = redact_for_user(error_str, settings)
    try:
        if update and update.message:
            await update.message.reply_text(error_str)
        elif update and update.callback_query:
            await update.callback_query.answer(error_str)
    except TelegramError:
        # The original error is already logged. A failed error reply, often
        # flood control, must not raise another uncaught error.
        pass


def build_parser() -> argparse.ArgumentParser:
    """The CLI is deliberately small: only settings you would want to change per run.

    Secrets are excluded on purpose. A bot token on the command line ends up in
    ``ps`` output and shell history, so it is env-only.
    """
    parser = argparse.ArgumentParser(prog="sector-ai", description="Sector AI Telegram Bot")
    parser.add_argument("--version", action="version", version=f"sector-ai {__version__}")
    parser.add_argument("--model", dest="default_model", help="Default model (SECTOR_AI_DEFAULT_MODEL)")
    parser.add_argument(
        "--vision-model",
        dest="vision_model",
        help="Model that describes photos when the default model cannot read images (SECTOR_AI_VISION_MODEL)",
    )
    parser.add_argument(
        "--provider",
        dest="llm_provider",
        type=str.lower,
        choices=LLM_PROVIDERS,
        help="Backend: Ollama's native API or an OpenAI-compatible endpoint (SECTOR_AI_LLM_PROVIDER)",
    )
    parser.add_argument("--ollama-url", dest="ollama_url", help="Ollama base URL (OLLAMA_URL)")
    parser.add_argument(
        "--openai-url",
        dest="openai_base_url",
        help="OpenAI-compatible base URL including /v1 (SECTOR_AI_OPENAI_BASE_URL)",
    )
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
    try:
        registry = MCPRegistry.from_file(settings.mcp_config) if settings.mcp_config else None
    except ValueError as error:
        parser.exit(2, f"{parser.prog}: {error}\n")

    if "SECTOR_AI_OLLAMA_CONTEXT_LENGTH" in os.environ or "SECTOR_AI_OLLAMA_CONTEXT_LENGTH" in dotenv_values(".env"):
        logger.warning(
            "SECTOR_AI_OLLAMA_CONTEXT_LENGTH is ignored; configure context length on the model server "
            "(OLLAMA_CONTEXT_LENGTH or a Modelfile)."
        )

    # Create the application with custom context type
    context_types = ContextTypes(context=SectorContext)
    application = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .context_types(context_types)
        # Photo/document uploads need more than the library's 5-second default.
        .connect_timeout(15.0)
        .read_timeout(30.0)
        .write_timeout(60.0)
        .pool_timeout(10.0)
        .build()
    )
    application.bot_data["settings"] = settings
    if registry is not None:
        application.bot_data["mcp_registry"] = registry

    endpoint = settings.openai_base_url if settings.llm_provider == "openai" else settings.ollama_url
    logger.info(
        "LLM backend: provider=%s model=%s host=%s prompt_cache=%s",
        settings.llm_provider,
        settings.default_model,
        urlsplit(endpoint).hostname or endpoint,
        settings.prompt_cache_mode,
    )
    mode = image_mode(settings)
    logger.info(
        "Photos: %s",
        {
            "native": "sent inline to the default model",
            "describe": f"described by {settings.vision_model}",
            "off": "recorded as placeholders (no image-capable model configured)",
        }[mode],
    )
    if mode == "native" and settings.vision_model.strip():
        logger.warning("SECTOR_AI_VISION_MODEL is ignored while SECTOR_AI_DEFAULT_MODEL_VISION is true")

    # Add command handlers
    # The first matching handler wins within this group. Edits must be captured
    # before every command (including admin controls) and before photo handling.
    # Edits, text and photos run concurrently (block=False) so a newer message
    # can supersede a turn that has not replied yet. Each turn still holds the
    # per-conversation lock, and controls/commands take it too.
    application.add_handler(MessageHandler(MESSAGE_EDIT_FILTER, handle_edited_message, block=False))
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
    application.add_handler(CommandHandler("usage", usage_cmd))
    application.add_handler(CommandHandler("compact", compact_cmd))
    application.add_handler(CommandHandler("autoreply", autoreply_cmd))

    # Add message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message, block=False))
    # Photos join the same agent turn as text; albums become one turn.
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo, block=False))
    application.add_handler(MessageHandler(CHAT_EVENT_FILTER, handle_chat_event))
    application.add_handler(PollHandler(handle_poll_result))
    application.add_handler(PollAnswerHandler(handle_poll_answer))
    application.add_handler(ChatMemberHandler(handle_bot_membership, ChatMemberHandler.MY_CHAT_MEMBER))

    application.add_error_handler(error_handler)

    # Start the bot
    # Retry bootstrapping forever: a network blip during get_me() must not
    # kill the process before polling even starts.
    application.run_polling(bootstrap_retries=-1, allowed_updates=ALLOWED_UPDATES)


if __name__ == "__main__":
    main()
