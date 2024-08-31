import json
import logging
import warnings
import pkgutil

from .vision import handle_vision
from .autoreply import autoreply_cmd, decide_to_respond
from .chat import respond, chat_cmd
from .poll import poll_cmd
from .topic import topic_poll_cmd
from .emoji import emoji_cmd
from .decision import decide_cmd
from .sector_context import SectorContext
from .coding import code_cmd, html_cmd, svg_cmd
from .characterize import characterize_cmd
from .summarize import summarize_cmd
from .admin import model_callback, model_cmd, models_cmd, system_prompt_cmd, clear_cmd, temperature_cmd
from .tokens import tokens_cmd
import argparse
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from langchain_core.messages import AIMessage
from datetime import datetime, timedelta, UTC

warnings.filterwarnings("ignore", category=FutureWarning)

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Command handler for /start
async def start_cmd(update: Update, context: SectorContext) -> None:
    await update.message.reply_text('Hi! I\'m a bot that can chat with you. Use /chat to start a conversation.')

# Message handler for non-command messages
async def handle_message(update: Update, context: SectorContext) -> None:
    if (update.effective_message.reply_to_message and
        update.effective_message.reply_to_message.from_user.id == context.bot.id and
        update.effective_message.reply_to_message.text):
        new_chat_message = AIMessage(content=update.effective_message.reply_to_message.text)
        if not context.message_exists(new_chat_message):
            context.chat_message_history.append(new_chat_message)
        context.save_user_message(update.message)
        await respond(update, context)
    elif datetime.now(UTC) - update.effective_message.date < timedelta(minutes=7):
        await decide_to_respond(update, context)

# Error handler
async def error_handler(update: Update, context: SectorContext) -> None:
    logger.exception('An error occurred', exc_info=context.error)
    error_str = str(context.error)
    new_chat_message = AIMessage(content=error_str)
    context.chat_message_history.append(new_chat_message)
    await update.message.reply_text(error_str)

# Main function
def main() -> None:
    parser = argparse.ArgumentParser(description='Sector AI Telegram Bot')
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as config_file:
            config_file = json.load(config_file)
    else:
        config_file = json.loads(pkgutil.get_data(__name__, "default_config.json"))

    # Create the application with custom context type
    context_types = ContextTypes(context=SectorContext)
    application = Application.builder().token(config_file["telegram_bot_token"]).context_types(context_types).build()
    application.bot_data['config'] = config_file

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
    application.add_handler(CommandHandler("model", model_cmd))
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

if __name__ == '__main__':
    main()
