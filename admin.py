from functools import wraps
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update

from .sector_context import SectorContext

logger = logging.getLogger(__name__)

def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: SectorContext, *args, **kwargs):
        if context is None or (update.effective_user and update.effective_user.username in context.config_admin_usernames):
            return await func(update, context, *args, **kwargs)
        else:
            logger.warning(f"Unauthorized access attempt by user: {update.effective_user.username} in update {update}")
            await update.message.reply_text("You are not authorized to use this command.")
    return wrapper

async def clear_cmd(update: Update, context: SectorContext) -> None:
    context.chat_message_history.clear()
    context.chat_system_prompt = context.config_default_system_prompt
    await update.message.reply_text('Cleared context and set the default system prompt.')

async def model_cmd(update: Update, context: SectorContext) -> None:
    # model_name = update.message.text.split(maxsplit=1)
    # if len(model_name) > 1:
    #     model_name = model_name[1].strip()
    #     if not any(model_name == model['model'] for model in context.get_models()):
    #         await update.message.reply_text(f'Model {model_name} is not available.')
    #     else:
    #         ollama = context.load_model(model_name)
    #         await update.message.reply_text(f'Loaded model {ollama.model}.')
    # else:
    await update.message.reply_text(f'Model {context.bot_ollama.model} is currently loaded.')

@admin_only
async def models_cmd(update: Update, context: SectorContext) -> None:
    models = context.get_models()
    keyboard = [[InlineKeyboardButton(model["model"], callback_data=model["model"])] for model in models]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Available models:', reply_markup=reply_markup)

@admin_only
async def model_callback(update: Update, context: SectorContext) -> None:
    query = update.callback_query
    model_name = query.data
    ollama = context.load_model(model_name)
    await query.answer()
    await query.edit_message_text(text=f'Loaded model {ollama.model}.')

# @admin_only
# async def models_cmd(update: Update, context: SectorContext) -> None:
#     await update.message.reply_text(f'Available models: {", ".join(model["model"] for model in context.get_models())}')

@admin_only
async def temperature_cmd(update: Update, context: SectorContext) -> None:
    temperature = update.message.text.split(maxsplit=1)
    if len(temperature) > 1:
        context.bot_ollama.temperature = float(temperature[1])
        await update.message.reply_text(f'Set the temperature to {temperature[1]}.')
    else:
        context.bot_ollama.temperature = None
        await update.message.reply_text('Cleared the temperature.')

@admin_only
async def system_prompt_cmd(update: Update, context: SectorContext) -> None:
    new_system_prompt = update.message.text.split(maxsplit=1)
    if len(new_system_prompt) > 1:
        context.chat_system_prompt = new_system_prompt[-1]
        await update.message.reply_text(f'Updated the system prompt.')
    else:
        context.chat_system_prompt = context.config_default_system_prompt
        await update.message.reply_text('Reset the system prompt to default.')
        