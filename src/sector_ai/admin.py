import hashlib
import logging
from functools import wraps

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest

from .sector_context import SectorContext

logger = logging.getLogger(__name__)


def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: SectorContext, *args, **kwargs):
        if context is None or (
            update.effective_user and update.effective_user.username in context.settings.admin_usernames
        ):
            return await func(update, context, *args, **kwargs)

        logger.warning(f"Unauthorized access attempt by user: {update.effective_user.username} in update {update}")
        text = "You are not authorized to use this command."
        if update.message is not None:
            await update.message.reply_text(text)
        elif update.effective_message is not None:
            await update.effective_message.reply_text(text)
        elif update.callback_query is not None:
            await update.callback_query.answer(text, show_alert=True)

    return wrapper


async def clear_cmd(update: Update, context: SectorContext) -> None:
    context.clear_history()
    context.chat_system_prompt = context.settings.system_prompts.default
    await update.message.reply_text("Cleared context and set the default system prompt.")


@admin_only
async def compact_cmd(update: Update, context: SectorContext) -> None:
    # The count and the summary are both model/HTTP calls; say something first.
    status = await update.message.reply_text("Compacting...")
    before = await context.count_context_tokens()
    try:
        result = await context.compact()
    except Exception:
        logger.exception("Manual compaction failed")
        await status.edit_text("Compaction failed; the history was left unchanged. Check the logs.")
        return
    if not result["compacted"]:
        await status.edit_text("Not enough history to compact yet.")
        return

    after = await context.count_context_tokens()
    await status.edit_text(f"Compacted {result['evicted']} messages into a summary: {before} → {after} tokens.")


# async def model_cmd(update: Update, context: SectorContext) -> None:
# model_name = update.message.text.split(maxsplit=1)
# if len(model_name) > 1:
#     model_name = model_name[1].strip()
#     if not any(model_name == model['model'] for model in context.get_models()):
#         await update.message.reply_text(f'Model {model_name} is not available.')
#     else:
#         ollama = context.load_model(model_name)
#         await update.message.reply_text(f'Loaded model {ollama.model}.')
# else:
# await update.message.reply_text(f'Model {context.get_model()} is currently loaded.')


MODELS_PER_PAGE = 40


def _model_key(name: str) -> str:
    """A callback_data value short enough for Telegram's 64-byte limit."""
    return hashlib.sha1(name.encode("utf-8")).hexdigest()


def _button_label(label: str, limit: int = 60) -> str:
    encoded = label.encode("utf-8")
    if len(encoded) <= limit:
        return label
    return encoded[: limit - 3].decode("utf-8", "ignore") + "…"


def _models_keyboard(models: list, page: int) -> list:
    start = page * MODELS_PER_PAGE
    visible = models[start : start + MODELS_PER_PAGE]
    keyboard = [
        [InlineKeyboardButton(_button_label(model["label"]), callback_data=_model_key(model["model"]))]
        for model in visible
    ]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"page:{page - 1}"))
    if start + MODELS_PER_PAGE < len(models):
        nav.append(InlineKeyboardButton("Next ▶", callback_data=f"page:{page + 1}"))
    if nav:
        keyboard.append(nav)
    return keyboard


def _models_page_text(page: int, total: int) -> str:
    pages = max(1, (total + MODELS_PER_PAGE - 1) // MODELS_PER_PAGE)
    return f"Available models (page {page + 1}/{pages}):"


@admin_only
async def models_cmd(update: Update, context: SectorContext) -> None:
    models = await context.get_models()
    if not models:
        context.chat_data.pop("models", None)
        context.chat_data.pop("model_choices", None)
        await update.message.reply_text("No models available.")
        return

    # Keep the whole list for pagination, and map short callback keys to names.
    context.chat_data["models"] = models
    context.chat_data["model_choices"] = {_model_key(model["model"]): model["model"] for model in models}

    page = 0
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) > 1:
        try:
            page = max(0, int(parts[1]) - 1)
        except ValueError:
            page = 0
    pages = max(1, (len(models) + MODELS_PER_PAGE - 1) // MODELS_PER_PAGE)
    page = min(page, pages - 1)
    await update.message.reply_text(
        _models_page_text(page, len(models)),
        reply_markup=InlineKeyboardMarkup(_models_keyboard(models, page)),
    )


@admin_only
async def model_callback(update: Update, context: SectorContext) -> None:
    query = update.callback_query

    if query.data.startswith("page:"):
        models = context.chat_data.get("models")
        if not models:
            await query.answer("This menu has expired. Run /models again.", show_alert=True)
            return
        try:
            page = int(query.data.split(":", 1)[1])
        except ValueError:
            page = 0
        pages = max(1, (len(models) + MODELS_PER_PAGE - 1) // MODELS_PER_PAGE)
        page = min(max(page, 0), pages - 1)
        await query.answer()
        try:
            await query.edit_message_text(
                _models_page_text(page, len(models)),
                reply_markup=InlineKeyboardMarkup(_models_keyboard(models, page)),
            )
        except BadRequest as error:
            if "not modified" not in str(error).lower():
                raise
        return

    choices = context.chat_data.get("model_choices") or {}
    if query.data not in choices:
        await query.answer("This menu has expired. Run /models again.", show_alert=True)
        return
    model_name = choices[query.data]
    context.load_model(model_name)
    model_details = await context.get_model_details()
    model_details_details = model_details.get("details", {})
    model_info = model_details.get("model_info", {})
    await query.answer()

    message_parts = [f"Loaded model: {model_name}"]

    families = model_details_details.get("families")
    if families:
        message_parts.append(f"Families: {', '.join(families)}")

    parameter_size = model_details_details.get("parameter_size")
    if parameter_size:
        message_parts.append(f"Parameters: {parameter_size}")

    quantization_level = model_details_details.get("quantization_level")
    if quantization_level:
        message_parts.append(f"Quantization Level: {quantization_level}")

    finetune = model_info.get("general.finetune")
    if finetune:
        message_parts.append(f"Finetune: {finetune}")

    languages = model_info.get("general.languages")
    if languages:
        message_parts.append(f"Languages: {', '.join(languages)}")

    tags = model_info.get("general.tags")
    if tags:
        message_parts.append(f"Tags: {', '.join(tags)}")

    message = "\n".join(message_parts)
    try:
        await query.edit_message_text(text=message)
    except BadRequest as error:
        if "not modified" not in str(error).lower():
            raise


# @admin_only
# async def models_cmd(update: Update, context: SectorContext) -> None:
#     await update.message.reply_text(f'Available models: {", ".join(model["model"] for model in context.get_models())}')


@admin_only
async def temperature_cmd(update: Update, context: SectorContext) -> None:
    temperature = update.message.text.split(maxsplit=1)
    if len(temperature) > 1:
        context.bot_llm.temperature = float(temperature[1])
        await update.message.reply_text(f"Set the temperature to {temperature[1]}.")
    else:
        context.bot_llm.temperature = None
        await update.message.reply_text("Cleared the temperature.")


@admin_only
async def system_prompt_cmd(update: Update, context: SectorContext) -> None:
    new_system_prompt = update.message.text.split(maxsplit=1)
    if len(new_system_prompt) > 1:
        context.chat_system_prompt = new_system_prompt[-1]
        await update.message.reply_text("Updated the system prompt.")
    else:
        context.chat_system_prompt = context.settings.system_prompts.default
        await update.message.reply_text("Reset the system prompt to default.")
