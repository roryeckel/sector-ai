"""Render model markdown into Telegram's text-plus-entities form.

Telegram clients do not parse markdown; the Bot API renders typed ranges of the
plain text, called entities. ``telegramify-markdown`` converts markdown to that
pair (with the UTF-16 offsets Telegram expects), so the bot never has to escape
MarkdownV2 or keep HTML tags balanced while a reply is still streaming.

Formatting is best effort: any failure returns the text unchanged, because a
decorative feature must never cost a reply.
"""

from __future__ import annotations

import logging

from telegram import MessageEntity
from telegramify_markdown import RenderConfig, convert, split_entities
from telegramify_markdown.entity import MessageEntity as RenderedEntity

logger = logging.getLogger(__name__)

#: Telegram's limit for one message, measured in UTF-16 code units.
MESSAGE_LIMIT = 4096

#: Schemes Telegram accepts in a ``text_link`` entity. The converter passes any
#: destination through, and one invalid URL rejects the whole message.
SAFE_LINK_PREFIXES = ("http://", "https://", "tg://", "mailto:")


def _build_config() -> RenderConfig:
    """A private render config with chat-friendly symbols.

    The library's defaults prefix headings with emoji; a chat reply reads
    better with headings as plain bold/underlined text.
    """
    config = RenderConfig.isolated()
    for level in range(1, 7):
        setattr(config.markdown_symbol, f"heading_level_{level}", "")
    config.markdown_symbol.unordered_list_item = "•"
    config.markdown_symbol.ordered_list_suffix = "."
    return config


_RENDER_CONFIG = _build_config()


def utf16_len(text: str) -> int:
    """The length Telegram measures: characters outside the BMP count twice."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _allowed(entity: RenderedEntity) -> bool:
    if entity.type != "text_link":
        return True
    url = (entity.url or "").strip().lower()
    return url.startswith(SAFE_LINK_PREFIXES)


def render_markdown(text: str, *, enabled: bool = True) -> tuple[str, list[RenderedEntity]]:
    """Convert markdown to plain text and the entities that style it."""
    if not enabled or not text:
        return text, []
    try:
        rendered, entities = convert(text, config=_RENDER_CONFIG)
    except Exception:
        logger.warning("Could not render markdown; sending plain text", exc_info=True)
        return text, []
    return rendered, [entity for entity in entities if _allowed(entity)]


def telegram_entities(entities: list[RenderedEntity]) -> list[MessageEntity] | None:
    """Translate the library's entities into python-telegram-bot's type."""
    converted = []
    for entity in entities:
        try:
            converted.append(MessageEntity.de_json(entity.to_dict()))
        except Exception:
            logger.warning("Dropping an unreadable entity (%s)", entity.type, exc_info=True)
    return converted or None


def split_for_telegram(
    text: str, entities: list[RenderedEntity], limit: int = MESSAGE_LIMIT
) -> list[tuple[str, list[RenderedEntity]]]:
    """Split rendered text into sendable messages, clipping entities at breaks."""
    if limit <= 0:
        return [(text, entities)]
    chunks = split_entities(text, entities, limit)
    if not chunks and text.strip():
        return [(text, entities)]
    return chunks


def render_preview(text: str, *, enabled: bool = True, cursor: str = "") -> tuple[str, list[RenderedEntity]]:
    """Render the first sendable message, with the streaming cursor appended.

    The cursor is added after conversion so it can never land inside a code
    block or link, and the preview is trimmed to the first message before it.
    """
    rendered, entities = render_markdown(text, enabled=enabled)
    budget = MESSAGE_LIMIT - utf16_len(cursor)
    if budget > 0 and utf16_len(rendered) > budget:
        chunks = split_for_telegram(rendered, entities, budget)
        rendered, entities = chunks[0] if chunks else ("", [])
    return rendered + cursor, entities


def render_chunks(text: str, *, enabled: bool = True) -> list[tuple[str, list[RenderedEntity]]]:
    """Render a complete reply as the sequence of messages it needs."""
    rendered, entities = render_markdown(text, enabled=enabled)
    return split_for_telegram(rendered, entities)
