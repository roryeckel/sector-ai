"""Native Telegram reactions, with explicit targets and honest delivery status."""

import asyncio
import logging

from langchain_core.messages import AIMessage
from telegram import ReactionTypeEmoji
from telegram.constants import ReactionEmoji
from telegram.error import BadRequest, NetworkError, TelegramError

from .logging_utils import redact_for_user
from .tools import AgentTool

logger = logging.getLogger(__name__)

#: Telegram's fixed set of non-custom reaction emoji. Sorted so the tool schema
#: and the cached tool prefix stay stable across processes.
ALLOWED_REACTIONS = tuple(sorted(item.value for item in ReactionEmoji))


def _reaction_aliases() -> dict[str, str]:
    """Accept the variation-selector forms clients and models commonly emit."""
    aliases = {}
    for emoji in ALLOWED_REACTIONS:
        aliases[emoji] = emoji
        aliases.setdefault(emoji.replace("\ufe0f", ""), emoji)
        aliases.setdefault(emoji + "\ufe0f", emoji)
    return aliases


_CANONICAL_REACTIONS = _reaction_aliases()


def canonical_reaction(value) -> str | None:
    """Return Telegram's canonical emoji string, or None when it is unsupported."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    return _CANONICAL_REACTIONS.get(candidate) or _CANONICAL_REACTIONS.get(candidate.replace("\ufe0f", ""))


def last_bot_message_id(context) -> int | None:
    """The bot's most recent text message, not an artifact receipt."""
    for item in reversed(context.chat_message_history):
        if not isinstance(item, AIMessage):
            continue
        metadata = item.additional_kwargs.get("telegram") or {}
        if not (metadata.get("sender") or {}).get("is_bot"):
            continue
        identifier = metadata.get("message_id")
        if type(identifier) is int:
            return identifier
    return None


def resolve_target(context, update, arguments) -> int:
    """Map the tool's target selector to a message ID in this conversation."""
    message = getattr(update, "effective_message", None)
    explicit = arguments.get("message_id")
    target = arguments.get("target") or "current"
    if explicit is not None and arguments.get("target") is not None:
        raise ValueError("Provide either message_id or target, not both.")
    if explicit is not None:
        if type(explicit) is not int or explicit < 1:
            raise ValueError("message_id must be a positive integer.")
        return explicit
    if target == "current":
        identifier = getattr(message, "message_id", None)
        if type(identifier) is not int:
            raise ValueError("The current message has no Telegram message ID.")
        return identifier
    if target == "last_bot":
        identifier = last_bot_message_id(context)
        if identifier is None:
            raise ValueError("There is no earlier bot message in this conversation to react to.")
        return identifier
    if target == "reply_to":
        reply = getattr(message, "reply_to_message", None)
        identifier = getattr(reply, "message_id", None)
        if type(identifier) is not int:
            raise ValueError("The current message is not a reply, so there is no reply target to react to.")
        return identifier
    raise ValueError("Unknown reaction target. Use current, last_bot, or reply_to.")


async def deliver_reaction(context, message_id: int, emoji: str) -> dict:
    """Set one reaction. Re-setting the same reaction is idempotent, so a retry is safe."""
    reaction = {"emoji": emoji, "message_id": message_id}
    chat_id = context._chat_id
    if type(chat_id) is not int:
        return {"status": "error", "error": "No Telegram chat is bound to this reaction."}
    try:
        try:
            confirmed = await context.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji)],
            )
        except BadRequest:
            # A definite rejection is not a transport failure and must not be retried.
            raise
        except NetworkError:
            # Unlike a message send, a repeated reaction cannot duplicate anything.
            await asyncio.sleep(0.5)
            confirmed = await context.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji)],
            )
        if confirmed is False:
            return {
                "status": "delivery_failed",
                "reaction": {
                    **reaction,
                    "status": "delivery_failed",
                    "error": "Telegram did not confirm the reaction.",
                },
            }
        return {
            "status": "delivered",
            "reaction": {**reaction, "status": "delivered"},
            "note": "Reaction set. You may finish without another message.",
        }
    except asyncio.CancelledError:
        result = {
            "status": "delivery_uncertain",
            "reaction": {
                **reaction,
                "status": "delivery_uncertain",
                "error": "Reaction delivery was interrupted; do not automatically resend.",
            },
        }
        context.conversation_data["interrupted_delivery"] = {
            "tool_call_id": context.conversation_data.get("active_tool_call_id"),
            "result": result,
        }
        raise
    except BadRequest as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction failed: %s", message)
        return {
            "status": "delivery_failed",
            "reaction": {**reaction, "status": "delivery_failed", "error": message},
        }
    except NetworkError as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction delivery was uncertain: %s", message)
        return {
            "status": "delivery_uncertain",
            "reaction": {
                **reaction,
                "status": "delivery_uncertain",
                "error": "Reaction delivery is uncertain; do not automatically resend.",
            },
        }
    except TelegramError as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction failed: %s", message)
        return {
            "status": "delivery_failed",
            "reaction": {**reaction, "status": "delivery_failed", "error": message},
        }
    except Exception as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction failed: %s", message)
        return {
            "status": "delivery_failed",
            "reaction": {**reaction, "status": "delivery_failed", "error": message},
        }


def reaction_tool(update, context) -> AgentTool:
    """The reaction-only native tool; speaking remains the normal text path."""

    async def invoke(arguments):
        emoji = canonical_reaction(arguments.get("emoji"))
        if emoji is None:
            return {
                "status": "error",
                "error": (
                    "Unsupported reaction emoji. Use one of Telegram's standard reaction emoji, "
                    "such as 👍, ❤, 🔥, 🎉, or 😂."
                ),
            }
        try:
            message_id = resolve_target(context, update, arguments)
        except ValueError as error:
            return {"status": "error", "error": str(error)}
        return await deliver_reaction(context, message_id, emoji)

    return AgentTool(
        name="react_to_message",
        description=(
            "Set one standard Telegram reaction on the current message, your own most recent text message, "
            "the message this turn replies to, or a specific past message in this conversation. "
            "Bots can set one reaction per message. This tool only sets the reaction; if you also want to say "
            "something, include that text in the same response as this tool call. "
            "Do not claim a reaction you did not set."
        ),
        parameters={
            "type": "object",
            "properties": {
                "emoji": {
                    "type": "string",
                    "enum": list(ALLOWED_REACTIONS),
                    "description": "One standard Telegram reaction emoji.",
                },
                "target": {
                    "type": "string",
                    "enum": ["current", "last_bot", "reply_to"],
                    "default": "current",
                    "description": (
                        "Which message to react to. current is the message that triggered this turn; "
                        "last_bot is your own most recent text message; reply_to is the message this turn replies to."
                    ),
                },
                "message_id": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "A specific past message ID in this conversation. Do not combine with target.",
                },
            },
            "required": ["emoji"],
            "additionalProperties": False,
        },
        invoke=invoke,
        timeout=context.settings.tool_timeout,
    )
