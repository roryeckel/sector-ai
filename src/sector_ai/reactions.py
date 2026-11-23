"""Native Telegram reactions, with explicit targets and honest delivery status."""

import asyncio
import logging

from langchain_core.messages import AIMessage
from telegram import ReactionTypeEmoji
from telegram.constants import ReactionEmoji
from telegram.error import BadRequest, NetworkError, TelegramError

from .chat_metadata import actor_metadata, text_value
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


def own_message_ids(context) -> set[int]:
    """Message IDs the bot itself sent: every recorded send, plus bot messages in history."""
    identifiers = {item for item in context.conversation_data.get("sent_message_ids", []) if type(item) is int}
    for item in context.chat_message_history:
        if not isinstance(item, AIMessage):
            continue
        metadata = item.additional_kwargs.get("telegram") or {}
        if not (metadata.get("sender") or {}).get("is_bot"):
            continue
        identifier = metadata.get("message_id")
        if type(identifier) is int:
            identifiers.add(identifier)
    return identifiers


def target_details(context, update, message_id: int) -> dict:
    """Who wrote the message and what it says, so a receipt is self-describing."""
    message = getattr(update, "effective_message", None)
    candidates = (message, getattr(message, "reply_to_message", None))
    for candidate in candidates:
        if candidate is None or getattr(candidate, "message_id", None) != message_id:
            continue
        sender = actor_metadata(getattr(candidate, "from_user", None))
        if not sender:
            sender = actor_metadata(getattr(candidate, "sender_chat", None))
        excerpt = text_value(getattr(candidate, "text", None), 200) or text_value(
            getattr(candidate, "caption", None), 200
        )
        return _target(sender, excerpt)
    for item in reversed(context.chat_message_history):
        metadata = item.additional_kwargs.get("telegram") or {}
        if metadata.get("message_id") != message_id:
            continue
        return _target(metadata.get("sender") or {}, text_value(item.content, 200))
    return {}


def _target(sender, excerpt) -> dict:
    result = {}
    if sender:
        result["sender"] = sender
    if excerpt:
        result["excerpt"] = excerpt
    return result


def is_own_message(context, update, message_id: int) -> bool:
    """Whether the target is a message the bot sent, so its receipt can say so."""
    if message_id in own_message_ids(context):
        return True
    sender = target_details(context, update, message_id).get("sender") or {}
    if sender.get("is_bot") is True:
        return True
    return sender.get("id") == getattr(context.bot, "id", None)


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
    if target == "reply_to":
        reply = getattr(message, "reply_to_message", None)
        identifier = getattr(reply, "message_id", None)
        if type(identifier) is not int:
            raise ValueError("The current message is not a reply, so there is no reply target to react to.")
        return identifier
    raise ValueError("Unknown reaction target. Use current, reply_to, or a specific message_id.")


def _receipt(status: str, reaction: dict, target: dict, *, note: str | None = None) -> dict:
    result = {"status": status, "reaction": reaction, "target": target or {}}
    if note:
        result["note"] = note
    return result


def _target_label(target: dict) -> str:
    sender = target.get("sender") or {}
    return sender.get("username") or sender.get("display_name") or "that user"


async def deliver_reaction(context, message_id: int, emoji: str, target: dict | None = None) -> dict:
    """Set one reaction. Re-setting the same reaction is idempotent, so a retry is safe."""
    target = target or {}
    reaction = {"emoji": emoji, "message_id": message_id}
    chat_id = context._chat_id
    if type(chat_id) is not int:
        return {"status": "error", "error": "No Telegram chat is bound to this reaction.", "target": target}
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
            return _receipt(
                "delivery_failed",
                {
                    **reaction,
                    "status": "delivery_failed",
                    "error": "Telegram did not confirm the reaction.",
                },
                target,
            )
        own = target.get("own_message") is True
        logger.info(
            "Reaction %s delivered on message %s (%s)",
            emoji,
            message_id,
            "own message" if own else (target.get("sender") or {}).get("username") or "unknown sender",
        )
        if own:
            label = f"your own message {message_id}"
        elif target.get("sender"):
            label = f"{_target_label(target)}'s message {message_id}"
        else:
            label = f"message {message_id}"
        excerpt = target.get("excerpt")
        detail = f' ("{excerpt[:120]}")' if excerpt else ""
        return _receipt(
            "delivered",
            {**reaction, "status": "delivered"},
            target,
            note=(
                f"Reaction set on {label}{detail}. Do not deny it or claim a different target. "
                "You may finish without another message."
            ),
        )
    except asyncio.CancelledError:
        result = _receipt(
            "delivery_uncertain",
            {
                **reaction,
                "status": "delivery_uncertain",
                "error": "Reaction delivery was interrupted; do not automatically resend.",
            },
            target,
        )
        context.conversation_data["interrupted_delivery"] = {
            "tool_call_id": context.conversation_data.get("active_tool_call_id"),
            "result": result,
        }
        raise
    except BadRequest as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction failed: %s", message)
        return _receipt("delivery_failed", {**reaction, "status": "delivery_failed", "error": message}, target)
    except NetworkError as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction delivery was uncertain: %s", message)
        return _receipt(
            "delivery_uncertain",
            {
                **reaction,
                "status": "delivery_uncertain",
                "error": "Reaction delivery is uncertain; do not automatically resend.",
            },
            target,
        )
    except TelegramError as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction failed: %s", message)
        return _receipt("delivery_failed", {**reaction, "status": "delivery_failed", "error": message}, target)
    except Exception as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram reaction failed: %s", message)
        return _receipt("delivery_failed", {**reaction, "status": "delivery_failed", "error": message}, target)


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
        target = target_details(context, update, message_id)
        if is_own_message(context, update, message_id):
            # Allowed when meant, but never ambiguous: the receipt says so.
            target["own_message"] = True
        return await deliver_reaction(context, message_id, emoji, target)

    return AgentTool(
        name="react_to_message",
        description=(
            "Set one standard Telegram reaction on the current message, the message this turn replies to, "
            "or a specific past message in this conversation. Reacting to your own message is allowed when you "
            "mean to; check the target first so it is deliberate. "
            "Bots can set one reaction per message. This tool only sets the reaction; if you also want to say "
            "something, include that text in the same response as this tool call. "
            "The receipt names the target's sender and excerpt, and says when the target was your own message. "
            "Do not claim a reaction the receipt does not set, and do not deny one it confirms."
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
                    "enum": ["current", "reply_to"],
                    "default": "current",
                    "description": (
                        "Which message to react to. current is the message that triggered this turn; "
                        "reply_to is the message this turn replies to, which may be your own."
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
