"""Observe selected service messages without starting autonomous agent turns."""

import json
import logging
from hashlib import sha256

from langchain_core.messages import HumanMessage
from telegram.ext import filters

from .chat_metadata import actor_metadata, message_metadata, text_value, utc_now, utc_timestamp
from .debounce import supersede, turn_for_message
from .history import scoped_control
from .poll_tracking import poll_vote_labels
from .sector_context import SectorContext

logger = logging.getLogger(__name__)

MESSAGE_EDIT_FILTER = filters.UpdateType.EDITED_MESSAGE | filters.UpdateType.EDITED_CHANNEL_POST

CHAT_EVENT_FILTER = (
    filters.StatusUpdate.NEW_CHAT_MEMBERS
    | filters.StatusUpdate.LEFT_CHAT_MEMBER
    | filters.StatusUpdate.NEW_CHAT_TITLE
    | filters.StatusUpdate.PINNED_MESSAGE
    | filters.StatusUpdate.FORUM_TOPIC_CREATED
    | filters.StatusUpdate.FORUM_TOPIC_EDITED
    | filters.StatusUpdate.FORUM_TOPIC_CLOSED
    | filters.StatusUpdate.FORUM_TOPIC_REOPENED
)


@scoped_control
async def handle_chat_event(update, context):
    message = update.effective_message
    event = None
    members = getattr(message, "new_chat_members", None)
    left = getattr(message, "left_chat_member", None)
    title = text_value(getattr(message, "new_chat_title", None))
    pinned = getattr(message, "pinned_message", None)
    created = getattr(message, "forum_topic_created", None)
    edited = getattr(message, "forum_topic_edited", None)
    if isinstance(members, (tuple, list)) and members:
        event = {"event": "members_joined", "members": [actor_metadata(user) for user in members[:20]]}
        if len(members) > 20:
            event["omitted_members"] = len(members) - 20
    elif left:
        event = {"event": "member_left", "member": actor_metadata(left)}
    elif title:
        event = {"event": "chat_title_changed", "title": title}
    elif pinned:
        target = message_metadata(pinned, include_reply=False)
        excerpt = text_value(getattr(pinned, "text", None), 600) or text_value(getattr(pinned, "caption", None), 600)
        if excerpt:
            target["excerpt"] = excerpt
        event = {"event": "message_pinned", "message": target}
    elif created:
        event = {"event": "topic_created", "name": text_value(created.name)}
    elif edited:
        event = {"event": "topic_edited"}
        name = text_value(getattr(edited, "name", None))
        if name:
            event["name"] = name
    elif getattr(message, "forum_topic_closed", None):
        event = {"event": "topic_closed"}
    elif getattr(message, "forum_topic_reopened", None):
        event = {"event": "topic_reopened"}
    if event is None or not context.accept_update(update):
        return
    if event["event"] == "chat_title_changed":
        context.chat_data.setdefault("chat_metadata", {})["title"] = event["title"]
    if event["event"].startswith("topic_"):
        topic = context.conversation_data.setdefault("topic_metadata", {})
        if event.get("name"):
            topic["name"] = event["name"]
        if event["event"] in {"topic_closed", "topic_reopened", "topic_created"}:
            topic["closed"] = event["event"] == "topic_closed"
    metadata = {"kind": "chat_event", **message_metadata(message)}
    context.chat_message_history.append(
        HumanMessage(
            name="telegram_event",
            content="Telegram service event (observed data): " + json.dumps(event, ensure_ascii=False),
            additional_kwargs={"telegram": metadata},
        )
    )
    context._enforce_history_cap()


def observed_event(event: dict, metadata: dict):
    return HumanMessage(
        name="telegram_event",
        content="Telegram event (observed data): " + json.dumps(event, ensure_ascii=False),
        additional_kwargs={"telegram": {"kind": "chat_event", **metadata}},
    )


async def handle_edited_message(update, context):
    """Apply an edit; if it changes a message the bot is still silently working on, restart that turn."""
    context.bind_update(update)
    message_id = getattr(update.effective_message, "message_id", None)
    limit = context.settings.debounce_max_restarts
    turn = turn_for_message(context, message_id) if type(message_id) is int and limit > 0 else None
    # Only a running turn has ingested its message, so only then does the edit
    # replace text the restarted turn will read.
    if turn is None or turn.task is None or turn.restarts >= limit:
        await handle_message_edit(update, context)
        return
    engage = True if turn.engaged else None
    supersede([turn])
    logger.info("Edit of message %s restarted its unanswered turn", message_id)
    await handle_message_edit(update, context)
    from .agent import run_agent

    await run_agent(
        update,
        context,
        ingest=False,
        debounce=True,
        engage=engage,
        edited=True,
        restarts=turn.restarts + 1,
    )


@scoped_control
async def handle_message_edit(update, context):
    """Update retained text and append a correction, never rerun the command."""
    message = update.effective_message
    message_id = getattr(message, "message_id", None)
    if type(message_id) is not int:
        return
    metadata = message_metadata(message)
    edited_at = metadata.get("edited_at") or utc_now()
    text = text_value(getattr(message, "text", None), 4096) or text_value(getattr(message, "caption", None), 4096)
    digest = sha256((text or "[non-text edit]").encode()).hexdigest()
    versions = context.conversation_data.setdefault("message_edits", {})
    previous = versions.get(message_id)
    update_id = getattr(update, "update_id", None)
    sequence = update_id if type(update_id) is int else -1
    if previous:
        if edited_at < previous["at"] or (edited_at == previous["at"] and sequence < previous["sequence"]):
            return
        if edited_at == previous["at"] and digest == previous["digest"]:
            return
    if not context.accept_update(update):
        return
    event = {
        "event": "message_edited",
        "message_id": message_id,
        "edited_at": edited_at,
        "prior_actions_reexecuted": False,
    }
    replacement = context.make_user_message(message, include_command=True)
    if replacement is None:
        sender = actor_metadata(getattr(message, "from_user", None))
        replacement = HumanMessage(
            name=sender.get("username") or f"user_{sender.get('id', 'unknown')}",
            content="[message edited; no text or caption is available]",
            additional_kwargs={"telegram": metadata},
        )
    found = False
    for index, item in enumerate(context.chat_message_history):
        original = item.additional_kwargs.get("telegram", {})
        if (
            isinstance(item, HumanMessage)
            and original.get("kind") != "chat_event"
            and original.get("message_id") == message_id
        ):
            # An edit arriving late must not replace a newer version.
            if original.get("edited_at", "") > edited_at:
                return
            event["previous_excerpt"] = item.content[:600]
            images = item.additional_kwargs.get("images")
            if images:
                # A caption edit keeps the photos (and any album) already stored.
                replacement = replacement.model_copy(
                    update={"additional_kwargs": {**replacement.additional_kwargs, "images": images}}
                )
            context.chat_message_history[index] = replacement
            found = True
            break
    event["original_in_recent_history"] = found
    event["new_excerpt" if found else "new_content"] = (text or "[non-text message edited]")[: 600 if found else 4096]
    versions[message_id] = {"at": edited_at, "sequence": sequence, "digest": digest}
    while len(versions) > 512:
        versions.pop(next(iter(versions)))
    context.chat_message_history.append(observed_event(event, metadata))
    context._enforce_history_cap()


def poll_voter(answer) -> dict:
    """Identify the voter, whether a user or an anonymous voter chat."""
    user = getattr(answer, "user", None)
    if user is not None:
        return actor_metadata(user)
    # Bot API 9.6 renamed this field; older payloads still arrive as "voter".
    raw = (getattr(answer, "api_kwargs", None) or {}).get("voter")
    if isinstance(raw, dict):
        result = {}
        if type(raw.get("id")) is int:
            result["id"] = raw["id"]
        for key, attribute in (("username", "username"), ("display_name", "first_name")):
            value = text_value(raw.get(attribute), 256)
            if value:
                result[key] = value
        if type(raw.get("is_bot")) is bool:
            result["is_bot"] = raw["is_bot"]
        return result
    chat = getattr(answer, "voter_chat", None)
    if chat is None:
        return {}
    result = {}
    identifier = getattr(chat, "id", None)
    if type(identifier) is int:
        result["id"] = identifier
    for key, attribute in (("username", "username"), ("display_name", "title")):
        value = text_value(getattr(chat, attribute, None))
        if value:
            result[key] = value
    return result


async def handle_poll_answer(update, context):
    """Record votes only for polls this bot sent; Telegram sends no read API."""
    answer = getattr(update, "poll_answer", None)
    if answer is None:
        return
    registry = context.bot_data.get("tracked_polls", {})
    origin = registry.get(answer.poll_id)
    if origin is None:
        return
    target = SectorContext(context.application, chat_id=origin["chat_id"])
    target.bind_topic(origin.get("thread_id") or 0)
    async with target.conversation_lock:
        if registry.get(answer.poll_id) is not origin:
            return  # /clear or bounded registry eviction while waiting.
        if not target.accept_update(update):
            return
        labels = poll_vote_labels(origin, answer)
        event = {
            "event": "poll_vote" if labels else "poll_vote_retracted",
            "poll_id": answer.poll_id,
            "question": origin.get("question"),
            "voter": poll_voter(answer),
            "options": labels,
        }
        target.chat_message_history.append(
            observed_event(
                event,
                {
                    "at": utc_now(),
                    "source": "telegram_poll_update",
                },
            )
        )
        target._enforce_history_cap()


async def handle_poll_result(update, context):
    """Poll updates have no chat ID: only trust an origin registered at send time."""
    poll = getattr(update, "poll", None)
    if poll is None or not poll.is_closed:
        return
    registry = context.bot_data.get("tracked_polls", {})
    origin = registry.get(poll.id)
    if origin is None:
        return
    target = SectorContext(context.application, chat_id=origin["chat_id"])
    target.bind_topic(origin["thread_id"])
    async with target.conversation_lock:
        if registry.get(poll.id) is not origin:
            return  # /clear or bounded registry eviction while waiting.
        event = {
            "event": "poll_closed",
            "poll_id": poll.id,
            "message_id": origin["message_id"],
            "question": poll.question,
            "total_voter_count": poll.total_voter_count,
            "is_anonymous": poll.is_anonymous,
            "allows_multiple_answers": poll.allows_multiple_answers,
            "options": [{"text": option.text, "voter_count": option.voter_count} for option in poll.options],
        }
        signature = sha256(json.dumps(event, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if origin.get("final_signature") == signature or not target.accept_update(update):
            return
        target.chat_message_history.append(
            observed_event(
                event,
                {
                    "at": utc_now(),
                    "source": "telegram_poll_update",
                },
            )
        )
        origin["final_signature"] = signature
        target._enforce_history_cap()


def membership_state(member):
    raw = member.to_dict()
    result = {
        "status": member.status,
        "permissions": {
            key: value
            for key, value in raw.items()
            if type(value) is bool and (key.startswith("can_") or key in {"is_member", "is_anonymous"})
        },
    }
    until = getattr(member, "until_date", None)
    timestamp = utc_timestamp(until)
    if timestamp:
        result["until_date"] = "indefinite" if until.timestamp() == 0 else timestamp
    return result


@scoped_control
async def handle_bot_membership(update, context):
    change = getattr(update, "my_chat_member", None)
    if change is None or change.new_chat_member.user.id != context.bot.id:
        return
    old, new = membership_state(change.old_chat_member), membership_state(change.new_chat_member)
    if old == new:
        return
    at = utc_timestamp(change.date) or utc_now()
    current = context.chat_data.get("bot_membership", {})
    update_id = getattr(update, "update_id", None)
    sequence = update_id if type(update_id) is int else -1
    if (at, sequence) < (current.get("changed_at", ""), current.get("sequence", -1)):
        return
    if not context.accept_update(update):
        return
    names = old["permissions"].keys() | new["permissions"].keys()
    event = {
        "event": "bot_membership_changed",
        "old_status": old["status"],
        "new_status": new["status"],
        "actor": actor_metadata(change.from_user),
        "permissions_changed": {
            name: {"old": old["permissions"].get(name), "new": new["permissions"].get(name)}
            for name in sorted(names)
            if old["permissions"].get(name) != new["permissions"].get(name)
        },
    }
    if old.get("until_date") != new.get("until_date"):
        event["restriction_until_changed"] = {"old": old.get("until_date"), "new": new.get("until_date")}
    context.chat_data["bot_membership"] = {**new, "changed_at": at, "sequence": sequence}
    context.record_shared_chat_event(observed_event(event, {"at": at, "source": "my_chat_member"}))
