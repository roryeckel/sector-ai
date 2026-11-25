"""Bounded, text-only engagement context. No tools or history mutation."""

from copy import deepcopy

from langchain_core.messages import HumanMessage, SystemMessage

from .chat_metadata import metadata_pair, render_metadata, render_prompt_message
from .history import plain_history
from .message_text import message_text

LATEST_PREFIX = "Latest incoming message (decide about this):\n"


def clip_message(message, budget: int, estimate):
    text = message_text(message)
    if estimate([message]) <= budget:
        return message
    suffix = "\n[truncated for engagement]"
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = message.model_copy(update={"content": text[:middle] + suffix})
        if estimate([candidate]) <= budget:
            low = middle
        else:
            high = middle - 1
    return message.model_copy(update={"content": text[:low] + suffix})


def without_metadata(message):
    kwargs = dict(message.additional_kwargs)
    kwargs.pop("telegram", None)
    kwargs.pop("telegram_rendered", None)
    return message.model_copy(update={"additional_kwargs": kwargs})


def with_metadata(message, metadata):
    kwargs = dict(message.additional_kwargs)
    kwargs.pop("telegram_rendered", None)
    if metadata is None:
        kwargs.pop("telegram", None)
        return message.model_copy(update={"additional_kwargs": kwargs})
    kwargs["telegram"] = metadata
    return render_metadata(message.model_copy(update={"additional_kwargs": kwargs}))


def clip_metadata(value, limit: int):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {key: clip_metadata(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [clip_metadata(item, limit) for item in value]
    return value


def metadata_variants(metadata):
    """Richest to smallest envelopes that still keep the latest body intact."""
    if not isinstance(metadata, dict) or not metadata:
        return []
    variants = [deepcopy(metadata)]
    reply = metadata.get("reply_to")
    if isinstance(reply, dict):
        without_excerpt = deepcopy(metadata)
        without_excerpt["reply_to"].pop("excerpt", None)
        variants.append(without_excerpt)
        without_reply = deepcopy(metadata)
        without_reply.pop("reply_to", None)
        variants.append(without_reply)
    for limit in (160, 48):
        variants.append(clip_metadata(metadata, limit))
        without_reply = clip_metadata(metadata, limit)
        without_reply.pop("reply_to", None)
        variants.append(without_reply)
    minimal = {key: metadata[key] for key in ("message_id", "sent_at", "edited_at") if key in metadata}
    sender = metadata.get("sender")
    if isinstance(sender, dict):
        minimal["sender"] = {key: sender[key] for key in ("id", "is_bot") if key in sender}
    variants.append(minimal)
    return variants


def latest_message(message, text: str, budget: int, estimate):
    """Reserve raw incoming text first; sender identity stays optional metadata."""
    body = without_metadata(message)
    body = body.model_copy(update={"content": LATEST_PREFIX + text})
    if estimate([body]) > budget:
        body = clip_message(body, budget, estimate)
    for metadata in metadata_variants(message.additional_kwargs.get("telegram")):
        rendered = with_metadata(body, metadata)
        if estimate([rendered]) <= budget:
            return rendered
    return body


async def engagement_messages(update, context, instructions: str):
    estimate = context._estimate_token_count
    budget = min(context.settings.autoreply_context_tokens, int(await context.get_context_budget() * 0.75))
    messages = [SystemMessage(content=instructions)]
    static, dynamic = metadata_pair(context)
    if static is not None:
        messages.append(static)
    tail = [dynamic] if dynamic is not None else []
    remaining = budget - estimate([*messages, *tail])
    if remaining < 64:
        raise ValueError("The engagement instructions and metadata do not fit the context budget.")
    latest = context.make_user_message(update.effective_message)
    if latest is None:
        raise ValueError("There is no incoming message to decide about.")
    incoming = update.effective_message.text or getattr(update.effective_message, "caption", None) or ""
    if not isinstance(incoming, str):
        incoming = ""
    latest = latest_message(latest, incoming, max(48, remaining // 2), estimate)
    remaining -= estimate([latest])
    summary = []
    if context.context_summary and remaining > 64:
        item = HumanMessage(
            name="conversation_summary", content="Earlier conversation summary (data):\n" + context.context_summary
        )
        item = clip_message(item, min(remaining // 3, 256), estimate)
        summary.append(item)
        remaining -= estimate([item])
    history = list(context.chat_message_history)
    # The graph has already ingested the incoming message. Do not include it a
    # second time, but also support standalone should_respond calls.
    current = context.make_user_message(update.effective_message)
    if history and history[-1].type == "human":
        latest_id = current.additional_kwargs.get("telegram", {}).get("message_id")
        previous_id = history[-1].additional_kwargs.get("telegram", {}).get("message_id")
        # Matching IDs suffice: an album's ingested entry carries every photo,
        # so its content differs from the single update being decided about.
        if (latest_id is not None and latest_id == previous_id) or (
            latest_id is None
            and previous_id is None
            and history[-1].content == current.content
            and current.additional_kwargs.get("telegram") == history[-1].additional_kwargs.get("telegram")
        ):
            history.pop()
    recent = []
    count = context.settings.autoreply_history_messages
    candidates = plain_history([*history, *context.shared_chat_events()][-count:]) if count else []
    for item in reversed(candidates):
        if remaining < 32:
            break
        item = clip_message(render_prompt_message(item), min(remaining, 320), estimate)
        recent.append(item)
        remaining -= estimate([item])
    result = [*messages, *summary, *reversed(recent), *tail, latest]
    # Per-message estimates round independently. Account for separators and
    # rounding again over the actual complete prompt before any model call.
    while recent and estimate(result) > budget:
        recent.pop()
        result = [*messages, *summary, *reversed(recent), *tail, latest]
    if estimate(result) > budget:
        raise ValueError("The engagement prompt exceeds its context budget.")
    return result
