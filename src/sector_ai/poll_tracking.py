"""Bounded, process-local routing for polls this bot actually delivered."""

from .chat_metadata import text_value

MAX_TRACKED_POLLS = 512

#: Telegram allows at most ten answers in a regular poll.
MAX_POLL_OPTIONS = 10


def poll_option_snapshot(poll) -> list[dict]:
    """Keep option text so a later vote update can name what was chosen."""
    options = []
    for option in (getattr(poll, "options", None) or [])[:MAX_POLL_OPTIONS]:
        text = text_value(getattr(option, "text", None), 100)
        if text is None:
            continue
        entry = {"text": text}
        persistent_id = text_value(getattr(option, "persistent_id", None), 100)
        if persistent_id:
            entry["persistent_id"] = persistent_id
        options.append(entry)
    return options


def track_delivered_poll(sent, update, context) -> None:
    poll = getattr(sent, "poll", None)
    poll_id = getattr(poll, "id", None)
    chat_id = getattr(sent, "chat_id", None)
    if type(chat_id) is not int:
        chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
    if not isinstance(poll_id, str) or type(chat_id) is not int:
        return
    registry = context.bot_data.setdefault("tracked_polls", {})
    registry[poll_id] = {
        "chat_id": chat_id,
        "thread_id": context._message_thread_id,
        "message_id": getattr(sent, "message_id", None),
        "question": text_value(getattr(poll, "question", None), 300),
        "options": poll_option_snapshot(poll),
        "is_anonymous": bool(getattr(poll, "is_anonymous", False)),
        "allows_multiple_answers": bool(getattr(poll, "allows_multiple_answers", False)),
    }
    while len(registry) > MAX_TRACKED_POLLS:
        registry.pop(next(iter(registry)))


def poll_vote_labels(origin: dict, answer) -> list[str]:
    """Map a vote update to option text, preferring persistent IDs."""
    options = origin.get("options") or []
    positions = list(getattr(answer, "option_ids", None) or [])
    persisted = list(getattr(answer, "option_persistent_ids", None) or [])
    labels = []
    for index, position in enumerate(positions):
        label = None
        if index < len(persisted):
            persistent_id = persisted[index]
            label = next(
                (option.get("text") for option in options if option.get("persistent_id") == persistent_id),
                None,
            )
        if label is None and type(position) is int and 0 <= position < len(options):
            label = options[position].get("text")
        if label and label not in labels:
            labels.append(label)
    return labels
