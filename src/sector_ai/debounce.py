"""Restart a turn that has not said anything yet when a newer relevant message arrives.

A turn is *committed* from the moment it starts its first Telegram write or
tool execution. Until then it has no visible effect, so a newer relevant
message (or an edit of the message it answers) can cancel it and answer
everything at once. A committed turn is never cancelled: newer messages queue
behind it as usual.

Turns are tracked per conversation in ``conversation_data["turns"]`` in arrival
order. A turn waiting for the conversation lock is tracked too, so a burst of
messages collapses into one answer to the newest of them. A turn only ever
supersedes turns that arrived before it, and it waits until every earlier turn
has stored its message before it takes the lock, so history keeps arrival
order and an answer always reads what it supersedes.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TURNS = "turns"
SEQUENCE = "turn_sequence"


class Superseded(asyncio.CancelledError):
    """Raised instead of a Telegram write or tool call once a newer message took over."""


@dataclass(eq=False)
class Turn:
    message_ids: set[int]
    sender_id: int | None
    sequence: int = 0
    restarts: int = 0
    #: The turn decided to answer (or was told to).
    engaged: bool = False
    #: The turn started a Telegram write or a tool; it can no longer be cancelled.
    committed: bool = False
    #: A newer message took over; this turn only records its message.
    superseded: bool = False
    done: bool = False
    #: Its message is in history (or it never will be: a duplicate, or it left).
    stored: bool = False
    task: asyncio.Task | None = field(default=None, repr=False)
    #: Older turns this one took over.
    replaced: list["Turn"] = field(default_factory=list, repr=False)
    #: Edits that arrived before the message was stored, by message ID. The
    #: turn stores these versions instead of the originals.
    edits: dict = field(default_factory=dict, repr=False)
    stored_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def live(self) -> bool:
        return not (self.committed or self.superseded or self.done)


def _sender_id(message) -> int | None:
    identifier = getattr(getattr(message, "from_user", None), "id", None)
    if type(identifier) is int:
        return identifier
    identifier = getattr(getattr(message, "sender_chat", None), "id", None)
    return identifier if type(identifier) is int else None


def message_ids(update, photos=None) -> set[int]:
    messages = list(photos or []) or [update.effective_message]
    return {item.message_id for item in messages if type(getattr(item, "message_id", None)) is int}


def try_commit(context) -> bool:
    """Claim the turn's first visible effect; False once a newer message took over.

    Called right before every outgoing Telegram call and tool execution. Commit
    and supersede both run on the event loop without awaiting in between, so
    exactly one of them wins.
    """
    turn = getattr(context, "_turn", None)
    if turn is None:
        return True
    if turn.superseded:
        return False
    turn.committed = True
    return True


def commit_or_stop(context) -> None:
    if not try_commit(context):
        raise Superseded()


def live_turns(context, before: "Turn") -> list[Turn]:
    """Uncommitted turns that arrived before ``before``."""
    return [turn for turn in context.conversation_data.get(TURNS, []) if turn.sequence < before.sequence and turn.live]


async def addressed_to_bot(update, context) -> bool:
    """A reply to the bot, or an @mention while autoreply is on."""
    message = update.effective_message
    reply = getattr(message, "reply_to_message", None)
    if reply and getattr(getattr(reply, "from_user", None), "id", None) == context.bot.id:
        return True
    if context.chat_autoreply_mode:
        username = await context.get_username()
        body = message.text or getattr(message, "caption", None) or ""
        if isinstance(body, str) and re.search(rf"(?<!\w)@{re.escape(username)}(?!\w)", body, re.IGNORECASE):
            return True
    return False


async def _relevant(update, context, targets: list[Turn]) -> bool:
    """Fast path first; otherwise the normal engagement decision, on a history snapshot."""
    if await addressed_to_bot(update, context):
        return True
    sender = _sender_id(update.effective_message)
    if sender is not None and any(turn.engaged and turn.sender_id == sender for turn in targets):
        return True
    from .autoreply import should_respond

    # The message is not in history yet; engagement treats it as the latest.
    return bool(await should_respond(update, context))


def register(update, context, *, photos=None, restarts: int = 0, edited: bool = False) -> Turn | None:
    """Track this message's turn. Synchronous, so the caller can clean it up in ``finally``.

    Returns ``None`` for a duplicate delivery, which must never take over the
    turn it duplicates.
    """
    data = context.conversation_data
    turns = data.setdefault(TURNS, [])
    identifiers = message_ids(update, photos)
    if not edited and context.is_processed(update):
        return None
    # An edit restart re-registers the message of the turn it superseded. Any
    # other overlap is a redelivery, even of a superseded turn that has not
    # stored its message yet.
    if any(turn.message_ids & identifiers for turn in turns if not (edited and turn.superseded)):
        return None
    sequence = data.get(SEQUENCE, 0) + 1
    data[SEQUENCE] = sequence
    turn = Turn(
        message_ids=identifiers,
        sender_id=_sender_id(update.effective_message),
        sequence=sequence,
        restarts=restarts,
    )
    turns.append(turn)
    return turn


async def take_over(update, context, turn: Turn) -> bool:
    """Supersede older unanswered turns if this message is relevant.

    Returns whether it did, in which case the turn answers without deciding
    again.
    """
    limit = context.settings.debounce_max_restarts
    targets = live_turns(context, turn)
    if limit <= 0 or not targets or max(item.restarts for item in targets) >= limit:
        return False
    try:
        relevant = await _relevant(update, context, targets)
    except Exception as error:
        logger.warning("Debounce relevance check failed: %s", error)
        return False
    if not relevant or turn.superseded:
        return False
    # Re-check: a turn may have committed while the decision ran.
    targets = live_turns(context, turn)
    if not targets:
        return False
    supersede(targets)
    turn.replaced.extend(targets)
    turn.restarts = max(turn.restarts, *(item.restarts + 1 for item in targets))
    logger.info(
        "Message %s superseded %d unanswered turn(s); restart %d",
        sorted(turn.message_ids),
        len(targets),
        turn.restarts,
    )
    return True


async def wait_for_earlier(context, turn: Turn) -> None:
    """Store messages in arrival order: wait until every earlier turn stored its own.

    This covers turns this one took over and turns it did not (an earlier
    message still being checked, while this one was judged irrelevant to it).
    Earlier turns never wait on a later one, so this cannot deadlock.
    """
    while True:
        pending = [
            item
            for item in context.conversation_data.get(TURNS, [])
            if item.sequence < turn.sequence and not item.stored
        ]
        if not pending:
            return
        await pending[0].stored_event.wait()


def mark_stored(turn: Turn | None) -> None:
    if turn is not None:
        turn.stored = True
        turn.stored_event.set()


def supersede(turns: list[Turn]) -> None:
    for item in turns:
        item.superseded = True
        if item.task is not None and not item.task.done():
            item.task.cancel()


def leave(context, turn: Turn) -> None:
    turn.done = True
    mark_stored(turn)
    turns = context.conversation_data.get(TURNS)
    if turns is not None and turn in turns:
        turns.remove(turn)


def turn_for_message(context, message_id: int) -> Turn | None:
    """The live turn answering ``message_id``, if any."""
    for turn in context.conversation_data.get(TURNS, []):
        if turn.live and message_id in turn.message_ids:
            return turn
    return None


def unstored_turn_for_message(context, message_id: int) -> Turn | None:
    """A turn, superseded or not, that will still store ``message_id``."""
    for turn in context.conversation_data.get(TURNS, []):
        if not turn.stored and message_id in turn.message_ids:
            return turn
    return None
