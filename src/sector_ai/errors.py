"""Errors this bot raises on purpose to explain a refusal to the user."""


class UserFacingError(ValueError):
    """A refusal written for the chat participant, safe to show as is.

    Anything else that fails a turn is logged and shown as a reference, so a
    backend URL, hostname or file path in a library's error stays out of chat.
    """
