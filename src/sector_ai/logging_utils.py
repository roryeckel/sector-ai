"""Keep the bot token out of the logs.

The token is part of every Telegram URL, and ``httpx`` logs those URLs at INFO:

    HTTP Request: POST https://api.telegram.org/bot<id>:<token>/getMe "HTTP/1.1 200 OK"

``telegram.Bot`` is no better: its ``__repr__`` renders ``Bot[token=<id>:<token>]``
verbatim. Both end up in journald on the deployed bot.

Redaction happens on the finished string, in a ``Formatter``, rather than by
rewriting log records: tracebacks are assembled inside ``format()``, so the URL
in an ``httpx`` error message only exists at that point. The log level is left
alone; the request lines are still useful, only the credential is removed.
"""

import logging
import re

from .settings import Settings

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# A Telegram token is "<bot id>:<35 characters>". Thirty is a low enough floor to
# catch a pasted URL while staying far away from ordinary prose.
TOKEN_IN_URL = re.compile(r"bot\d+:[A-Za-z0-9_-]{30,}")


def redact_token(text: str, token: str) -> str:
    """Replace the configured token and any token-shaped URL segment in ``text``."""
    text = TOKEN_IN_URL.sub("bot<redacted>", text)
    if token:
        text = text.replace(token, "<redacted>")
    return text


class TokenRedactingFormatter(logging.Formatter):
    """A formatter that scrubs the bot token after formatting the whole record.

    Formatting first matters: ``Formatter.format`` appends the traceback, which
    is where an exception carrying the request URL would otherwise escape.
    """

    def __init__(self, *args, token: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._token = token

    def format(self, record: logging.LogRecord) -> str:
        return redact_token(super().format(record), self._token)


def configure_logging(settings: Settings) -> None:
    """Install the redacting formatter on the root handler(s).

    ``basicConfig`` creates the stderr handler when the process owns logging. An
    embedding host or a test run may already have handlers, so the level and the
    formatter are applied to whatever the root logger has.
    """
    logging.basicConfig(format=LOG_FORMAT, level=settings.log_level)

    root = logging.getLogger()
    root.setLevel(settings.log_level)

    formatter = TokenRedactingFormatter(LOG_FORMAT, token=settings.telegram_bot_token.get_secret_value())
    for handler in root.handlers:
        handler.setFormatter(formatter)
