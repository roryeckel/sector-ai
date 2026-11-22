"""Keep credentials out of the logs and user-facing errors.

The Telegram token is part of every Telegram URL, and ``httpx`` logs those URLs
at INFO:

    HTTP Request: POST https://api.telegram.org/bot<id>:<token>/getMe "HTTP/1.1 200 OK"

``telegram.Bot`` is no better: its ``__repr__`` renders ``Bot[token=<id>:<token>]``
verbatim. Model backends can echo an ``Authorization`` header in an error body,
and ``SecretStr`` does not protect exception strings. Redaction happens on the
finished string, in a ``Formatter``, rather than by rewriting log records:
tracebacks are assembled inside ``format()``, so a credential in an ``httpx``
error message only exists at that point.

The log level is left alone; the request lines are still useful, only the
credential is removed.
"""

import logging
import re
from collections.abc import Iterable

from .settings import Settings

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# A Telegram token is "<bot id>:<35 characters>". Thirty is a low enough floor to
# catch a pasted URL while staying far away from ordinary prose.
TOKEN_IN_URL = re.compile(r"bot\d+:[A-Za-z0-9_-]{30,}")

# Credentials registered after startup, kept for the process lifetime so a
# rotated key does not make the previous one loggable again.
_REGISTERED_SECRETS: set[str] = set()

CREDENTIAL_HEADER_WORDS = ("auth", "key", "token", "secret", "cookie")


def settings_secrets(settings: Settings | None) -> set[str]:
    """Every credential in ``settings`` that must not reach a log or a reply."""
    if settings is None:
        return set()

    secrets = {
        settings.telegram_bot_token.get_secret_value(),
        settings.openai_api_key.get_secret_value(),
    }
    for name, value in settings.llm_headers.items():
        if any(word in name.lower() for word in CREDENTIAL_HEADER_WORDS):
            if value.lower().startswith("bearer "):
                value = value.split(None, 1)[1]
            secrets.add(value)
    # An explicitly configured key is redacted even when it is short; only
    # empty values are ignored.
    return {secret for secret in secrets if secret}


def register_secrets(settings: Settings | None) -> None:
    """Teach the log formatter about credentials that appear after startup."""
    _REGISTERED_SECRETS.update(settings_secrets(settings))


def redact_secrets(text: str, secrets: Iterable[str]) -> str:
    """Replace token-shaped URLs and any configured secret in ``text``."""
    text = TOKEN_IN_URL.sub("bot<redacted>", text)
    # Longest first, so a short key that prefixes a longer one cannot leave the
    # rest of the longer credential visible.
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def redact_token(text: str, token: str) -> str:
    """Replace the configured token and any token-shaped URL segment in ``text``."""
    return redact_secrets(text, [token])


def redact_for_user(text: str, settings: Settings | None) -> str:
    """Redact configured secrets before showing an error to a chat participant."""
    return redact_secrets(text, settings_secrets(settings) | _REGISTERED_SECRETS)


class TokenRedactingFormatter(logging.Formatter):
    """A formatter that scrubs credentials after formatting the whole record.

    Formatting first matters: ``Formatter.format`` appends the traceback, which
    is where an exception carrying the request URL or an echoed header would
    otherwise escape. Registered secrets are read on every call, so a key that
    rotates while the bot runs is covered without a restart.
    """

    def __init__(self, *args, token: str = "", secrets: Iterable[str] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._secrets = {token, *secrets}

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record), self._secrets | _REGISTERED_SECRETS)


def configure_logging(settings: Settings) -> None:
    """Install the redacting formatter on the root handler(s).

    ``basicConfig`` creates the stderr handler when the process owns logging. An
    embedding host or a test run may already have handlers, so the level and the
    formatter are applied to whatever the root logger has.
    """
    logging.basicConfig(format=LOG_FORMAT, level=settings.log_level)

    root = logging.getLogger()
    root.setLevel(settings.log_level)

    register_secrets(settings)
    formatter = TokenRedactingFormatter(
        LOG_FORMAT,
        token=settings.telegram_bot_token.get_secret_value(),
        secrets=settings_secrets(settings),
    )
    for handler in root.handlers:
        handler.setFormatter(formatter)
