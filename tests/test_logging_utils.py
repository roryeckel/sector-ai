"""Tests for keeping the bot token out of log output."""

import io
import logging
import sys

import pytest

from sector_ai.logging_utils import (
    LOG_FORMAT,
    TokenRedactingFormatter,
    configure_logging,
    redact_for_user,
    redact_secrets,
    redact_token,
    register_secrets,
)
from sector_ai.settings import Settings

TOKEN = "7427439607:AAHfake-token-value-for-tests-12345678"
OTHER_TOKEN = "8000000000:BBHanother-token-value-for-tests-87654321"


def make_record(msg, args=(), exc_info=None):
    return logging.LogRecord("httpx", logging.INFO, __file__, 1, msg, args, exc_info)


@pytest.fixture
def isolated_root():
    """Give a test the root logger to itself, then put it back."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers = []
    try:
        yield root
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


# --- redact_token ---


def test_redacts_the_token_in_a_request_url():
    text = f"HTTP Request: POST https://api.telegram.org/bot{TOKEN}/getMe"
    assert redact_token(text, TOKEN) == "HTTP Request: POST https://api.telegram.org/bot<redacted>/getMe"


def test_redacts_a_percent_encoded_token_in_a_file_download_url():
    # httpx logs file downloads with the colon encoded, which slipped past the
    # plain pattern and leaked a real token into a log.
    encoded = TOKEN.replace(":", "%3A")
    text = f'HTTP Request: GET https://api.telegram.org/file/bot{encoded}/photos/file_1.jpg "HTTP/1.1 200 OK"'

    assert redact_token(text, TOKEN) == (
        'HTTP Request: GET https://api.telegram.org/file/bot<redacted>/photos/file_1.jpg "HTTP/1.1 200 OK"'
    )
    assert redact_token(f"bot{OTHER_TOKEN.replace(':', '%3a')}/x", TOKEN) == "bot<redacted>/x"


def test_redacts_a_percent_encoded_configured_secret():
    assert redact_secrets("key=a%2Fb%3Ac", ["a/b:c"]) == "key=<redacted>"


def test_redacts_the_bot_repr():
    assert redact_token(f"Bot[token={TOKEN}]", TOKEN) == "Bot[token=<redacted>]"


def test_redacts_a_token_that_is_not_the_configured_one():
    text = f"https://api.telegram.org/bot{OTHER_TOKEN}/getUpdates"
    assert redact_token(text, TOKEN) == "https://api.telegram.org/bot<redacted>/getUpdates"


def test_leaves_text_without_a_token_alone():
    text = "Application started"
    assert redact_token(text, TOKEN) == text


def test_empty_token_is_safe():
    assert redact_token("nothing to see", "") == "nothing to see"


# --- formatter ---


def test_formatter_redacts_lazy_format_arguments():
    # httpx logs 'HTTP Request: %s %s "%s" %d' and passes the URL as an argument.
    record = make_record(
        'HTTP Request: %s %s "%s" %d',
        ("POST", f"https://api.telegram.org/bot{TOKEN}/getMe", "HTTP/1.1", 200),
    )

    output = TokenRedactingFormatter(LOG_FORMAT, token=TOKEN).format(record)

    assert TOKEN not in output
    assert "bot<redacted>/getMe" in output
    assert "HTTP/1.1" in output


def test_formatter_redacts_a_traceback():
    # httpx raises with the URL in the message, and only the formatter sees that text.
    try:
        raise RuntimeError(f"Client error for url 'https://api.telegram.org/bot{TOKEN}/getMe'")
    except RuntimeError:
        exc_info = sys.exc_info()

    output = TokenRedactingFormatter(LOG_FORMAT, token=TOKEN).format(make_record("request failed", exc_info=exc_info))

    assert TOKEN not in output
    assert "bot<redacted>/getMe" in output
    assert "RuntimeError" in output


def test_formatter_output_is_identical_without_a_token():
    plain = logging.Formatter(LOG_FORMAT)
    redacting = TokenRedactingFormatter(LOG_FORMAT, token=TOKEN)

    record = make_record("Decision: True")
    assert redacting.format(record) == plain.format(record)


def test_formatter_redacts_configured_api_secrets():
    api_key = "sk-live-abcdefghijklmnop"
    record = make_record("401: %s", (f"Authorization: Bearer {api_key}",))

    output = TokenRedactingFormatter(LOG_FORMAT, token=TOKEN, secrets=[api_key]).format(record)

    assert api_key not in output
    assert "Authorization: Bearer <redacted>" in output


def test_redact_for_user_removes_api_keys_and_bearer_headers():
    api_key = "sk-live-abcdefghijklmnop"
    settings = Settings(
        telegram_bot_token=TOKEN,
        openai_api_key=api_key,
        llm_headers={"Authorization": "Bearer header-secret-value"},
        _env_file=None,
    )

    text = redact_for_user(f"{api_key} and Bearer header-secret-value", settings)

    assert api_key not in text
    assert "header-secret-value" not in text
    assert text.count("<redacted>") == 2


def test_short_and_cookie_credentials_are_redacted():
    settings = Settings(
        telegram_bot_token=TOKEN,
        openai_api_key="abc",
        llm_headers={"Cookie": "session=shh"},
        _env_file=None,
    )

    text = redact_for_user("abc and session=shh", settings)

    assert "abc" not in text
    assert "session=shh" not in text


def test_overlapping_secrets_redact_the_longest_first():
    text = "sk-live-credential-1"

    assert redact_secrets(text, ["sk", "sk-live-credential-1"]) == "<redacted>"


def test_in_place_credential_rotation_is_registered(make_context, make_settings):
    from unittest.mock import patch

    from pydantic import SecretStr

    context = make_context("unused", settings=make_settings())
    rotated = "sk-in-place-rotated-key"
    context.settings.openai_api_key = SecretStr(rotated)

    with patch("sector_ai.sector_context.ChatOllama"):
        context.load_model()

    output = TokenRedactingFormatter(LOG_FORMAT).format(make_record(rotated))

    assert rotated not in output


def test_registered_secrets_cover_a_rotated_key():
    rotated = "sk-rotated-abcdefghijkl"
    register_secrets(Settings(telegram_bot_token=TOKEN, openai_api_key=rotated, _env_file=None))

    output = TokenRedactingFormatter(LOG_FORMAT, token=TOKEN).format(make_record(rotated))

    assert rotated not in output
    assert "<redacted>" in output


# --- configure_logging ---


def test_configure_logging_redacts_real_records(isolated_root):
    stream = io.StringIO()
    isolated_root.addHandler(logging.StreamHandler(stream))
    settings = Settings(telegram_bot_token=TOKEN, log_level="INFO", _env_file=None)

    configure_logging(settings)

    assert isolated_root.level == logging.INFO
    assert isinstance(isolated_root.handlers[0].formatter, TokenRedactingFormatter)

    logging.getLogger("httpx").info(
        'HTTP Request: %s %s "%s" %d',
        "POST",
        f"https://api.telegram.org/bot{TOKEN}/getMe",
        "HTTP/1.1",
        200,
    )

    output = stream.getvalue()
    assert TOKEN not in output
    assert "bot<redacted>/getMe" in output
    assert "HTTP Request: POST" in output
