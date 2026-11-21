"""Tests for message routing, settings wiring and handler registration in __main__."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

from sector_ai import __main__ as entrypoint
from sector_ai.__main__ import error_handler, handle_message, start_cmd

EXPECTED_COMMANDS = {
    "start",
    "chat",
    "summarize",
    "poll",
    "topic",
    "emoji",
    "code",
    "svg",
    "html",
    "decide",
    "clear",
    "system",
    "characterize",
    "temperature",
    "models",
    "tokens",
    "autoreply",
}


async def test_start_sends_intro(make_context, make_update):
    context = make_context("unused")
    update = make_update(text="/start")

    await start_cmd(update, context)

    assert update.message.texts("reply_text") == [
        "Hi! I'm a bot that can chat with you. Use /chat to start a conversation."
    ]


async def test_handle_message_records_reply_to_bot(make_context, make_update, monkeypatch):
    handled = AsyncMock()
    monkeypatch.setattr(entrypoint, "handle_chat", handled)
    context = make_context("unused")
    reply_to_bot = SimpleNamespace(from_user=SimpleNamespace(id=context.bot.id), text="earlier bot reply")
    update = make_update(text="hello again", reply_to_message=reply_to_bot)

    await handle_message(update, context)

    assert isinstance(context.chat_message_history[0], AIMessage)
    assert context.chat_message_history[0].content == "earlier bot reply"
    assert isinstance(context.chat_message_history[1], HumanMessage)
    handled.assert_awaited_once()


async def test_handle_message_does_not_duplicate_known_bot_reply(make_context, make_update, monkeypatch):
    monkeypatch.setattr(entrypoint, "handle_chat", AsyncMock())
    context = make_context("unused")
    context.chat_message_history.append(AIMessage(content="already here"))
    reply_to_bot = SimpleNamespace(from_user=SimpleNamespace(id=context.bot.id), text="already here")
    update = make_update(text="hello", reply_to_message=reply_to_bot)

    await handle_message(update, context)

    assert [m.content for m in context.chat_message_history].count("already here") == 1


async def test_handle_message_ignores_replies_to_other_users(make_context, make_update, monkeypatch):
    responded = AsyncMock(return_value=False)
    monkeypatch.setattr(entrypoint, "should_respond", responded)
    monkeypatch.setattr(entrypoint, "handle_chat", AsyncMock())
    context = make_context("unused")
    reply_to_other = SimpleNamespace(from_user=SimpleNamespace(id=999), text="someone else")
    update = make_update(text="hello", reply_to_message=reply_to_other)

    await handle_message(update, context)

    entrypoint.handle_chat.assert_not_awaited()


async def test_handle_message_answers_fresh_messages(make_context, make_update, monkeypatch):
    monkeypatch.setattr(entrypoint, "should_respond", AsyncMock(return_value=True))
    monkeypatch.setattr(entrypoint, "handle_chat", AsyncMock())
    context = make_context("unused")
    update = make_update(text="hello")

    await handle_message(update, context)

    entrypoint.handle_chat.assert_awaited_once()


async def test_handle_message_ignores_stale_messages(make_context, make_update, monkeypatch):
    monkeypatch.setattr(entrypoint, "should_respond", AsyncMock(return_value=True))
    monkeypatch.setattr(entrypoint, "handle_chat", AsyncMock())
    context = make_context("unused")
    update = make_update(text="hello", date=datetime.now(UTC) - timedelta(minutes=5))

    await handle_message(update, context)

    entrypoint.handle_chat.assert_not_awaited()


async def test_handle_message_respects_autoreply_decision(make_context, make_update, monkeypatch):
    monkeypatch.setattr(entrypoint, "should_respond", AsyncMock(return_value=False))
    monkeypatch.setattr(entrypoint, "handle_chat", AsyncMock())
    context = make_context("unused")
    update = make_update(text="hello")

    await handle_message(update, context)

    entrypoint.handle_chat.assert_not_awaited()


async def test_error_handler_reports_to_message(make_context, make_update):
    context = make_context("unused")
    context.error = ValueError("boom")
    update = make_update(text="hello")

    await error_handler(update, context)

    assert update.message.texts("reply_text") == ["boom"]
    assert context.chat_message_history[-1].content == "boom"


async def test_error_handler_reports_to_callback_query(make_context):
    context = make_context("unused")
    context.error = ValueError("boom")
    query = SimpleNamespace(answer=AsyncMock())
    update = SimpleNamespace(message=None, callback_query=query)

    await error_handler(update, context)

    query.answer.assert_awaited_once_with("boom")


async def test_error_handler_swallows_bad_request(make_context, make_update):
    context = make_context("unused")
    context.error = ValueError("boom")
    update = make_update(text="hello", fail_times=1)

    await error_handler(update, context)  # must not raise

    assert context.chat_message_history[-1].content == "boom"


class FakeApplication:
    def __init__(self):
        self.handlers = []
        self.error_handler = None
        self.bot_data = {}
        self.polling = False

    def add_handler(self, handler):
        self.handlers.append(handler)

    def add_error_handler(self, handler):
        self.error_handler = handler

    def run_polling(self):
        self.polling = True


class FakeBuilder:
    def __init__(self, application):
        self.application = application

    def token(self, token):
        self.token_value = token
        return self

    def context_types(self, context_types):
        return self

    def build(self):
        return self.application


def test_main_registers_every_command(tmp_path, monkeypatch):
    application = FakeApplication()
    monkeypatch.setattr(entrypoint, "Application", SimpleNamespace(builder=lambda: FakeBuilder(application)))
    monkeypatch.chdir(tmp_path)  # keep a developer's .env out of the test
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")

    entrypoint.main([])

    commands = set()
    for handler in application.handlers:
        if isinstance(handler, CommandHandler):
            commands |= set(handler.commands)

    assert commands == EXPECTED_COMMANDS
    assert any(isinstance(handler, CallbackQueryHandler) for handler in application.handlers)
    assert len([handler for handler in application.handlers if isinstance(handler, MessageHandler)]) == 2
    assert application.error_handler is error_handler
    assert application.polling is True
    settings = application.bot_data["settings"]
    assert settings.telegram_bot_token.get_secret_value() == "123:abc"


def test_main_reports_missing_token_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SECTOR_AI_TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        entrypoint.main([])

    assert exit_info.value.code == 2
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err


def test_main_reports_malformed_environment_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("SECTOR_AI_OLLAMA_HEADERS", "not-json")

    with pytest.raises(SystemExit) as exit_info:
        entrypoint.main([])

    assert exit_info.value.code == 2
    assert "ollama_headers" in capsys.readouterr().err


def test_main_cli_overrides_environment(tmp_path, monkeypatch):
    application = FakeApplication()
    monkeypatch.setattr(entrypoint, "Application", SimpleNamespace(builder=lambda: FakeBuilder(application)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("SECTOR_AI_DEFAULT_MODEL", "env-model")
    monkeypatch.setenv("SECTOR_AI_ADMIN_USERNAMES", "envadmin,someone")

    entrypoint.main(["--model", "cli-model", "--admin", "cliadmin", "--admin", "other", "--log-level", "debug"])

    settings = application.bot_data["settings"]
    assert settings.default_model == "cli-model"
    assert settings.admin_usernames == ["cliadmin", "other"]
    assert settings.log_level == "DEBUG"


def test_version_flag_prints_version(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        entrypoint.main(["--version"])

    assert exit_info.value.code == 0
    assert entrypoint.__version__ in capsys.readouterr().out


def test_parser_rejects_unknown_log_level():
    parser = entrypoint.build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--log-level", "LOUD"])

    assert exit_info.value.code == 2
