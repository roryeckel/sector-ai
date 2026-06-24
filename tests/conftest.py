import json
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest  # noqa: F401 (used by pytest fixture decorator)

from sector_ai.sector_context import SectorContext


@pytest.fixture
def sample_config():
    config_path = Path(__file__).resolve().parent.parent / "default_config.json"
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def mock_context(sample_config):
    ctx = MagicMock(spec=SectorContext)

    # bot_data / chat_data backing stores
    ctx.bot_data = {"config": sample_config}
    chat_data = {"message_history": deque(maxlen=15), "autoreply_mode": True}
    ctx.chat_data = chat_data

    # Properties that delegate to config
    type(ctx).bot_config = PropertyMock(return_value=sample_config)
    type(ctx).config_admin_usernames = PropertyMock(return_value=sample_config["admin_usernames"])
    type(ctx).config_default_system_prompt = PropertyMock(return_value=sample_config["system_prompts"]["default"])
    type(ctx).config_decide_system_prompt = PropertyMock(return_value=sample_config["system_prompts"]["decide"])
    type(ctx).config_ollama_url = PropertyMock(return_value=sample_config["ollama"]["url"])
    type(ctx).config_ollama_headers = PropertyMock(return_value=sample_config["ollama"].get("headers", {}))
    type(ctx).config_basic_poll_system_prompt = PropertyMock(return_value=sample_config["system_prompts"]["basic_poll"])
    type(ctx).config_mood_system_prompt = PropertyMock(return_value=sample_config["system_prompts"]["mood"])
    type(ctx).config_roast_system_prompt = PropertyMock(return_value=sample_config["system_prompts"]["roast"])
    type(ctx).config_hype_system_prompt = PropertyMock(return_value=sample_config["system_prompts"]["hype"])
    type(ctx).config_quiz_system_prompt = PropertyMock(return_value=sample_config["system_prompts"]["quiz"])

    # Chat properties backed by chat_data
    type(ctx).chat_message_history = PropertyMock(return_value=chat_data["message_history"])
    type(ctx).chat_autoreply_mode = PropertyMock(return_value=True)
    # Don't use PropertyMock for chat_system_prompt - let MagicMock handle
    # attribute assignment naturally so tests can verify setter calls
    ctx.chat_system_prompt = sample_config["system_prompts"]["default"]

    # Async methods
    ctx.get_system_template_dict = AsyncMock(return_value={"MAX_LEN": 4096, "USERNAME": "testbot"})

    # bot_ollama
    ctx.bot_ollama = MagicMock(model="test-model")

    return ctx


@pytest.fixture
def mock_update():
    def _factory(text="/cmd", username="testuser"):
        update = MagicMock()
        update.message.text = text
        update.effective_message.text = text
        update.message.from_user.username = username
        update.effective_user.username = username
        update.message.reply_text = AsyncMock()
        update.message.reply_poll = AsyncMock()
        return update

    return _factory
