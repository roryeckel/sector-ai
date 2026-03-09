"""Tests for SectorContext core state management."""

from collections import deque
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from sector_ai.sector_context import SectorContext

# --- save_user_message ---


def _make_mock_self(maxlen=15):
    """Create a mock self with a real deque for chat_message_history."""
    mock_self = MagicMock()
    mock_self.chat_message_history = deque(maxlen=maxlen)
    return mock_self


def _make_message(text, username="testuser"):
    msg = MagicMock()
    msg.text = text
    msg.from_user.username = username
    return msg


def test_save_plain_text():
    mock_self = _make_mock_self()
    msg = _make_message("hello world")
    result = SectorContext.save_user_message(mock_self, msg)
    assert result is True
    assert len(mock_self.chat_message_history) == 1
    saved = mock_self.chat_message_history[0]
    assert isinstance(saved, HumanMessage)
    assert saved.content == "testuser: hello world"


def test_save_command_with_arg():
    mock_self = _make_mock_self()
    msg = _make_message("/chat tell me a joke")
    result = SectorContext.save_user_message(mock_self, msg)
    assert result is True
    assert mock_self.chat_message_history[0].content == "testuser: tell me a joke"


def test_save_bare_command():
    mock_self = _make_mock_self()
    msg = _make_message("/clear")
    result = SectorContext.save_user_message(mock_self, msg)
    assert result is False
    assert len(mock_self.chat_message_history) == 0


def test_save_preserves_username():
    mock_self = _make_mock_self()
    msg = _make_message("hi", username="alice")
    SectorContext.save_user_message(mock_self, msg)
    assert mock_self.chat_message_history[0].name == "alice"


def test_save_history_maxlen():
    mock_self = _make_mock_self(maxlen=15)
    for i in range(16):
        msg = _make_message(f"message {i}")
        SectorContext.save_user_message(mock_self, msg)
    assert len(mock_self.chat_message_history) == 15
    # First message evicted, oldest is message 1
    assert "message 1" in mock_self.chat_message_history[0].content


# --- message_exists ---


def test_exists_matching():
    mock_self = _make_mock_self()
    mock_self.chat_message_history.append(HumanMessage(content="x"))
    # Use HumanMessage for the check so .type matches ("human" == "human")
    check = HumanMessage(content="x")
    result = SectorContext.message_exists(mock_self, check)
    assert result is True


def test_exists_no_match():
    mock_self = _make_mock_self()
    check = HumanMessage(content="anything")
    result = SectorContext.message_exists(mock_self, check)
    assert result is False


# --- config_ollama_url ---


def test_url_present():
    mock_self = MagicMock()
    mock_self.bot_config = {"ollama": {"url": "http://custom:1234"}}
    result = SectorContext.config_ollama_url.fget(mock_self)
    assert result == "http://custom:1234"


def test_url_fallback():
    mock_self = MagicMock()
    mock_self.bot_config = {"ollama": {}}
    result = SectorContext.config_ollama_url.fget(mock_self)
    assert result == "http://localhost:11434"


# --- get_model_details ---


def test_get_model_details_strips_fields():
    mock_self = MagicMock()
    mock_self.get_model.return_value = "test-model"
    mock_self.config_ollama_url = "http://localhost:11434"
    mock_self.config_ollama_headers = {}

    fake_response = MagicMock()
    fake_response.ok = True
    fake_response.json.return_value = {
        "license": "MIT",
        "modelfile": "FROM ...",
        "parameters": "num_ctx 2048",
        "template": "{{ .System }}",
        "details": {"family": "llama", "parameter_size": "7B"},
        "model_info": {"general.architecture": "llama"},
    }

    with patch("sector_ai.sector_context.requests.post", return_value=fake_response):
        result = SectorContext.get_model_details(mock_self)

    assert "details" in result
    assert "model_info" in result
    assert "license" not in result
    assert "modelfile" not in result
    assert "parameters" not in result
    assert "template" not in result
