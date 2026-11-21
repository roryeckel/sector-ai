"""Tests for the environment-driven settings.

``_env_file=None`` is passed wherever a test is not specifically about
``.env`` files so a developer's local configuration cannot leak in.
"""

import pytest
from pydantic import ValidationError

from sector_ai.prompts import DEFAULT_SYSTEM_PROMPT, SVG_SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
from sector_ai.settings import Settings

TOKEN = "123:test-token"


def env_only(**env):
    """Build settings from the environment alone (plus the given token)."""
    return Settings(_env_file=None, telegram_bot_token=env.pop("telegram_bot_token", TOKEN), **env)


def test_defaults_match_the_shipped_configuration(make_settings):
    settings = make_settings()

    assert settings.default_model == "llama3.2-vision:11b-instruct-fp16"
    assert settings.vision_model == "llama3.2-vision:11b-instruct-fp16"
    assert settings.default_svg_size == 250
    assert settings.admin_usernames == ["ex0dus"]
    assert settings.streaming_cursor == "🦙"
    assert settings.streaming_interval_sec == 0.7
    assert settings.streaming_chunk_size == 150
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.ollama_timeout == 200.0
    assert settings.ollama_context_length == 2048
    assert settings.ollama_headers == {}
    assert settings.ollama_disallowed_models == []
    assert settings.log_level == "INFO"
    assert settings.system_prompts.default == DEFAULT_SYSTEM_PROMPT
    assert settings.system_prompts.svg == SVG_SYSTEM_PROMPT
    assert settings.system_prompts.vision == VISION_SYSTEM_PROMPT


def test_token_is_secret_and_hidden_from_repr(make_settings):
    settings = make_settings()

    assert settings.telegram_bot_token.get_secret_value() == "123:test-token"
    assert "123:test-token" not in repr(settings)


def test_bare_token_variable_is_accepted(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "111:bare")
    monkeypatch.delenv("SECTOR_AI_TELEGRAM_BOT_TOKEN", raising=False)

    assert Settings(_env_file=None).telegram_bot_token.get_secret_value() == "111:bare"


def test_prefixed_token_variable_is_accepted(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("SECTOR_AI_TELEGRAM_BOT_TOKEN", "222:prefixed")

    assert Settings(_env_file=None).telegram_bot_token.get_secret_value() == "222:prefixed"


def test_missing_token_is_rejected(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SECTOR_AI_TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(ValidationError, match="TELEGRAM_BOT_TOKEN"):
        Settings(_env_file=None)


def test_empty_token_is_rejected(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "   ")

    with pytest.raises(ValidationError, match="TELEGRAM_BOT_TOKEN"):
        Settings(_env_file=None)


def test_prefixed_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_DEFAULT_MODEL", "env-model")
    monkeypatch.setenv("SECTOR_AI_STREAMING_INTERVAL_SEC", "2.5")
    monkeypatch.setenv("SECTOR_AI_OLLAMA_CONTEXT_LENGTH", "4096")

    settings = env_only()

    assert settings.default_model == "env-model"
    assert settings.streaming_interval_sec == 2.5
    assert settings.ollama_context_length == 4096


def test_bare_ollama_url_is_accepted(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://ollama:11434")

    assert env_only().ollama_url == "http://ollama:11434"


def test_init_values_beat_the_environment(monkeypatch):
    """The CLI passes values as init kwargs and must win over the environment."""
    monkeypatch.setenv("SECTOR_AI_DEFAULT_MODEL", "env-model")

    settings = env_only(default_model="cli-model")

    assert settings.default_model == "cli-model"


def test_admin_usernames_accepts_comma_separated(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_ADMIN_USERNAMES", "alice, bob ,carol")

    assert env_only().admin_usernames == ["alice", "bob", "carol"]


def test_admin_usernames_accepts_json_list(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_ADMIN_USERNAMES", '["alice", "bob"]')

    assert env_only().admin_usernames == ["alice", "bob"]


def test_admin_usernames_accepts_empty_string(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_ADMIN_USERNAMES", "")

    assert env_only().admin_usernames == []


def test_admin_usernames_accepts_an_init_list(make_settings):
    assert make_settings(admin_usernames=["solo"]).admin_usernames == ["solo"]


def test_admin_usernames_rejects_broken_json(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_ADMIN_USERNAMES", '["alice"')

    with pytest.raises(ValidationError):
        env_only()


def test_disallowed_models_accepts_comma_separated(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_OLLAMA_DISALLOWED_MODELS", "big:latest,huge:latest")

    assert env_only().ollama_disallowed_models == ["big:latest", "huge:latest"]


def test_nested_system_prompt_override(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_SYSTEM_PROMPTS__SVG", "draw only squares")

    settings = env_only()

    assert settings.system_prompts.svg == "draw only squares"
    assert settings.system_prompts.default == DEFAULT_SYSTEM_PROMPT


def test_ollama_headers_are_json(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_OLLAMA_HEADERS", '{"Authorization": "Bearer x"}')

    assert env_only().ollama_headers == {"Authorization": "Bearer x"}


def test_log_level_is_upper_cased(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_LOG_LEVEL", "debug")

    assert env_only().log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError, match="must be one of"):
        env_only()


@pytest.mark.parametrize(
    "overrides",
    [
        {"streaming_interval_sec": -1},
        {"streaming_chunk_size": -1},
        {"ollama_timeout": 0},
        {"ollama_context_length": 0},
        {"default_svg_size": 1},
        {"default_svg_size": 4096},
    ],
)
def test_out_of_range_values_are_rejected(make_settings, overrides):
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_null_context_length_disables_the_cap(make_settings):
    assert make_settings(ollama_context_length=None).ollama_context_length is None


def test_dotenv_file_is_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SECTOR_AI_TELEGRAM_BOT_TOKEN", raising=False)
    (tmp_path / ".env").write_text("TELEGRAM_BOT_TOKEN=333:dotenv\nSECTOR_AI_DEFAULT_MODEL=dotenv-model\n")

    settings = Settings()

    assert settings.telegram_bot_token.get_secret_value() == "333:dotenv"
    assert settings.default_model == "dotenv-model"


def test_environment_beats_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SECTOR_AI_DEFAULT_MODEL=dotenv-model\n")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("SECTOR_AI_DEFAULT_MODEL", "env-model")

    assert Settings().default_model == "env-model"


def test_unknown_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("SECTOR_AI_NOT_A_SETTING", "whatever")

    assert env_only().default_model == "llama3.2-vision:11b-instruct-fp16"
