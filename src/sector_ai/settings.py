"""Application settings, loaded from the environment.

There is no configuration file. Values come from environment variables, an
optional ``.env`` file next to the process, or the CLI flags in
:mod:`sector_ai.__main__` (which win over both). ``SECTOR_AI_`` is the prefix
and ``__`` separates nested names, so ``SECTOR_AI_SYSTEM_PROMPTS__SVG``
overrides the SVG prompt. ``TELEGRAM_BOT_TOKEN`` and ``OLLAMA_URL`` are
accepted bare as well, because that is what container users expect to set.
"""

import json
from typing import Annotated

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .prompts import (
    BASIC_POLL_SYSTEM_PROMPT,
    CHARACTERIZER_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    DECIDE_AUTOREPLY_SYSTEM_PROMPT,
    DECIDE_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    EMOJI_SYSTEM_PROMPT,
    HTML_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    SVG_SYSTEM_PROMPT,
    TOPIC_POLL_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
)

LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _split_list(value: object) -> object:
    """Accept a comma/space separated string or a JSON list for list-valued settings."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [part for part in text.replace(",", " ").split() if part]


class SystemPrompts(BaseModel):
    default: str = DEFAULT_SYSTEM_PROMPT
    summarization: str = SUMMARIZATION_SYSTEM_PROMPT
    emoji: str = EMOJI_SYSTEM_PROMPT
    basic_poll: str = BASIC_POLL_SYSTEM_PROMPT
    topic_poll: str = TOPIC_POLL_SYSTEM_PROMPT
    characterizer: str = CHARACTERIZER_SYSTEM_PROMPT
    code: str = CODE_SYSTEM_PROMPT
    html: str = HTML_SYSTEM_PROMPT
    svg: str = SVG_SYSTEM_PROMPT
    decide: str = DECIDE_SYSTEM_PROMPT
    decide_autoreply: str = DECIDE_AUTOREPLY_SYSTEM_PROMPT
    vision: str = VISION_SYSTEM_PROMPT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SECTOR_AI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # CLI overrides are passed by field name, which pydantic otherwise
        # rejects for the two fields that carry a validation alias.
        validate_by_name=True,
    )

    telegram_bot_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("TELEGRAM_BOT_TOKEN", "SECTOR_AI_TELEGRAM_BOT_TOKEN"),
    )
    default_model: str = "llama3.2-vision:11b-instruct-fp16"
    vision_model: str = "llama3.2-vision:11b-instruct-fp16"
    default_svg_size: int = Field(default=250, ge=16, le=2048)

    admin_usernames: Annotated[list[str], NoDecode] = ["ex0dus"]

    streaming_cursor: str = "🦙"
    streaming_interval_sec: float = Field(default=0.7, ge=0)
    streaming_chunk_size: int = Field(default=150, ge=0)

    ollama_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_URL", "SECTOR_AI_OLLAMA_URL"),
    )
    ollama_timeout: float = Field(default=200.0, gt=0)
    ollama_context_length: int | None = Field(default=2048, gt=0)
    ollama_headers: dict[str, str] = {}
    ollama_disallowed_models: Annotated[list[str], NoDecode] = []

    system_prompts: SystemPrompts = Field(default_factory=SystemPrompts)

    log_level: str = "INFO"

    @field_validator("admin_usernames", "ollama_disallowed_models", mode="before")
    @classmethod
    def _list_fields(cls, value: object) -> object:
        return _split_list(value)

    @field_validator("log_level")
    @classmethod
    def _log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"must be one of {', '.join(LOG_LEVELS)}")
        return level

    @model_validator(mode="after")
    def _require_token(self) -> "Settings":
        if not self.telegram_bot_token.get_secret_value().strip():
            raise ValueError("TELEGRAM_BOT_TOKEN (or SECTOR_AI_TELEGRAM_BOT_TOKEN) must be set")
        return self
