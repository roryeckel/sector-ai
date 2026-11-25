"""Application settings, loaded from the environment.

There is no configuration file. Values come from environment variables, an
optional ``.env`` file next to the process, or the CLI flags in
:mod:`sector_ai.__main__` (which win over both). ``SECTOR_AI_`` is the prefix
and ``__`` separates nested names, so ``SECTOR_AI_SYSTEM_PROMPTS__SVG``
overrides the SVG prompt. ``TELEGRAM_BOT_TOKEN`` and ``OLLAMA_URL`` are
accepted bare as well, because that is what container users expect to set.

Two backends are supported: Ollama's native API (the default) and any
OpenAI-compatible ``/v1`` endpoint. Set ``SECTOR_AI_LLM_PROVIDER=openai`` and
point ``SECTOR_AI_OPENAI_BASE_URL`` at a server such as Ollama, LM Studio,
vLLM or OpenRouter.
"""

import json
from typing import Annotated

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .prompts import (
    BASIC_POLL_SYSTEM_PROMPT,
    CHARACTERIZER_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    COMPACTION_SYSTEM_PROMPT,
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
LLM_PROVIDERS = ("ollama", "openai")
PROMPT_CACHE_MODES = ("auto", "off", "openai", "openrouter")
PROMPT_CACHE_RETENTION = ("", "in_memory", "24h")
# The bot uses Chat Completions, where OpenAI accepts only "30m" (12h is
# Responses-only and is rejected). "1h" is OpenRouter's Anthropic-style cache.
PROMPT_CACHE_TTL = ("", "30m", "1h")


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
    compaction: str = COMPACTION_SYSTEM_PROMPT
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
        # rejects for fields that carry a validation alias.
        validate_by_name=True,
    )

    telegram_bot_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("TELEGRAM_BOT_TOKEN", "SECTOR_AI_TELEGRAM_BOT_TOKEN"),
    )
    default_model: str = "llama3.2-vision:11b-instruct-fp16"
    # Photos join the normal transcript. When the default model reads images
    # they are sent inline; otherwise an optional vision model describes them,
    # and without one the transcript only records a placeholder.
    default_model_vision: bool = True
    vision_model: str = ""
    # Largest Telegram photo rendition to use (longest side, in pixels).
    image_max_side: int = Field(default=1280, ge=90, le=10000)
    # Photos from earlier messages kept attached after a compaction.
    image_history_limit: int = Field(default=4, ge=0, le=100)
    # Safety cap on attached photos per request (0 = no cap). Exceeding it
    # detaches older photos down to image_history_limit in one step.
    image_max_attached: int = Field(default=32, ge=0, le=1000)
    # Pixels per image token for local estimates; 28 is conservative for Qwen-VL.
    image_patch_px: int = Field(default=28, ge=8, le=128)
    # Telegram sends GIFs as short MP4s; they are sampled into still frames.
    # 0 keeps only the still thumbnail, because the model cannot read video.
    gif_max_frames: int = Field(default=8, ge=0, le=32)
    gif_frame_side: int = Field(default=480, ge=90, le=2000)
    gif_max_bytes: int = Field(default=15 * 1024 * 1024, ge=1024)
    gif_timeout: float = Field(default=20.0, gt=0, le=120)
    # System ffmpeg, or an absolute path. Missing ffmpeg falls back to stills.
    ffmpeg_path: str = "ffmpeg"
    # Seconds to wait for the rest of an album before answering it as one turn.
    album_wait: float = Field(default=1.5, ge=0, le=30)
    # How many times a newer relevant message (or an edit) may restart a turn
    # that has not sent anything yet. 0 disables debouncing.
    debounce_max_restarts: int = Field(default=3, ge=0, le=20)
    default_svg_size: int = Field(default=250, ge=16, le=2048)

    admin_usernames: Annotated[list[str], NoDecode] = ["ex0dus"]

    streaming_cursor: str = "🦙"
    streaming_interval_sec: float = Field(default=0.7, ge=0)
    streaming_chunk_size: int = Field(default=150, ge=0)
    # Render model markdown as Telegram entities (bold, code, lists, quotes).
    markdown: bool = True

    ollama_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_URL", "SECTOR_AI_OLLAMA_URL"),
    )
    llm_provider: str = "ollama"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: SecretStr = SecretStr("")

    llm_timeout: float = Field(
        default=200.0,
        gt=0,
        validation_alias=AliasChoices("SECTOR_AI_LLM_TIMEOUT", "SECTOR_AI_OLLAMA_TIMEOUT"),
    )
    llm_headers: dict[str, str] = Field(
        default={},
        validation_alias=AliasChoices("SECTOR_AI_LLM_HEADERS", "SECTOR_AI_OLLAMA_HEADERS"),
    )
    llm_disallowed_models: Annotated[list[str], NoDecode] = Field(
        default=[],
        validation_alias=AliasChoices("SECTOR_AI_LLM_DISALLOWED_MODELS", "SECTOR_AI_OLLAMA_DISALLOWED_MODELS"),
    )

    # Prompt caching is backend-specific. "auto" applies controls only for
    # endpoints that are known to accept them; self-hosted compatible servers
    # keep their own prefix cache and are never sent vendor-only fields.
    prompt_cache_mode: str = "auto"
    prompt_cache_key_prefix: str = "sector-ai"
    prompt_cache_retention: str = ""
    prompt_cache_ttl: str = ""
    # Ask OpenAI-compatible endpoints to include token usage in stream chunks.
    # Off by default because older local servers reject stream_options.
    stream_usage: bool = False

    # A single bounded tool-calling agent. Explicit commands remain available
    # when autonomous tools are disabled or unsupported by the selected model.
    agent_tools_enabled: bool = True
    agent_max_tool_calls: int = Field(default=6, ge=1, le=30)
    agent_max_iterations: int = Field(default=8, ge=1, le=40)
    agent_timeout: float = Field(default=240.0, gt=0)
    tool_timeout: float = Field(default=90.0, gt=0)
    tool_result_max_chars: int = Field(default=16000, ge=1024, le=100000)
    mcp_config: str = ""
    autoreply_history_messages: int = Field(default=8, ge=0, le=50)
    autoreply_context_tokens: int = Field(default=1536, ge=256, le=8192)

    # Context compaction. A zero context_length means auto-detect from the
    # server, falling back to a conservative default when it advertises none.
    context_length: int = Field(default=0, ge=0)
    compaction_enabled: bool = True
    compaction_threshold: float = Field(default=0.75, gt=0, le=1)
    compaction_keep_messages: int = Field(default=8, ge=1)
    compaction_max_messages: int = Field(default=500, ge=1)
    compaction_summary_max_chars: int = Field(default=4000, ge=200)

    system_prompts: SystemPrompts = Field(default_factory=SystemPrompts)

    log_level: str = "INFO"

    @field_validator("admin_usernames", "llm_disallowed_models", mode="before")
    @classmethod
    def _list_fields(cls, value: object) -> object:
        return _split_list(value)

    @field_validator("llm_provider")
    @classmethod
    def _llm_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if not provider:
            return "ollama"
        if provider not in LLM_PROVIDERS:
            raise ValueError(f"must be one of {', '.join(LLM_PROVIDERS)}")
        return provider

    @field_validator("log_level")
    @classmethod
    def _log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"must be one of {', '.join(LOG_LEVELS)}")
        return level

    @field_validator("prompt_cache_mode")
    @classmethod
    def _prompt_cache_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in PROMPT_CACHE_MODES:
            raise ValueError(f"must be one of {', '.join(PROMPT_CACHE_MODES)}")
        return mode

    @field_validator("prompt_cache_retention")
    @classmethod
    def _prompt_cache_retention(cls, value: str) -> str:
        retention = value.strip()
        if retention not in PROMPT_CACHE_RETENTION:
            raise ValueError(f"must be empty or one of {', '.join(PROMPT_CACHE_RETENTION[1:])}")
        return retention

    @field_validator("prompt_cache_ttl")
    @classmethod
    def _prompt_cache_ttl(cls, value: str) -> str:
        ttl = value.strip()
        if ttl not in PROMPT_CACHE_TTL:
            raise ValueError(f"must be empty or one of {', '.join(PROMPT_CACHE_TTL[1:])}")
        return ttl

    @model_validator(mode="after")
    def _bounded_photo_memory(self) -> "Settings":
        # Only native mode keeps pixels: describe mode replaces them with text
        # and off mode stores a marker, so neither can grow without a cap.
        if self.image_max_attached == 0 and not self.compaction_enabled and self.default_model_vision:
            raise ValueError(
                "SECTOR_AI_IMAGE_MAX_ATTACHED=0 (no cap) needs compaction enabled in native photo mode; "
                "otherwise attached photos are never released"
            )
        return self

    @model_validator(mode="after")
    def _require_token(self) -> "Settings":
        if not self.telegram_bot_token.get_secret_value().strip():
            raise ValueError("TELEGRAM_BOT_TOKEN (or SECTOR_AI_TELEGRAM_BOT_TOKEN) must be set")
        return self
