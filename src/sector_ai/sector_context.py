import asyncio
import json
import logging
from time import time
from typing import Optional
from urllib.parse import urlsplit

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from telegram import Message
from telegram.constants import MessageLimit
from telegram.error import BadRequest
from telegram.ext import Application, CallbackContext

from .chat_metadata import (
    actor_metadata,
    context_message,
    context_messages,
    message_metadata,
    render_prompt_message,
    strip_internal_envelope,
    text_value,
    utc_now,
)
from .history import history_groups, history_text, plain_history, prompt_history_text, safe_cut
from .images import (
    attach_images,
    attached_count,
    describe_pending_images,
    detach_images,
    has_image_blocks,
    image_mode,
    image_signature,
    message_image_tokens,
    photo_records,
    with_images,
)
from .logging_utils import register_secrets
from .message_text import message_refused, message_text
from .prompts import AGENT_SYSTEM_PROMPT
from .settings import Settings
from .usage import record_usage

logger = logging.getLogger(__name__)

# Used when neither the settings nor the server can provide a context window.
DEFAULT_CONTEXT_LENGTH = 8192


def _positive_int(value: object) -> Optional[int]:
    """Return ``value`` as a positive int, or None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        try:
            return _positive_int(int(value.strip()))
        except ValueError:
            return None
    return None


def _estimate_text_tokens(text: str) -> int:
    """A conservative character-based bound.

    Non-ASCII scripts and emoji can run to several tokens per character, while
    ASCII prose stays near four characters per token. Overestimating only costs
    an exact count; underestimating would let an oversized prompt through.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for character in text if ord(character) < 128)
    return max(1, ascii_chars // 4 + (len(text) - ascii_chars) * 4)


def _split_text(text: str, limit: int) -> list[str]:
    """Split text into pieces that each estimate at or below ``limit`` tokens."""
    pieces = []
    start = 0
    tokens = 0.0
    for index, character in enumerate(text):
        cost = 0.25 if ord(character) < 128 else 4.0
        if tokens + cost > limit and index > start:
            pieces.append(text[start:index])
            start = index
            tokens = 0.0
        tokens += cost
    if start < len(text):
        pieces.append(text[start:])
    return pieces or [""]


class SectorContext(CallbackContext):
    def __init__(self, application: Application, chat_id: Optional[int] = None, user_id: Optional[int] = None):
        super().__init__(application=application, chat_id=chat_id, user_id=user_id)
        self._username = None
        self._message_thread_id = 0
        self._request_model = None
        self._request_model_name = None
        self._context_time = None
        # The debounced turn this context is running, if any.
        self._turn = None

    def bind_update(self, update) -> None:
        thread = getattr(update.effective_message, "message_thread_id", None)
        self.bind_topic(thread)
        chat = getattr(update, "effective_chat", None) or getattr(update.effective_message, "chat", None)
        metadata = self.chat_data.setdefault("chat_metadata", {})
        identifier = getattr(chat, "id", None)
        if type(identifier) is int:
            metadata["id"] = identifier
        for key in ("title", "type", "username"):
            value = text_value(getattr(chat, key, None))
            if value:
                metadata[key] = value
        new_title = text_value(getattr(update.effective_message, "new_chat_title", None))
        if new_title:
            metadata["title"] = new_title

    def bind_topic(self, thread_id=0) -> None:
        """Bind updates such as poll results that have no effective message."""
        self._message_thread_id = thread_id if type(thread_id) is int else 0
        self._context_time = utc_now()
        if self._message_thread_id:
            self.conversation_data.setdefault("topic_metadata", {})["id"] = self._message_thread_id

    def remember_placeholder(self, message) -> None:
        """Track the current processing placeholder so a failed turn can clean it."""
        message_id = getattr(message, "message_id", None)
        if type(message_id) is int and type(self._chat_id) is int:
            previous = self.conversation_data.get("pending_placeholder")
            if self.conversation_data.get("placeholder_unsettled") and isinstance(previous, dict):
                # An earlier reply in this turn still shows a cursor. A later
                # reply must not take over its slot and forget it.
                self.conversation_data.setdefault("abandoned_placeholders", []).append(previous)
            self.conversation_data["pending_placeholder"] = {
                "message_id": message_id,
                "chat_id": self._chat_id,
            }
            self.conversation_data.pop("placeholder_unsettled", None)

    def settle_placeholder(self) -> None:
        """Record that the latest edit replaced the streaming cursor."""
        self.conversation_data.pop("placeholder_unsettled", None)

    def mark_placeholder_unsettled(self) -> None:
        """A failed edit may have left the cursor visible in the placeholder."""
        self.conversation_data["placeholder_unsettled"] = True

    def finish_placeholder(self) -> None:
        """Forget the placeholder unless its last delivery left a cursor behind."""
        if self.conversation_data.pop("placeholder_unsettled", False):
            # Keep pending_placeholder so the next turn removes the artifact.
            return
        self.clear_pending_placeholder()

    def clear_pending_placeholder(self) -> None:
        self.conversation_data.pop("pending_placeholder", None)
        self.conversation_data.pop("placeholder_unsettled", None)

    async def remove_pending_placeholder(self) -> None:
        """Remove placeholders abandoned by earlier failed deliveries."""
        abandoned = self.conversation_data.get("abandoned_placeholders") or []
        kept = [item for item in abandoned if not await self._delete_placeholder(item)]
        if kept:
            self.conversation_data["abandoned_placeholders"] = kept
        else:
            self.conversation_data.pop("abandoned_placeholders", None)
        pending = self.conversation_data.get("pending_placeholder")
        if not isinstance(pending, dict):
            return
        if await self._delete_placeholder(pending):
            self.clear_pending_placeholder()

    async def _delete_placeholder(self, pending) -> bool:
        """Delete one placeholder; False only when it is worth trying again later."""
        if not isinstance(pending, dict):
            return True
        message_id = pending.get("message_id")
        chat_id = pending.get("chat_id")
        if type(message_id) is not int or type(chat_id) is not int:
            return True
        try:
            await self.bot.delete_message(chat_id=chat_id, message_id=message_id)
            logger.info("Removed an abandoned processing placeholder")
        except BadRequest:
            # Already gone, or too old to delete. Do not retry forever.
            pass
        except Exception as error:
            logger.debug("Could not remove an abandoned processing placeholder: %s", error)
            return False
        return True

    def context_metadata_messages(self):
        """The static trust note and the volatile snapshot, in that order."""
        if self._context_time is None:
            return None, None
        return context_messages(
            self._context_time,
            dict(self.chat_data.get("chat_metadata", {})),
            dict(self.conversation_data.get("topic_metadata", {})),
            self.chat_data.get("bot_membership"),
        )

    def context_metadata_message(self):
        if self._context_time is None:
            return None
        return context_message(
            self._context_time,
            dict(self.chat_data.get("chat_metadata", {})),
            dict(self.conversation_data.get("topic_metadata", {})),
            self.chat_data.get("bot_membership"),
        )

    def shared_chat_events(self) -> list:
        """Recent chat-wide changes, without duplicating this transcript's events."""
        after = self.conversation_data.get("shared_events_after", 0)
        present = {
            item.additional_kwargs.get("telegram", {}).get("chat_sequence") for item in self.chat_message_history
        }
        return [
            item
            for item in self.chat_data.get("shared_chat_events", [])
            if item.additional_kwargs["telegram"]["chat_sequence"] > after
            and item.additional_kwargs["telegram"]["chat_sequence"] not in present
        ][-2:]

    def record_shared_chat_event(self, message) -> None:
        sequence = self.chat_data.get("chat_event_sequence", 0) + 1
        self.chat_data["chat_event_sequence"] = sequence
        message.additional_kwargs.setdefault("telegram", {})["chat_sequence"] = sequence
        events = self.chat_data.setdefault("shared_chat_events", [])
        events.append(message)
        del events[:-16]
        self.chat_message_history.append(message)
        self._enforce_history_cap()

    def make_user_message(self, message, *, include_command=False, album=None):
        """Build the transcript entry; ``album`` lists every message whose photo belongs to it.

        Photos are recorded as metadata only. Ingestion replaces the records
        with downloaded ones; other callers just need the markers.
        """
        text = message.text or getattr(message, "caption", None) or ""
        if not isinstance(text, str):
            text = ""
        images = photo_records(album if album is not None else [message], self.settings.image_max_side)
        if not text.strip() and not images:
            return None
        if text.startswith("/") and not include_command:
            parts = text.split(maxsplit=1)
            if len(parts) < 2 and not images:
                return None
            text = parts[1] if len(parts) > 1 else ""
        user = getattr(message, "from_user", None)
        sender = actor_metadata(user)
        if not sender:
            sender = actor_metadata(getattr(message, "sender_chat", None))
        name = sender.get("username") or f"user_{sender.get('id', 'unknown')}"
        label = sender.get("username") or sender.get("display_name") or str(sender.get("id", "unknown"))
        additional_kwargs = {"telegram": message_metadata(message)}
        if images:
            additional_kwargs["images"] = images
        return HumanMessage(
            name=name,
            content=f"{label}: {text}" if text.strip() else f"{label}:",
            additional_kwargs=additional_kwargs,
        )

    def save_assistant_message(self, content: str, message=None) -> None:
        metadata = message_metadata(message) if message is not None else {}
        metadata.setdefault("sent_at", utc_now())
        metadata["sender"] = {"is_bot": True}
        self.chat_message_history.append(AIMessage(content=content, additional_kwargs={"telegram": metadata}))

    @property
    def conversation_data(self) -> dict:
        """The canonical in-process graph conversation, isolated by forum topic."""
        if self._message_thread_id:
            return self.chat_data.setdefault("topics", {}).setdefault(self._message_thread_id, {})
        return self.chat_data

    @property
    def conversation_lock(self) -> asyncio.Lock:
        return self.conversation_data.setdefault("lock", asyncio.Lock())

    @staticmethod
    def _update_key(update):
        key = getattr(update, "update_id", None)
        if type(key) is int:
            return ("update", key)
        key = getattr(update.effective_message, "message_id", None)
        return ("message", key) if type(key) is int else None

    def is_processed(self, update) -> bool:
        """Whether ``accept_update`` already took this update, without recording it."""
        key = self._update_key(update)
        return key is not None and key in self.conversation_data.get("processed_updates", [])

    def remember_sent(self, message) -> None:
        """Record a message the bot sent, independent of retained history."""
        identifier = getattr(message, "message_id", None)
        if type(identifier) is not int:
            return
        sent = self.conversation_data.setdefault("sent_message_ids", [])
        if identifier not in sent:
            sent.append(identifier)
            del sent[:-1024]

    def sent_by_bot(self, message_id) -> bool:
        return message_id in self.conversation_data.get("sent_message_ids", [])

    def accept_update(self, update) -> bool:
        """Deduplicate known Telegram updates, including ones with uncertain sends."""
        key = self._update_key(update)
        if key is None:
            return True
        seen = self.conversation_data.setdefault("processed_updates", [])
        if key in seen:
            return False
        seen.append(key)
        del seen[:-512]
        return True

    # Settings

    @property
    def settings(self) -> Settings:
        return self.bot_data["settings"]

    @settings.setter
    def settings(self, new_settings: Settings) -> None:
        self.bot_data["settings"] = new_settings
        # The log formatter keeps redacting a key rotated at runtime.
        register_secrets(new_settings)

    @property
    def bot_llm(self) -> BaseChatModel:
        if self._request_model is not None:
            return self._request_model
        try:
            self.bot_data["llm"]
        except KeyError:
            return self.load_model()
        # Settings can change in-process (the /models menu swaps the model); make
        # sure every caller sees a client built from the current settings.
        self._refresh_model_if_settings_changed()
        return self.bot_data["llm"]

    # Chat

    @property
    def chat_system_prompt(self) -> str:
        try:
            return self.chat_data["system_prompt"]
        except KeyError:
            return self.settings.system_prompts.default

    @chat_system_prompt.setter
    def chat_system_prompt(self, new_prompt: str) -> None:
        self.chat_data["system_prompt"] = new_prompt

    @property
    def chat_autoreply_mode(self) -> bool:
        try:
            return self.chat_data["autoreply_mode"]
        except KeyError:
            self.chat_data["autoreply_mode"] = True
            return True

    @chat_autoreply_mode.setter
    def chat_autoreply_mode(self, mode: bool) -> None:
        self.chat_data["autoreply_mode"] = mode

    @property
    def chat_message_history(self) -> list:
        """The recent transcript, compacted before it can outgrow the server's window."""
        try:
            return self.conversation_data["message_history"]
        except KeyError:
            self.conversation_data["message_history"] = []
            return self.conversation_data["message_history"]

    @chat_message_history.setter
    def chat_message_history(self, new_history: list) -> None:
        self.conversation_data["message_history"] = new_history

    @property
    def context_summary(self) -> str:
        """A terse summary of messages that were compacted out of the history."""
        summary = self.conversation_data.get("context_summary")
        return summary if isinstance(summary, str) else ""

    @context_summary.setter
    def context_summary(self, summary: str) -> None:
        if summary:
            self.conversation_data["context_summary"] = summary
        else:
            self.conversation_data.pop("context_summary", None)

    def clear_history(self) -> None:
        """Drop the transcript, its summary, and anything counted from them."""
        for key in (
            "message_history",
            "context_summary",
            "context_compaction",
            "token_count",
            "tool_schemas",
            "active_tool_call_id",
            "started_tool_call_ids",
            "interrupted_delivery",
            "ingested_messages",
            "message_edits",
        ):
            self.conversation_data.pop(key, None)
        self.conversation_data["shared_events_after"] = self.chat_data.get("chat_event_sequence", 0)
        polls = self.bot_data.get("tracked_polls", {})
        for poll_id, origin in list(polls.items()):
            if origin["chat_id"] == self._chat_id and origin["thread_id"] == self._message_thread_id:
                polls.pop(poll_id, None)

    @property
    def chat_svg_size(self) -> int:
        try:
            return self.chat_data["svg_size"]
        except KeyError:
            default_svg_size = self.settings.default_svg_size
            self.chat_data["svg_size"] = default_svg_size
            return default_svg_size

    @chat_svg_size.setter
    def chat_svg_size(self, size: int) -> None:
        self.chat_data["svg_size"] = size

    # Methods

    async def get_username(self) -> str:
        if self._username:
            return self._username
        me = await self.application.bot.get_me()
        self._username = me.username
        return self._username

    def load_model(self, model_name=None) -> BaseChatModel:
        """Build the chat model for the configured backend.

        The model keeps its own request timeout and headers, and the name is
        stored separately because the two integrations expose it under
        different attributes (``model`` and ``model_name``).
        """
        # In-place credential changes skip the settings setter; the log
        # formatter still needs to learn the new value before it is used.
        name = model_name or self.settings.default_model
        result = self.create_model(name)
        self.bot_data["llm"] = result
        self.bot_data["model_name"] = name
        self.bot_data["llm_config"] = self._model_config(name)
        return result

    def create_model(self, name: str) -> BaseChatModel:
        """Build a request-local client without changing the bot's selected model."""
        register_secrets(self.settings)
        if self.settings.llm_provider == "openai":
            result = ChatOpenAI(
                model=name,
                base_url=self.settings.openai_base_url,
                # Local servers ignore the key, but the client requires one.
                api_key=self.settings.openai_api_key.get_secret_value() or "dummy",
                timeout=self.settings.llm_timeout,
                default_headers=self.settings.llm_headers,
                # This provider is documented as Chat Completions. LangChain
                # otherwise picks Responses for names like "gpt-5-pro" or a
                # server alias ending in "codex", which most gateways reject.
                use_responses_api=False,
                # One slow local request should not double the configured timeout.
                max_retries=0,
                # Off by default: older compatible servers reject stream_options.
                stream_usage=self.settings.stream_usage,
            )
        else:
            result = ChatOllama(
                model=name,
                base_url=self.settings.ollama_url,
                client_kwargs={"timeout": self.settings.llm_timeout, "headers": self.settings.llm_headers},
            )
        return result

    def _model_config(self, name: str) -> tuple:
        provider = self.settings.llm_provider
        endpoint = self.settings.openai_base_url if provider == "openai" else self.settings.ollama_url
        key = self.settings.openai_api_key.get_secret_value() if provider == "openai" else ""
        headers = tuple(sorted(self.settings.llm_headers.items()))
        return (
            provider,
            endpoint.rstrip("/"),
            name,
            self.settings.llm_timeout,
            hash(headers),
            hash(key),
            self.settings.stream_usage,
            self.settings.prompt_cache_mode,
            self.settings.prompt_cache_key_prefix,
            self.settings.prompt_cache_retention,
            self.settings.prompt_cache_ttl,
        )

    def _refresh_model_if_settings_changed(self) -> None:
        """Rebuild the client when settings changed in-process, so probes use it."""
        if self._request_model is not None:
            return
        config = self.bot_data.get("llm_config")
        if config is not None and config != self._model_config(self.get_model()):
            self.load_model(self.get_model())

    def get_model(self) -> str:
        if self._request_model_name is not None:
            return self._request_model_name
        if "model_name" in self.bot_data:
            return self.bot_data["model_name"]
        model = self.bot_data.get("llm")
        for attribute in ("model", "model_name"):
            name = getattr(model, attribute, None)
            if isinstance(name, str):
                return name
        return self.settings.default_model

    @property
    def endpoint_host(self) -> str:
        """The configured host, without credentials or a full path."""
        provider = self.settings.llm_provider
        endpoint = self.settings.openai_base_url if provider == "openai" else self.settings.ollama_url
        return urlsplit(endpoint).hostname or endpoint

    def cache_backend(self) -> str:
        """Which cache controls this endpoint is known to accept.

        Self-hosted OpenAI-compatible servers often do prefix caching on their
        own, but vendor-only request fields can be rejected. ``auto`` therefore
        stays quiet unless the host is recognizable.
        """
        mode = self.settings.prompt_cache_mode
        if mode == "off":
            return "off"
        if self.settings.llm_provider != "openai":
            # The native Ollama client has no fields for these controls.
            return "none"
        if mode in {"openai", "openrouter"}:
            return mode
        host = self.endpoint_host.lower()
        if "openrouter.ai" in host:
            return "openrouter"
        if host == "api.openai.com" or host.endswith(".openai.azure.com"):
            return "openai"
        return "none"

    def cache_request_options(self, workload: str) -> dict:
        """Request-local cache affinity, never credentials or prompt content."""
        backend = self.cache_backend()
        if backend in {"off", "none"}:
            return {}
        prefix = self.settings.prompt_cache_key_prefix.strip() or "sector-ai"
        key = f"{prefix}:{self._chat_id}:{self._message_thread_id}:{workload}"[:64]
        if backend == "openai":
            options = {"prompt_cache_key": key}
            if self.settings.prompt_cache_retention:
                options["prompt_cache_retention"] = self.settings.prompt_cache_retention
            if self.settings.prompt_cache_ttl in {"30m", "12h"}:
                options["prompt_cache_options"] = {
                    "mode": "implicit",
                    "ttl": self.settings.prompt_cache_ttl,
                }
            return options
        if backend == "openrouter":
            # Session affinity keeps a conversation on the provider holding the
            # cache. Top-level cache_control enables automatic caching on
            # Anthropic-style routes and is ignored by providers without it.
            extra = {"session_id": key, "cache_control": {"type": "ephemeral"}}
            if self.settings.prompt_cache_ttl == "1h":
                extra["cache_control"]["ttl"] = "1h"
            return {"extra_body": extra}
        return {}

    def bind_cache(self, model, workload: str):
        """Bind cache affinity to one request without mutating the shared model."""
        options = self.cache_request_options(workload)
        return model.bind(**options) if options else model

    def usage_summary(self) -> list[dict]:
        stats = self.bot_data.get("usage_stats")
        if not isinstance(stats, dict):
            return []
        rows = []
        for (model, workload), entry in sorted(stats.items()):
            if not isinstance(entry, dict):
                continue
            input_tokens = entry.get("input_tokens", 0)
            cache_read = entry.get("cache_read", 0)
            rows.append(
                {
                    "model": model,
                    "workload": workload,
                    "requests": entry.get("requests", 0),
                    "input_tokens": input_tokens,
                    "output_tokens": entry.get("output_tokens", 0),
                    "cache_read": cache_read,
                    "cache_write": entry.get("cache_write", 0),
                    "hit_rate": (100.0 * cache_read / input_tokens) if input_tokens else 0.0,
                }
            )
        return rows

    async def get_model_details(self) -> dict:
        if self.settings.llm_provider != "ollama":
            return {}
        async with httpx.AsyncClient(headers=self.settings.llm_headers) as client:
            response = await client.post(
                f"{self.settings.ollama_url}/api/show",
                json={"name": self.get_model()},
            )

        if not response.is_success:
            return {}

        result = response.json()
        for field in ("license", "modelfile", "parameters", "template"):
            result.pop(field, None)
        return result

    async def get_models(self) -> list:
        if self.settings.llm_provider == "openai":
            return await self._get_openai_models()
        return await self._get_ollama_models()

    async def _get_ollama_models(self) -> list:
        async with httpx.AsyncClient(headers=self.settings.llm_headers) as client:
            response = await client.get(f"{self.settings.ollama_url}/api/tags")
            if not response.is_success:
                return []

            result = [r for r in response.json()["models"] if r["model"] not in self.settings.llm_disallowed_models]
            # One lookup per installed model: gathering them keeps the order of the
            # list while costing one round trip instead of one per model. A single
            # slow or missing model must not cost the whole list.
            shows = await asyncio.gather(
                *(client.post(f"{self.settings.ollama_url}/api/show", json={"name": r["model"]}) for r in result),
                return_exceptions=True,
            )

        for r, show_response in zip(result, shows):
            if isinstance(show_response, Exception):
                logger.warning("Ollama metadata lookup failed for %s: %s", r["model"], show_response)
            elif show_response.is_success:
                r.update(show_response.json())

        return self._label_models(result)

    async def _get_openai_models(self) -> list:
        async with httpx.AsyncClient(headers=self._openai_headers()) as client:
            response = await client.get(f"{self.settings.openai_base_url.rstrip('/')}/models")
        if not response.is_success:
            logger.warning("OpenAI model list request failed with %s", response.status_code)
            return []

        data = response.json().get("data")
        if not isinstance(data, list):
            logger.warning("OpenAI model list returned no data")
            return []

        result = [
            {"model": entry["id"]}
            for entry in data
            if isinstance(entry, dict) and entry.get("id") and entry["id"] not in self.settings.llm_disallowed_models
        ]
        return self._label_models(result)

    def _openai_headers(self) -> httpx.Headers:
        # Headers is case-insensitive, so a custom "authorization" is not
        # duplicated by the Bearer default below.
        headers = httpx.Headers(self.settings.llm_headers)
        if "authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key.get_secret_value() or 'dummy'}"
        return headers

    def _label_models(self, models: list) -> list:
        mode = image_mode(self.settings)
        for r in models:
            r["is_default_active"] = r["model"] == self.get_model()
            # The model that actually reads photos: the default one in native
            # mode, the describing one in describe mode, none when off.
            r["is_vision_active"] = (
                r["is_default_active"]
                if mode == "native"
                else mode == "describe" and r["model"] == self.settings.vision_model
            )
            r["label"] = (
                f"{'✅ ' if r['is_default_active'] else ''}{'👁️‍🗨️ ' if r['is_vision_active'] else ''}{r['model']}"
            )
        return models

    # Context window

    async def get_context_length(self) -> Optional[int]:
        """The server's context window for the active model, when it advertises one.

        Settings win, then a cached value, then a lookup. A successful response
        that carries no window is cached as unknown; a timeout or an auth
        failure is not cached, so fixing the server or its credentials triggers
        a fresh lookup. Values derived from Ollama metadata (rather than the
        running model or an explicit num_ctx) are provisional and expire, since
        a cold model reports training capacity instead of the served window.
        """
        if self.settings.context_length > 0:
            return self.settings.context_length

        key = self._context_length_key()
        cache = self.bot_data.setdefault("context_lengths", {})
        cached = cache.get(key)
        if isinstance(cached, tuple):
            length, expires = cached
            if time() < expires:
                return length
            cache.pop(key, None)
        elif key in cache:
            return cached

        try:
            if self.settings.llm_provider == "openai":
                length, provisional = await self._openai_context_length()
            else:
                length, provisional = await self._ollama_context_length()
        except Exception as error:
            logger.debug("Could not detect the context length (%s); will retry", error)
            return None

        cache[key] = (length, time() + 60) if provisional else length
        return length

    def _context_length_key(self) -> tuple:
        provider = self.settings.llm_provider
        endpoint = (self.settings.openai_base_url if provider == "openai" else self.settings.ollama_url).rstrip("/")
        key = self.settings.openai_api_key.get_secret_value() if provider == "openai" else ""
        headers = tuple(sorted(self.settings.llm_headers.items()))
        # Credentials and headers belong in the key: a 401 or a wrong endpoint
        # must not leave a stale "unknown" cached against the fixed settings.
        return provider, endpoint, self.get_model(), hash(headers), hash(key)

    async def get_context_budget(self) -> int:
        """The token budget compaction works against."""
        return await self.get_context_length() or DEFAULT_CONTEXT_LENGTH

    async def _openai_context_length(self) -> tuple[Optional[int], bool]:
        async with httpx.AsyncClient(
            headers=self._openai_headers(),
            timeout=min(self.settings.llm_timeout, 10.0),
        ) as client:
            response = await client.get(f"{self.settings.openai_base_url.rstrip('/')}/models")
        if not response.is_success:
            if response.status_code in {404, 405, 501}:
                # The route does not exist here; that will not change.
                return None, False
            raise RuntimeError(f"the model list returned HTTP {response.status_code}")

        for entry in response.json().get("data") or []:
            if not isinstance(entry, dict) or entry.get("id") != self.get_model():
                continue
            # Runtime limits first: a training window can be larger than the
            # window the server was started with.
            for field in (
                "context_length",
                "max_model_len",
                "max_context_length",
                "loaded_context_length",
                "max_input_tokens",
            ):
                length = _positive_int(entry.get(field))
                if length:
                    return length, False
            meta = entry.get("meta")
            if isinstance(meta, dict):
                for field in ("n_ctx", "n_ctx_train"):
                    length = _positive_int(meta.get(field))
                    if length:
                        return length, False
            top_provider = entry.get("top_provider")
            if isinstance(top_provider, dict):
                length = _positive_int(top_provider.get("context_length"))
                if length:
                    return length, False
            return None, False
        return None, False

    async def _ollama_context_length(self) -> tuple[Optional[int], bool]:
        base = self.settings.ollama_url.rstrip("/")
        timeout = min(self.settings.llm_timeout, 10.0)
        async with httpx.AsyncClient(headers=self.settings.llm_headers, timeout=timeout) as client:
            running = await client.get(f"{base}/api/ps")
            if not running.is_success and running.status_code not in {404, 405, 501}:
                raise RuntimeError(f"the running model list returned HTTP {running.status_code}")
            if running.is_success:
                # The loaded model reports the window it was actually started with.
                for model in running.json().get("models") or []:
                    if not isinstance(model, dict):
                        continue
                    if (model.get("model") or model.get("name")) == self.get_model():
                        length = _positive_int(model.get("context_length"))
                        if length:
                            return length, False
            shown = await client.post(f"{base}/api/show", json={"name": self.get_model()})
        if not shown.is_success:
            if shown.status_code in {404, 405, 501}:
                return None, False
            raise RuntimeError(f"the model details returned HTTP {shown.status_code}")

        data = shown.json()
        # An explicit num_ctx is the window the user configured. The
        # architecture's context_length is training capacity, which is only an
        # upper bound and can be far above what the server actually serves.
        for line in str(data.get("parameters") or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "num_ctx":
                length = _positive_int(parts[1])
                if length:
                    return length, False
        info = data.get("model_info") or {}
        lengths = [
            length
            for key, value in info.items()
            if key.endswith(".context_length")
            for length in [_positive_int(value)]
            if length
        ]
        if lengths:
            # Provisional: a cold model reports what it was trained for.
            return max(lengths), True
        return None, False

    # Token counting

    async def count_context_tokens(self) -> int:
        """Count the prompt the next chat request would send.

        The count comes from the model server, not a local tokenizer. When the
        context has not changed since the last count, the cached value is
        returned without another request. If the server cannot report a count,
        a local estimate is returned and ``token_count_is_exact`` is False.
        """
        self._refresh_model_if_settings_changed()
        messages = await self._current_context_messages()
        fingerprint = self._token_count_fingerprint(messages)
        cached = self.conversation_data.get("token_count")
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            cached_tokens = cached.get("tokens")
            if isinstance(cached_tokens, int):
                return cached_tokens

        tokens = None
        exact = True
        if self.settings.llm_provider == "openai":
            # Free on OpenAI and any server that implements it; Ollama does not.
            tokens = await self._count_with_openai_endpoint(messages)
        if tokens is None:
            try:
                tokens = await self._probe_token_count(messages)
            except Exception:
                logger.warning("Exact token count failed; using a local estimate", exc_info=True)
                tokens = self._estimate_token_count(messages, include_tools=True)
                exact = False

        self.conversation_data["token_count"] = {"fingerprint": fingerprint, "tokens": tokens, "exact": exact}
        return tokens

    def _estimate_token_count(self, messages: list[BaseMessage], *, include_tools: bool = False) -> int:
        """A dependency-free fallback; conservative for non-ASCII text."""
        text = "\n".join(prompt_history_text(message) for message in messages)
        if include_tools:
            text += json.dumps(self.conversation_data.get("tool_schemas", []), ensure_ascii=False)
        # Image blocks are pixels, not text; estimate them from their size.
        patch = self.settings.image_patch_px
        return _estimate_text_tokens(text) + sum(message_image_tokens(message, patch) for message in messages)

    @property
    def token_count_is_exact(self) -> bool:
        cached = self.conversation_data.get("token_count")
        return bool(isinstance(cached, dict) and cached.get("exact"))

    def _token_count_fingerprint(self, messages: list[BaseMessage]) -> tuple:
        return (
            *self._model_config(self.get_model()),
            tuple((m.type, getattr(m, "name", None), prompt_history_text(m), image_signature(m)) for m in messages),
            repr(self.conversation_data.get("tool_schemas", [])),
        )

    async def _count_with_openai_endpoint(self, messages: list[BaseMessage]) -> int | None:
        """Use ``/responses/input_tokens`` when the server has it, which costs nothing."""
        # This endpoint's Responses projection would omit Chat Completions tool
        # calls/results. Use the exact chat probe instead of undercounting them.
        if self.conversation_data.get("tool_schemas") or any(
            getattr(m, "tool_calls", None) or m.type == "tool" for m in messages
        ):
            return None
        # The text projection below would drop images and undercount them.
        if has_image_blocks(messages):
            return None
        endpoint = self.settings.openai_base_url.rstrip("/")
        unavailable = self.bot_data.setdefault("token_count_unavailable", set())
        if endpoint in unavailable:
            return None

        instructions = []
        input_items = []
        for message in messages:
            text = history_text(message)
            if message.type == "system":
                instructions.append(text)
            elif message.type == "human":
                input_items.append({"role": "user", "content": text})
            elif message.type == "ai":
                input_items.append({"role": "assistant", "content": text})

        payload = {"model": self.get_model(), "input": input_items}
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)

        url = f"{endpoint}/responses/input_tokens"
        try:
            async with httpx.AsyncClient(
                headers=self._openai_headers(),
                # A server without this endpoint should not hold up /tokens.
                timeout=min(self.settings.llm_timeout, 10.0),
            ) as client:
                response = await client.post(url, json=payload)
            if not response.is_success:
                # Forget only endpoints that clearly do not implement the route;
                # transient failures are retried on the next uncached call.
                if response.status_code in {404, 405, 501}:
                    unavailable.add(endpoint)
                logger.debug("Token count endpoint returned %s; using a probe instead", response.status_code)
                return None
            tokens = response.json().get("input_tokens")
        except Exception as error:
            logger.debug("Token count endpoint unavailable (%s); using a probe instead", error)
            return None
        return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0 else None

    async def _probe_token_count(self, messages: list[BaseMessage]) -> int:
        """Ask the model for one token and read the prompt count it reports."""
        model = self.bot_llm
        schemas = self.conversation_data.get("tool_schemas", [])
        if schemas:
            model = model.bind_tools(schemas)
        probe_model = self.bind_cache(model, "token_probe")
        if self.settings.llm_provider == "ollama":
            probe = probe_model.bind(options={"num_predict": 1})
        elif self._uses_responses_api():
            # LangChain turns max_tokens into the Responses API's max_output_tokens.
            probe = probe_model.bind(max_tokens=1)
        else:
            # ChatOpenAI rewrites max_tokens to max_completion_tokens, which
            # Ollama ignores, so put max_tokens straight into the request body.
            probe = probe_model.bind(extra_body={"max_tokens": 1})

        try:
            response = await probe.ainvoke(messages)
        except Exception as error:
            if self.settings.llm_provider != "openai" or getattr(error, "status_code", None) != 400:
                raise
            if self._uses_responses_api():
                # Some Responses models reject a one-token cap.
                response = await probe_model.bind(extra_body={"max_output_tokens": 16}).ainvoke(messages)
            else:
                # Newer OpenAI models only accept max_completion_tokens.
                response = await probe_model.bind(extra_body={"max_completion_tokens": 16}).ainvoke(messages)

        record_usage(self, "token_probe", response)
        tokens = (getattr(response, "usage_metadata", None) or {}).get("input_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise RuntimeError("the model server did not report prompt tokens")
        return tokens

    def _uses_responses_api(self) -> bool:
        explicit = getattr(self.bot_llm, "use_responses_api", None)
        if isinstance(explicit, bool):
            return explicit
        check = getattr(self.bot_llm, "_use_responses_api", None)
        if not callable(check):
            return False
        try:
            return bool(check({}))
        except Exception:
            return False

    async def maybe_compact(self) -> bool:
        """Compact the history when the next request would approach the budget.

        Returns True when a compaction happened. Any failure leaves the
        transcript in place where possible, and the request is still made to fit
        by dropping the oldest messages as a last resort.

        Photos waiting for a description are described first (describe mode
        only), so the estimate and the next request see the final text.
        """
        try:
            await describe_pending_images(self)
        except Exception:
            logger.warning("Describing pending photos failed", exc_info=True)
        try:
            if not self.settings.compaction_enabled:
                self._enforce_history_cap()
                return False
            if not self.chat_message_history:
                return False

            budget = await self.get_context_budget()
            threshold = self.settings.compaction_threshold
            estimate = self._estimate_token_count(await self._current_context_messages(), include_tools=True)
            # A cheap preflight that scales with the configured threshold. The
            # real window and the exact count are resolved past this point.
            if estimate < budget * threshold * 0.5:
                return False

            # A one-token counting probe is itself an inference request. Do
            # not send an already oversized prompt merely to discover its
            # exact size; compact using the conservative estimate instead.
            tokens = estimate if estimate > budget * 0.9 else await self.count_context_tokens()
            if tokens <= budget * threshold:
                return False

            compacted = False
            try:
                result = await self.compact()
                compacted = bool(result.get("compacted"))
            except Exception:
                logger.warning("Context compaction failed; dropping the oldest messages", exc_info=True)
            # Keeping the newest messages verbatim can still leave an oversized
            # prompt, and a failed summary leaves the old transcript in place.
            await self._trim_to_budget(budget)
            return compacted
        except Exception:
            logger.warning("Context compaction check failed; keeping the transcript", exc_info=True)
            self._enforce_history_cap()
            return False

    async def compact(self) -> dict:
        """Fold the oldest messages into the running summary.

        The most recent ``compaction_keep_messages`` messages stay verbatim so
        the reply still sees the immediate conversation exactly.
        """
        history = list(self.chat_message_history)
        if len(history) <= 1:
            return {"compacted": False, "evicted": 0, "kept": len(history), "summary": self.context_summary}

        # Keep the configured tail, but always leave at least one message to
        # summarize; a forced compaction on a short history still makes progress.
        keep = max(1, min(self.settings.compaction_keep_messages, len(history) - 1))

        cut = safe_cut(history, len(history) - keep)
        if cut == 0:
            return {"compacted": False, "evicted": 0, "kept": len(history), "summary": self.context_summary}
        evicted = history[:cut]
        max_chars = self.settings.compaction_summary_max_chars
        budget = await self.get_context_budget()
        summary = self.context_summary
        remaining = list(evicted)
        while remaining:
            # The summary grows between requests, so size each chunk against
            # the space left after it and the reply allowance.
            capacity = self._chunk_capacity(budget, summary, max_chars)
            chunk, remaining = self._take_summary_chunk(remaining, capacity)
            summary = await self._summarize_chunk(summary, chunk, max_chars)

        # The prompt prefix changes here anyway, so this is where older photos
        # are detached without costing an extra cache miss.
        kept, detached = detach_images(history[cut:], self.settings.image_history_limit)
        if detached:
            logger.info("Detached %s older photos during compaction", detached)
        self.chat_message_history = kept
        self.context_summary = summary
        self.conversation_data["context_compaction"] = {
            "evicted": len(evicted),
            "kept": len(history) - len(evicted),
            "at": time(),
        }
        self.conversation_data.pop("token_count", None)
        logger.info("Compacted %s messages into a %s character summary", len(evicted), len(summary))
        return {"compacted": True, "evicted": len(evicted), "kept": len(history) - len(evicted), "summary": summary}

    async def _summarize_chunk(self, previous: str, messages: list, max_chars: int) -> str:
        system_prompt = self.settings.system_prompts.compaction.replace("{MAX_CHARS}", str(max_chars))
        transcript = "\n".join(f"[{message.type}] {history_text(message)}" for message in messages)
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Previous summary:\n{previous or '(none yet)'}\n\n"
                    f"Transcript to merge into the summary:\n{transcript}"
                )
            ),
        ]

        response = await self.bind_cache(self.bot_llm, "compaction").ainvoke(prompt)
        record_usage(self, "compaction", response)
        if message_refused(response):
            raise RuntimeError("the model refused to summarize the transcript")
        summary = strip_internal_envelope(message_text(response)).strip()
        if not summary:
            raise RuntimeError("the model returned an empty summary")
        return summary[:max_chars]

    def _chunk_capacity(self, budget: int, summary: str, max_chars: int) -> int:
        """Tokens available for one summarization transcript."""
        instructions = _estimate_text_tokens(self.settings.system_prompts.compaction)
        output = min(max_chars // 4, max(1, budget // 4))
        capacity = budget - instructions - _estimate_text_tokens(summary) - output - 64
        if capacity < 128:
            raise RuntimeError("no room to summarize inside the context window")
        return capacity

    @staticmethod
    def _take_summary_chunk(messages: list, limit: int) -> tuple[list, list]:
        """Take one chunk from the front, splitting an oversized first message."""
        chunk = []
        tokens = 0
        for index, message in enumerate(messages):
            text = history_text(message)
            size = _estimate_text_tokens(text)
            if size > limit:
                pieces = _split_text(text, limit)
                if chunk:
                    return chunk, messages[index:]
                rest = [HumanMessage(content=piece) for piece in pieces[1:]] + messages[index + 1 :]
                return [HumanMessage(content=pieces[0])], rest
            if chunk and tokens + size > limit:
                return chunk, messages[index:]
            chunk.append(message)
            tokens += size
        return chunk, []

    async def _trim_to_budget(self, budget: int) -> None:
        """Drop context until the complete rendered prompt fits the window.

        The summary is shrunk first because it is the cheapest thing to lose.
        This is the last resort after a compaction that could not free enough,
        or a summarizer that failed: it trades old context for a request the
        server can still accept.
        """
        limit = budget * self.settings.compaction_threshold * 0.9
        history = self.chat_message_history

        def estimate(messages) -> int:
            # Count history as it is sent, attached photos included.
            return self._estimate_token_count(self._prompt_projection(messages))

        async def overhead() -> int:
            full = self._estimate_token_count(await self._current_context_messages(), include_tools=True)
            return max(0, full - estimate(history))

        # The newest message stays whatever happens, so the summary must leave
        # room for it as well as for the system prompt.
        groups = history_groups(history)
        newest = estimate(groups[-1]) if groups else 0
        while self.context_summary and await overhead() + newest > limit:
            summary = self.context_summary
            self.context_summary = summary[: len(summary) // 2]

        base = await overhead()
        keep_from = 0
        while len(groups) > 1 and base + estimate(history[keep_from:]) > limit:
            keep_from += len(groups.pop(0))
        if keep_from:
            self.chat_message_history = history[keep_from:]
            self.conversation_data.pop("token_count", None)
            logger.warning("Dropped %s messages to fit the context window", keep_from)

        final = self._estimate_token_count(await self._current_context_messages(), include_tools=True)
        if final > limit:
            logger.warning("The context still estimates %s tokens over the window", final - limit)

    def _enforce_history_cap(self) -> None:
        """Bound memory when compaction is off or failed."""
        history = self.chat_message_history
        cap = self.settings.compaction_max_messages
        if len(history) > cap:
            self.chat_message_history = history[safe_cut(history, len(history) - cap) :]
            self.conversation_data.pop("token_count", None)
            logger.warning("History exceeded %s messages; dropped the oldest ones", cap)

    async def _current_context_messages(self) -> list:
        has_tools = bool(self.conversation_data.get("tool_schemas"))
        messages = self.get_templated_messages(tool_history=has_tools).format_messages(
            **(await self.get_system_template_dict())
        )
        if has_tools:
            messages.insert(1, SystemMessage(content=AGENT_SYSTEM_PROMPT))
        return messages

    def save_user_message(self, message: Message, *, include_command: bool = False, album=None, images=None) -> bool:
        """Append one incoming message; ``images`` are downloaded photo records."""
        event = SectorContext.make_user_message(self, message, include_command=include_command, album=album)
        if event is None:
            return False
        if images:
            event = with_images(event, images)
        seen = self.conversation_data.setdefault("ingested_messages", [])
        message_id = getattr(message, "message_id", None)
        if type(message_id) is int and message_id in seen:
            return True
        for item in album or [message]:
            identifier = getattr(item, "message_id", None)
            if type(identifier) is int and identifier not in seen:
                seen.append(identifier)
        del seen[:-512]
        self.chat_message_history.append(event)
        # Messages arrive even when autoreply stays silent, so the cap cannot
        # wait for the reply path.
        self._enforce_history_cap()
        if images:
            self._enforce_image_cap()
        return True

    def _enforce_image_cap(self) -> None:
        """Bound attached photos (and their memory) even when the bot stays silent.

        Detaching goes down to ``image_history_limit`` in one step, so the
        cache-breaking change happens once rather than on every new photo.
        """
        cap = self.settings.image_max_attached
        history = self.chat_message_history
        if not cap or attached_count(history) <= cap:
            return
        kept, detached = detach_images(history, min(self.settings.image_history_limit, cap))
        if detached:
            self.chat_message_history = kept
            self.conversation_data.pop("token_count", None)
            logger.info("More than %s photos attached; detached %s older ones", cap, detached)

    def _prompt_projection(self, messages: list) -> list:
        """Render history as the model receives it, with photos attached in native mode."""
        native = image_mode(self.settings) == "native"
        rendered = [render_prompt_message(message) for message in messages]
        return [attach_images(message) for message in rendered] if native else rendered

    def message_exists(self, new_message: ChatMessage) -> bool:
        return any(
            existing_message.content == new_message.content and existing_message.type == new_message.type
            for existing_message in self.chat_message_history
        )

    async def get_system_template_dict(self) -> dict:
        username = await self.get_username()
        return {
            "MAX_LEN": MessageLimit.MAX_TEXT_LENGTH,
            "USERNAME": username,
        }

    def get_templated_messages(self, system_prompt=None, num_history=None, *, tool_history=False) -> ChatPromptTemplate:
        # logger.info(f"System Prompt: {system_prompt or self.chat_system_prompt}")
        system_prompt = system_prompt or self.chat_system_prompt
        included_history = (
            self.chat_message_history
            if num_history is None
            else list(self.chat_message_history)[
                safe_cut(self.chat_message_history, max(0, len(self.chat_message_history) - num_history)) :
            ]
        )
        if not tool_history:
            included_history = plain_history(included_history)
        messages = [SystemMessagePromptTemplate.from_template(template=system_prompt)]
        static, dynamic = self.context_metadata_messages()
        if static is not None:
            messages.append(static)
        if self.context_summary:
            # A separate message keeps a user-supplied prompt's braces intact.
            messages.append(SystemMessage(content=f"Summary of the earlier conversation:\n{self.context_summary}"))
        # Retained history comes before changing shared events and the current
        # snapshot so the longest possible prompt prefix stays reusable.
        messages.extend(self._prompt_projection(included_history))
        messages.extend(render_prompt_message(message) for message in self.shared_chat_events())
        if dynamic is not None:
            messages.append(dynamic)
        return ChatPromptTemplate.from_messages(messages)
