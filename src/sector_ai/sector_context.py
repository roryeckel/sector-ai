import asyncio
import logging
from time import time
from typing import Optional

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from telegram import Message
from telegram.constants import MessageLimit
from telegram.ext import Application, CallbackContext

from .message_text import message_refused, message_text
from .settings import Settings

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


class SectorContext(CallbackContext):
    def __init__(self, application: Application, chat_id: Optional[int] = None, user_id: Optional[int] = None):
        super().__init__(application=application, chat_id=chat_id, user_id=user_id)
        self._username = None

    # Settings

    @property
    def settings(self) -> Settings:
        return self.bot_data["settings"]

    @settings.setter
    def settings(self, new_settings: Settings) -> None:
        self.bot_data["settings"] = new_settings

    @property
    def bot_llm(self) -> BaseChatModel:
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
            return self.chat_data["message_history"]
        except KeyError:
            self.chat_data["message_history"] = []
            return self.chat_data["message_history"]

    @chat_message_history.setter
    def chat_message_history(self, new_history: list) -> None:
        self.chat_data["message_history"] = new_history

    @property
    def context_summary(self) -> str:
        """A terse summary of messages that were compacted out of the history."""
        summary = self.chat_data.get("context_summary")
        return summary if isinstance(summary, str) else ""

    @context_summary.setter
    def context_summary(self, summary: str) -> None:
        if summary:
            self.chat_data["context_summary"] = summary
        else:
            self.chat_data.pop("context_summary", None)

    def clear_history(self) -> None:
        """Drop the transcript, its summary, and anything counted from them."""
        self.chat_data["message_history"] = []
        self.chat_data.pop("context_summary", None)
        self.chat_data.pop("context_compaction", None)
        self.chat_data.pop("token_count", None)

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
        name = model_name or self.settings.default_model
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
            )
        else:
            result = ChatOllama(
                model=name,
                base_url=self.settings.ollama_url,
                client_kwargs={"timeout": self.settings.llm_timeout, "headers": self.settings.llm_headers},
            )
        self.bot_data["llm"] = result
        self.bot_data["model_name"] = name
        self.bot_data["llm_config"] = self._model_config(name)
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
        )

    def _refresh_model_if_settings_changed(self) -> None:
        """Rebuild the client when settings changed in-process, so probes use it."""
        config = self.bot_data.get("llm_config")
        if config is not None and config != self._model_config(self.get_model()):
            self.load_model(self.get_model())

    def get_model(self) -> str:
        if "model_name" in self.bot_data:
            return self.bot_data["model_name"]
        model = self.bot_data.get("llm")
        for attribute in ("model", "model_name"):
            name = getattr(model, attribute, None)
            if isinstance(name, str):
                return name
        return self.settings.default_model

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
        for r in models:
            r["is_default_active"] = r["model"] == self.get_model()
            r["is_vision_active"] = r["model"] == self.settings.vision_model
            r["label"] = (
                f"{'✅ ' if r['is_default_active'] else ''}{'👁️‍🗨️ ' if r['is_vision_active'] else ''}{r['model']}"
            )
        return models

    # Context window

    async def get_context_length(self) -> Optional[int]:
        """The server's context window for the active model, when it advertises one.

        Settings win, then a cached value, then a one-time lookup. A successful
        response that carries no window is cached as unknown; a timeout or an
        auth failure is not cached, so fixing the server or its credentials
        triggers a fresh lookup.
        """
        if self.settings.context_length > 0:
            return self.settings.context_length

        key = self._context_length_key()
        cache = self.bot_data.setdefault("context_lengths", {})
        if key in cache:
            return cache[key]

        try:
            if self.settings.llm_provider == "openai":
                length = await self._openai_context_length()
            else:
                length = await self._ollama_context_length()
        except Exception as error:
            logger.debug("Could not detect the context length (%s); will retry", error)
            return None

        cache[key] = length
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

    async def _openai_context_length(self) -> Optional[int]:
        async with httpx.AsyncClient(
            headers=self._openai_headers(),
            timeout=min(self.settings.llm_timeout, 10.0),
        ) as client:
            response = await client.get(f"{self.settings.openai_base_url.rstrip('/')}/models")
        if not response.is_success:
            if response.status_code in {404, 405, 501}:
                # The route does not exist here; that will not change.
                return None
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
                    return length
            meta = entry.get("meta")
            if isinstance(meta, dict):
                for field in ("n_ctx", "n_ctx_train"):
                    length = _positive_int(meta.get(field))
                    if length:
                        return length
            top_provider = entry.get("top_provider")
            if isinstance(top_provider, dict):
                length = _positive_int(top_provider.get("context_length"))
                if length:
                    return length
            return None
        return None

    async def _ollama_context_length(self) -> Optional[int]:
        base = self.settings.ollama_url.rstrip("/")
        timeout = min(self.settings.llm_timeout, 10.0)
        async with httpx.AsyncClient(headers=self.settings.llm_headers, timeout=timeout) as client:
            running = await client.get(f"{base}/api/ps")
            if running.is_success:
                # The loaded model reports the window it was actually started with.
                for model in running.json().get("models") or []:
                    if not isinstance(model, dict):
                        continue
                    if (model.get("model") or model.get("name")) == self.get_model():
                        length = _positive_int(model.get("context_length"))
                        if length:
                            return length
            shown = await client.post(f"{base}/api/show", json={"name": self.get_model()})
        if not shown.is_success:
            if shown.status_code in {404, 405, 501}:
                return None
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
                    return length
        info = data.get("model_info") or {}
        lengths = [
            length
            for key, value in info.items()
            if key.endswith(".context_length")
            for length in [_positive_int(value)]
            if length
        ]
        if lengths:
            return max(lengths)
        return None

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
        cached = self.chat_data.get("token_count")
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
                tokens = self._estimate_token_count(messages)
                exact = False

        self.chat_data["token_count"] = {"fingerprint": fingerprint, "tokens": tokens, "exact": exact}
        return tokens

    def _estimate_token_count(self, messages: list[BaseMessage]) -> int:
        """A dependency-free fallback; roughly four characters per token."""
        text = "\n".join(message_text(message) for message in messages)
        if not text:
            return 0
        return max(1, len(text) // 4)

    @property
    def token_count_is_exact(self) -> bool:
        cached = self.chat_data.get("token_count")
        return bool(isinstance(cached, dict) and cached.get("exact"))

    def _token_count_fingerprint(self, messages: list[BaseMessage]) -> tuple:
        return (
            *self._model_config(self.get_model()),
            tuple((m.type, getattr(m, "name", None), repr(m.content)) for m in messages),
        )

    async def _count_with_openai_endpoint(self, messages: list[BaseMessage]) -> int | None:
        """Use ``/responses/input_tokens`` when the server has it, which costs nothing."""
        endpoint = self.settings.openai_base_url.rstrip("/")
        unavailable = self.bot_data.setdefault("token_count_unavailable", set())
        if endpoint in unavailable:
            return None

        instructions = []
        input_items = []
        for message in messages:
            text = message_text(message)
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
        if self.settings.llm_provider == "ollama":
            probe = self.bot_llm.bind(options={"num_predict": 1})
        elif self._uses_responses_api():
            # LangChain turns max_tokens into the Responses API's max_output_tokens.
            probe = self.bot_llm.bind(max_tokens=1)
        else:
            # ChatOpenAI rewrites max_tokens to max_completion_tokens, which
            # Ollama ignores, so put max_tokens straight into the request body.
            probe = self.bot_llm.bind(extra_body={"max_tokens": 1})

        try:
            response = await probe.ainvoke(messages)
        except Exception as error:
            if self.settings.llm_provider != "openai" or getattr(error, "status_code", None) != 400:
                raise
            if self._uses_responses_api():
                # Some Responses models reject a one-token cap.
                response = await self.bot_llm.bind(extra_body={"max_output_tokens": 16}).ainvoke(messages)
            else:
                # Newer OpenAI models only accept max_completion_tokens.
                response = await self.bot_llm.bind(extra_body={"max_completion_tokens": 16}).ainvoke(messages)

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
        """
        try:
            if not self.settings.compaction_enabled:
                self._enforce_history_cap()
                return False
            if not self.chat_message_history:
                return False

            budget = await self.get_context_budget()
            threshold = self.settings.compaction_threshold
            estimate = self._estimate_token_count(await self._current_context_messages())
            # A cheap preflight that scales with the configured threshold. The
            # real window and the exact count are resolved past this point.
            if estimate < budget * threshold * 0.5:
                return False

            tokens = await self.count_context_tokens()
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
            self._trim_to_budget(budget)
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

        evicted = history[:-keep]
        max_chars = self.settings.compaction_summary_max_chars
        # Most evictions fit one request. Only a transcript large enough to
        # approach a server window needs the extra lookup and chunking.
        tokens = sum(max(1, len(message_text(message)) // 4) for message in evicted)
        limit = max(256, tokens)
        if tokens > 2048:
            limit = max(256, (await self.get_context_budget()) // 2)
        summary = self.context_summary
        for chunk in self._summary_chunks(evicted, limit):
            summary = await self._summarize_chunk(summary, chunk, max_chars)

        self.chat_data["message_history"] = history[-keep:]
        self.context_summary = summary
        self.chat_data["context_compaction"] = {
            "evicted": len(evicted),
            "kept": len(history) - len(evicted),
            "at": time(),
        }
        self.chat_data.pop("token_count", None)
        logger.info("Compacted %s messages into a %s character summary", len(evicted), len(summary))
        return {"compacted": True, "evicted": len(evicted), "kept": len(history) - len(evicted), "summary": summary}

    async def _summarize_chunk(self, previous: str, messages: list, max_chars: int) -> str:
        system_prompt = self.settings.system_prompts.compaction.replace("{MAX_CHARS}", str(max_chars))
        transcript = "\n".join(f"[{message.type}] {message_text(message)}" for message in messages)
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Previous summary:\n{previous or '(none yet)'}\n\n"
                    f"Transcript to merge into the summary:\n{transcript}"
                )
            ),
        ]

        response = await self.bot_llm.ainvoke(prompt)
        if message_refused(response):
            raise RuntimeError("the model refused to summarize the transcript")
        summary = message_text(response).strip()
        if not summary:
            raise RuntimeError("the model returned an empty summary")
        return summary[:max_chars]

    @staticmethod
    def _summary_chunks(messages: list, limit: int) -> list[list]:
        """Split a transcript so no summarization request is oversized."""
        chunks = []
        chunk = []
        tokens = 0
        for message in messages:
            size = max(1, len(message_text(message)) // 4)
            if chunk and tokens + size > limit:
                chunks.append(chunk)
                chunk = []
                tokens = 0
            chunk.append(message)
            tokens += size
        if chunk:
            chunks.append(chunk)
        return chunks

    def _trim_to_budget(self, budget: int) -> None:
        """Drop the oldest messages until the estimated context fits the window.

        This is the last resort after a compaction that could not free enough,
        or a summarizer that failed. It trades old context for a request the
        server can still accept.
        """
        limit = budget * self.settings.compaction_threshold * 0.9
        history = self.chat_message_history
        keep_from = 0
        while len(history) - keep_from > 1 and self._estimate_token_count(history[keep_from:]) > limit:
            keep_from += 1
        if keep_from:
            self.chat_data["message_history"] = history[keep_from:]
            self.chat_data.pop("token_count", None)
            logger.warning("Dropped %s messages to fit the context window", keep_from)
        elif len(history) == 1 and self._estimate_token_count(history) > limit:
            logger.warning("The newest message alone exceeds the context window")

    def _enforce_history_cap(self) -> None:
        """Bound memory when compaction is off or failed."""
        history = self.chat_message_history
        cap = self.settings.compaction_max_messages
        if len(history) > cap:
            self.chat_data["message_history"] = history[-cap:]
            self.chat_data.pop("token_count", None)
            logger.warning("History exceeded %s messages; dropped the oldest ones", cap)

    async def _current_context_messages(self) -> list:
        return self.get_templated_messages().format_messages(**(await self.get_system_template_dict()))

    def save_user_message(self, message: Message) -> bool:
        text = message.text
        if text.startswith("/"):
            text = text.split(maxsplit=1)
            if len(text) > 1:
                text = text[-1]
            else:
                return False
        self.chat_message_history.append(
            HumanMessage(name=message.from_user.username, content=f"{message.from_user.username}: {text}")
        )
        # Messages arrive even when autoreply stays silent, so the cap cannot
        # wait for the reply path.
        self._enforce_history_cap()
        return True

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

    def get_templated_messages(self, system_prompt=None, num_history=None) -> ChatPromptTemplate:
        # logger.info(f"System Prompt: {system_prompt or self.chat_system_prompt}")
        system_prompt = system_prompt or self.chat_system_prompt
        included_history = (
            self.chat_message_history if num_history is None else list(self.chat_message_history)[-num_history:]
        )
        messages = [SystemMessagePromptTemplate.from_template(template=system_prompt)]
        if self.context_summary:
            # A separate message keeps a user-supplied prompt's braces intact.
            messages.append(SystemMessage(content=f"Summary of the earlier conversation:\n{self.context_summary}"))
        messages.extend(included_history)
        return ChatPromptTemplate.from_messages(messages)
