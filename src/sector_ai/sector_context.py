import asyncio
import logging
from collections import deque
from typing import Optional

import httpx
from langchain_core.messages import ChatMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_ollama.chat_models import ChatOllama
from telegram import Message
from telegram.constants import MessageLimit
from telegram.ext import Application, CallbackContext

from .settings import Settings

logger = logging.getLogger(__name__)


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
    def bot_ollama(self) -> ChatOllama:
        try:
            return self.bot_data["ollama"]
        except KeyError:
            return self.load_model()

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
    def chat_message_history(self) -> deque:
        try:
            return self.chat_data["message_history"]
        except KeyError:
            self.chat_data["message_history"] = deque(maxlen=15)
            return self.chat_data["message_history"]

    @chat_message_history.setter
    def chat_message_history(self, new_history: deque) -> None:
        self.chat_data["message_history"] = new_history

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

    def load_model(self, model_name=None) -> ChatOllama:
        result = ChatOllama(
            model=model_name or self.settings.default_model,
            base_url=self.settings.ollama_url,
            num_ctx=self.settings.ollama_context_length,
            client_kwargs={"timeout": self.settings.ollama_timeout, "headers": self.settings.ollama_headers},
        )
        self.bot_data["ollama"] = result
        return result

    def get_model(self) -> str:
        return self.bot_ollama.model

    async def get_model_details(self) -> dict:
        async with httpx.AsyncClient(headers=self.settings.ollama_headers) as client:
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
        async with httpx.AsyncClient(headers=self.settings.ollama_headers) as client:
            response = await client.get(f"{self.settings.ollama_url}/api/tags")
            if not response.is_success:
                return []

            result = [r for r in response.json()["models"] if r["model"] not in self.settings.ollama_disallowed_models]
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

            r["is_default_active"] = r["model"] == self.get_model()
            r["is_vision_active"] = r["model"] == self.settings.vision_model
            r["label"] = (
                f"{'✅ ' if r['is_default_active'] else ''}{'👁️‍🗨️ ' if r['is_vision_active'] else ''}{r['model']}"
            )
        return result

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
        return ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(template=system_prompt), *included_history]
        )
