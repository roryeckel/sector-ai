from typing import Optional
import requests
from telegram import Message
from telegram.ext import Application, CallbackContext
from telegram.constants import MessageLimit
from collections import deque
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage, ChatMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

class SectorContext(CallbackContext):
    def __init__(self, application: Application, chat_id: Optional[int] = None, user_id: Optional[int] = None):
        super().__init__(application=application, chat_id=chat_id, user_id=user_id)
        self._username = None

    # Ollama Config

    @property
    def config_ollama_url(self) -> str:
        try:
            return self.bot_config['ollama']['url']
        except KeyError:
            return 'http://localhost:11434'

    @property
    def config_ollama_timeout(self) -> int:
        return self.bot_config['ollama']['timeout']
    
    @property
    def config_ollama_headers(self) -> dict:
        return self.bot_config['ollama']['headers']
    
    @property
    def config_ollama_disallowed_models(self) -> list:
        return self.bot_config['ollama']['disallowed_models']

    # System Prompt Config

    @property
    def config_default_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['default']
    
    @property
    def config_summarization_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['summarization']
    
    @property
    def config_emoji_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['emoji']
    
    @property
    def config_basic_poll_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['basic_poll']
    
    @property
    def config_topic_poll_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['topic_poll']
    
    @property
    def config_characterizer_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['characterizer']
    
    @property
    def config_code_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['code']
    
    @property
    def config_html_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['html']
    
    @property
    def config_svg_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['svg']
    
    @property
    def config_decide_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['decide']
    
    @property
    def config_decide_autoreply_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['decide_autoreply']
    
    @property
    def config_vision_system_prompt(self) -> str:
        return self.bot_config['system_prompts']['vision']
    
    # Streaming Config

    @property
    def config_streaming_cursor(self) -> str:
        return self.bot_config['streaming']['cursor']
    
    @property
    def config_streaming_interval_sec(self) -> int:
        return self.bot_config['streaming']['interval_sec']
    
    @property
    def config_streaming_chunk_size(self) -> int:
        return self.bot_config['streaming']['chunk_size']

    # Misc Config

    @property
    def config_default_svg_size(self) -> int:
        return self.bot_config['default_svg_size']
    
    @property
    def config_default_model(self) -> str:
        return self.bot_config['default_model']
    
    @property
    def config_vision_model(self) -> str:
        return self.bot_config['vision_model']
    
    @property
    def config_admin_usernames(self) -> list:
        return self.bot_config['admin_usernames']
    
    # Bot

    @property
    def bot_config(self) -> dict:
        return self.bot_data['config']
    
    @bot_config.setter
    def bot_config(self, new_config: dict) -> None:
        self.bot_data['config'] = new_config

    @property
    def bot_ollama(self) -> Ollama:
        try:
            return self.bot_data['ollama']
        except KeyError:
            return self.load_model()
        
    # Chat

    @property
    def chat_system_prompt(self) -> str:
        try:
            return self.chat_data['system_prompt']
        except KeyError:
            return self.config_default_system_prompt
    
    @chat_system_prompt.setter
    def chat_system_prompt(self, new_prompt: str) -> None:
        self.chat_data['system_prompt'] = new_prompt

    @property
    def chat_autoreply_mode(self) -> bool:
        try:
            return self.chat_data['autoreply_mode']
        except KeyError:
            self.chat_data['autoreply_mode'] = True
            return True
        
    @chat_autoreply_mode.setter
    def chat_autoreply_mode(self, mode: bool) -> None:
        self.chat_data['autoreply_mode'] = mode

    @property
    def chat_message_history(self) -> deque:
        try:
            return self.chat_data['message_history']
        except KeyError:
            self.chat_data['message_history'] = deque(maxlen=15)
            return self.chat_data['message_history']
    
    @chat_message_history.setter
    def chat_message_history(self, new_history: deque) -> None:
        self.chat_data['message_history'] = new_history

    @property
    def chat_svg_size(self) -> int:
        try:
            return self.chat_data['svg_size']
        except KeyError:
            default_svg_size = self.config_default_svg_size
            self.chat_data['svg_size'] = default_svg_size
            return default_svg_size
        
    @chat_svg_size.setter
    def chat_svg_size(self, size: int) -> None:
        self.chat_data['svg_size'] = size

    # Methods

    async def get_username(self) -> str:
        if self._username:
            return self._username
        me = await self.application.bot.get_me()
        self._username = me.username
        return self._username

    def load_model(self, model_name=None) -> Ollama:
        result = Ollama(
            model=model_name or self.config_default_model,
            base_url=self.config_ollama_url,
            timeout=self.config_ollama_timeout,
            headers=self.config_ollama_headers)
        self.bot_data['ollama'] = result
        return result
    
    def get_model(self) -> str:
        return self.bot_ollama.model
    
    def get_model_details(self) -> dict:
        response = requests.post(
            f"{self.config_ollama_url}/api/show",
            headers=self.config_ollama_headers,
            json={"name": self.get_model()}
        )
        if response.ok:
            result = response.json()
            result.pop("license", None)
            result.pop("modelfile", None)
            result.pop("parameters", None)
            result.pop("template", None)
            return result
        else:
            return {}

    def get_models(self) -> list:
        response = requests.get(f"{self.config_ollama_url}/api/tags", headers=self.config_ollama_headers)
        if response.ok:
            data = response.json()
            result = data["models"]
            result = [r for r in result if r["model"] not in self.config_ollama_disallowed_models]
            
            for r in result:
                show_response = requests.post(
                    f"{self.config_ollama_url}/api/show",
                    headers=self.config_ollama_headers,
                    json={"name": r["model"]}
                )
                
                if show_response.ok:
                    show_data = show_response.json()
                    r.update(show_data)
                
                r["is_default_active"] = r["model"] == self.get_model()
                r["is_vision_active"] = r["model"] == self.config_vision_model
                r["label"] = f"{'✅ ' if r['is_default_active'] else ''}{'👁️‍🗨️ ' if r['is_vision_active'] else ''}{r['model']}"
            return result
        else:
            return []

    def save_user_message(self, message: Message) -> bool:
        text = message.text
        if text.startswith('/'):
            text = text.split(maxsplit=1)
            if len(text) > 1:
                text = text[-1]
            else:
                return False
        self.chat_message_history.append(HumanMessage(name=message.from_user.username, content=f'{message.from_user.username}: {text}'))
        return True
    
    def message_exists(self, new_message: ChatMessage) -> bool:
        return any(
            existing_message.content == new_message.content and
            existing_message.type == new_message.type
            for existing_message in self.chat_message_history
        )
    
    async def get_system_template_dict(self) -> dict:
        username = await self.get_username()
        return {
            'MAX_LEN': MessageLimit.MAX_TEXT_LENGTH,
            'USERNAME': username,
        }
    
    def get_templated_messages(self, system_prompt = None, num_history = None) -> ChatPromptTemplate:
        # logger.info(f"System Prompt: {system_prompt or self.chat_system_prompt}")
        system_prompt = system_prompt or self.chat_system_prompt
        included_history = self.chat_message_history if num_history is None else self.chat_message_history[-num_history:]
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(template=system_prompt), *included_history])
        