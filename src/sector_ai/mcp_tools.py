"""Opt-in MCP discovery with local policy and environment-only credentials.

The adapter opens/closes a session per invocation, so no task-affine persistent
stdio context managers have to be shared between Telegram handlers.
"""

import asyncio
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4

from dotenv import dotenv_values
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .logging_utils import register_secret_values
from .tools import AgentTool

logger = logging.getLogger(__name__)


class MCPServer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio", "http"]
    command: str = ""
    args: list[str] = Field(default_factory=list)
    url: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_chat_ids: list[int] | None = None
    connect_timeout: float = Field(default=10.0, gt=0, le=120)
    tool_timeout: float = Field(default=60.0, gt=0, le=600)

    @model_validator(mode="after")
    def validate_transport(self):
        if self.transport == "stdio":
            if not self.command or self.url or self.headers:
                raise ValueError("stdio requires command; url/headers are HTTP-only")
        elif not self.url.startswith(("https://", "http://")) or self.command or self.args or self.env:
            raise ValueError("http requires an http(s) url; command/args/env are stdio-only")
        # Credentials may not be pasted into this file. Each value is a complete
        # environment variable reference, optionally prefixed with 'Bearer '.
        for value in [*self.env.values(), *self.headers.values()]:
            if not re.fullmatch(r"(?:Bearer )?\$\{[A-Za-z_][A-Za-z0-9_]*\}", value):
                raise ValueError("env/header values must reference ${ENV_NAME} (or Bearer ${ENV_NAME})")
        return self

    def connection(self) -> dict:
        def resolve(value):
            name = re.search(r"\$\{([^}]+)\}", value).group(1)
            secret = os.environ.get(name, dotenv_values(".env").get(name))
            if not secret:
                raise ValueError(f"MCP environment variable {name} is missing")
            register_secret_values([secret])
            return value.replace("${" + name + "}", secret)

        if self.transport == "stdio":
            return {
                "transport": "stdio",
                "command": self.command,
                "args": self.args,
                "env": {name: resolve(value) for name, value in self.env.items()},
            }
        return {
            "transport": "http",
            "url": self.url,
            "headers": {name: resolve(value) for name, value in self.headers.items()},
        }


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    servers: dict[str, MCPServer] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_names(self):
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,19}", name) for name in self.servers):
            raise ValueError("MCP server names must be 1–20 letters/digits/underscores, starting with a letter")
        return self


class MCPRegistry:
    def __init__(self, config: MCPConfig):
        self.config = config
        self._tools: dict[str, list] = {}
        self._retry_at: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def from_file(cls, filename: str):
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read MCP configuration: {type(error).__name__}") from None
        # Don't echo JSON or Pydantic input values into startup errors.
        try:
            return cls(MCPConfig.model_validate(data))
        except ValueError:
            raise ValueError(
                "Invalid MCP configuration; check transports, allowlists and environment references"
            ) from None

    async def tools_for_chat(self, chat_id: int) -> list[AgentTool]:
        tools = []
        for name, server in self.config.servers.items():
            if not server.allowed_tools or (
                server.allowed_chat_ids is not None and chat_id not in server.allowed_chat_ids
            ):
                continue
            async with self._locks.setdefault(name, asyncio.Lock()):
                if name not in self._tools and monotonic() >= self._retry_at.get(name, 0):
                    try:
                        client = MultiServerMCPClient({name: server.connection()})
                        discovered = await asyncio.wait_for(client.get_tools(server_name=name), server.connect_timeout)
                        allowed = [tool for tool in discovered if tool.name in server.allowed_tools]
                        self._tools[name] = allowed
                        missing = set(server.allowed_tools) - {tool.name for tool in allowed}
                        if missing:
                            logger.warning("MCP server %s did not advertise %s", name, sorted(missing))
                    except Exception:
                        # A bad server must not disable chat or leak its response.
                        logger.warning("MCP server %s unavailable; discovery will retry in 60s", name)
                        self._retry_at[name] = monotonic() + 60
                for tool in self._tools.get(name, []):
                    exposed = f"mcp_{name}__{tool.name}"
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", exposed):
                        logger.warning("MCP server %s advertised an unsupported tool name", name)
                        continue

                    async def invoke(arguments, target=tool):
                        try:
                            response = await target.ainvoke(
                                {
                                    "type": "tool_call",
                                    "id": uuid4().hex,
                                    "name": target.name,
                                    "args": arguments,
                                }
                            )
                        except Exception:
                            # A disconnected remote server may already have
                            # performed a write. Never silently reissue it.
                            return {
                                "status": "execution_uncertain",
                                "error": "MCP transport/execution failed; an external side effect may have occurred. Do not automatically retry.",
                            }
                        if isinstance(response, ToolMessage):
                            # Only text blocks enter history. No base64 blobs or
                            # adapter artifacts with hidden raw resource content.
                            content = response.content
                            if isinstance(content, list):
                                content = [
                                    block
                                    for block in content
                                    if isinstance(block, dict) and block.get("type") == "text"
                                ]
                            result = {"status": response.status, "result": content}
                            artifact = response.artifact
                            if isinstance(artifact, dict) and artifact.get("structured_content") is not None:
                                result["structured_content"] = artifact["structured_content"]
                            return result
                        return {"status": "success", "result": str(response)}

                    tools.append(
                        AgentTool(
                            exposed,
                            tool.description,
                            tool.get_input_schema().model_json_schema()
                            if not isinstance(tool.args_schema, dict)
                            else tool.args_schema,
                            invoke,
                            server.tool_timeout,
                        )
                    )
        counts = Counter(tool.name for tool in tools)
        ambiguous = {name for name, count in counts.items() if count > 1}
        if ambiguous:
            # Keep unambiguous tools available, but never route an ambiguous
            # namespace to whichever server happened to be discovered last.
            logger.error("Ignoring ambiguous MCP tool names: %s", sorted(ambiguous))
        return [tool for tool in tools if tool.name not in ambiguous]
