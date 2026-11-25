import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from sector_ai.logging_utils import redact_for_user, register_secret_values
from sector_ai.mcp_tools import MCPConfig, MCPRegistry, MCPServer
from sector_ai.tools import result_content


def test_credentials_are_environment_references_only():
    with pytest.raises(ValidationError):
        MCPServer(transport="http", url="http://localhost/mcp", headers={"Authorization": "Bearer literal-secret"})


def test_environment_credentials_are_resolved_and_redacted(monkeypatch, make_settings):
    monkeypatch.setenv("SECTOR_TEST_MCP_TOKEN", "test-mcp-credential-not-a-real-secret")
    server = MCPServer(
        transport="http",
        url="https://example.org/mcp",
        headers={
            "Authorization": "Bearer ${SECTOR_TEST_MCP_TOKEN}",
        },
    )
    assert server.connection()["headers"]["Authorization"] == "Bearer test-mcp-credential-not-a-real-secret"
    assert "test-mcp-credential-not-a-real-secret" not in redact_for_user(
        "error: test-mcp-credential-not-a-real-secret",
        make_settings(),
    )


@pytest.mark.parametrize(
    "data",
    [
        {"transport": "stdio"},
        {"transport": "http", "url": "file:///private/file"},
        {"transport": "http", "url": "http://example.org", "command": "python"},
        {"transport": "stdio", "command": "python", "unexpected": True},
    ],
)
def test_invalid_server_configuration_is_rejected(data):
    with pytest.raises(ValidationError):
        MCPServer.model_validate(data)


async def test_empty_allowlist_and_chat_restrictions_skip_discovery(monkeypatch):
    discovery = AsyncMock()
    monkeypatch.setattr("sector_ai.mcp_tools.MultiServerMCPClient.get_tools", discovery)
    config = MCPConfig(
        servers={
            "disabled": MCPServer(transport="http", url="http://localhost/mcp"),
            "private": MCPServer(
                transport="http", url="http://localhost/mcp", allowed_tools=["read"], allowed_chat_ids=[99]
            ),
        }
    )
    assert await MCPRegistry(config).tools_for_chat(1) == []
    discovery.assert_not_awaited()


async def test_server_failure_does_not_disable_other_tools(monkeypatch):
    discovery = AsyncMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr("sector_ai.mcp_tools.MultiServerMCPClient.get_tools", discovery)
    registry = MCPRegistry(
        MCPConfig(
            servers={
                "offline": MCPServer(transport="http", url="http://localhost/mcp", allowed_tools=["read"]),
            }
        )
    )
    assert await registry.tools_for_chat(1) == []
    assert await registry.tools_for_chat(1) == []
    assert discovery.await_count == 1


async def test_real_stdio_discovery_execution_and_allowlist():
    registry = MCPRegistry(
        MCPConfig(
            servers={
                "math": MCPServer(
                    transport="stdio",
                    command=sys.executable,
                    args=[str(Path(__file__).parent / "fixtures" / "mcp_server.py")],
                    allowed_tools=["add"],
                    connect_timeout=20,
                ),
            }
        )
    )
    tools = await registry.tools_for_chat(1)
    assert [tool.name for tool in tools] == ["mcp_math__add"]
    result = await tools[0].invoke({"a": 3, "b": 5})
    assert result["status"] == "success"
    assert "8" in str(result)


def test_invalid_config_does_not_echo_sensitive_input(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"bad": {"transport": "wrong", "token": "private-test-value"}}}))
    with pytest.raises(ValueError) as error:
        MCPRegistry.from_file(str(path))
    assert "private-test-value" not in str(error.value)


def test_large_tool_result_is_bounded_valid_json(make_settings):
    settings = make_settings(tool_result_max_chars=1024)
    text = result_content({"status": "success", "data": '\x00"\\' * 10000}, settings)
    assert len(text) <= 1024
    assert json.loads(text)["truncated"] is True


@pytest.mark.parametrize(
    "secret",
    [
        'test-credential-"quoted"',
        "test-credential-with\\backslash",
        "test-credential-with\nmultiple\nlines",
    ],
)
def test_escaped_credentials_are_redacted_before_json_encoding(secret, make_settings):
    register_secret_values([secret])
    result = {
        "status": "success",
        "result": secret,
        "structured_content": {"nested": [secret, {secret: f"error: {secret}"}]},
    }
    decoded = json.loads(result_content(result, make_settings()))
    assert decoded["result"] == "<redacted>"
    assert decoded["structured_content"]["nested"] == ["<redacted>", {"<redacted>": "error: <redacted>"}]


def test_fastmcp_json_text_blocks_redact_escaped_credentials(make_settings):
    from mcp.server.fastmcp.utilities.func_metadata import _convert_to_content

    secret = 'synthetic-mcp-credential-"with\\escapes\nand-newlines'
    register_secret_values([secret])
    blocks = [block.model_dump() for block in _convert_to_content({"credential": secret})]
    decoded = json.loads(result_content({"status": "success", "result": blocks}, make_settings()))
    assert json.loads(decoded["result"][0]["text"])["credential"] == "<redacted>"


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_json_encoded_credentials_are_redacted_in_text_payloads(depth, ensure_ascii, make_settings):
    secret = 'synthetic-unicode-credential-é-"with\\escapes\n'
    register_secret_values([secret])
    payload = {"credential": secret}
    for _ in range(depth):
        payload = json.dumps(payload, ensure_ascii=ensure_ascii)
    decoded = json.loads(result_content({"status": "success", "result": payload}, make_settings()))["result"]
    for _ in range(depth):
        decoded = json.loads(decoded)
    assert decoded["credential"] == "<redacted>"


async def test_mcp_namespace_collision_rejects_both_ambiguous_tools(monkeypatch):
    sources = {
        "a": [SimpleNamespace(name="b__read", description="first", args_schema={}, ainvoke=AsyncMock())],
        "a__b": [
            SimpleNamespace(name="read", description="second", args_schema={}, ainvoke=AsyncMock()),
            SimpleNamespace(name="safe", description="safe", args_schema={}, ainvoke=AsyncMock()),
        ],
    }

    async def discover(*, server_name):
        return sources[server_name]

    monkeypatch.setattr("sector_ai.mcp_tools.MultiServerMCPClient.get_tools", AsyncMock(side_effect=discover))
    registry = MCPRegistry(
        MCPConfig(
            servers={
                "a": MCPServer(transport="http", url="http://localhost/mcp", allowed_tools=["b__read"]),
                "a__b": MCPServer(transport="http", url="http://localhost/mcp", allowed_tools=["read", "safe"]),
            }
        )
    )
    tools = await registry.tools_for_chat(1)
    assert [tool.name for tool in tools] == ["mcp_a__b__safe"]
    for tool in sources["a"] + sources["a__b"]:
        tool.ainvoke.assert_not_awaited()
