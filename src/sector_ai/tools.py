"""Small common tool contract; native capabilities do not need an MCP server."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .logging_utils import redact_for_user


def canonical_schema(value):
    """Sort object keys so discovery order cannot churn the tool prefix."""
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            item = value[key]
            if key == "required" and isinstance(item, list) and all(isinstance(entry, str) for entry in item):
                item = sorted(item)
            result[key] = canonical_schema(item)
        return result
    if isinstance(value, list):
        return [canonical_schema(item) for item in value]
    return value


@dataclass
class AgentTool:
    name: str
    description: str
    parameters: dict
    invoke: Callable[[dict], Awaitable[dict]]
    timeout: float
    #: The tool sets ``conversation_data["side_effect_started"]`` to the active
    #: call ID right before its side effect, so a timeout earlier than that is
    #: known to have changed nothing.
    marks_side_effects: bool = False

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": canonical_schema(self.parameters),
            },
        }


def result_content(result: dict, settings) -> str:
    """Bound external output without leaving invalid JSON or leaking credentials."""

    def redact_text(value):
        return redact_for_user(value, settings, json_escaped=True)

    def redact_value(value):
        # Redact before JSON escaping: a quote, backslash or newline in a
        # credential otherwise prevents matching its registered plaintext.
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, dict):
            return {redact_text(str(key)): redact_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [redact_value(item) for item in value]
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return redact_text(str(value))

    redacted = redact_value(result)
    text = json.dumps(redacted, ensure_ascii=False)
    limit = settings.tool_result_max_chars
    if len(text) <= limit:
        return text
    # Keep the short fields that say what happened (the delivery note, the
    # Telegram IDs, the error) intact, and cut only the bulky rest.
    kept = {key: redacted[key] for key in RECEIPT_KEYS if isinstance(redacted, dict) and key in redacted}
    if len(json.dumps(kept, ensure_ascii=False)) > limit // 2:
        kept = {}
    content = {"status": redacted.get("status", "success"), "truncated": True, **kept, "result": text[: limit // 2]}
    text = json.dumps(content, ensure_ascii=False)
    # Escaping can expand arbitrary external strings to six characters each.
    while len(text) > limit and content["result"]:
        content["result"] = content["result"][: len(content["result"]) // 2]
        text = json.dumps(content, ensure_ascii=False)
    return text


#: Receipt fields that survive truncation of a long tool result.
RECEIPT_KEYS = ("kind", "telegram", "note", "error", "target", "emoji", "message_id")
