"""Text extraction that also surfaces refusals from block-based responses."""

from langchain_core.messages import BaseMessage


def _content_blocks(message: BaseMessage) -> list:
    blocks = getattr(message, "content_blocks", None)
    if blocks is not None:
        return blocks
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [{"type": "text", "text": part} if isinstance(part, str) else part for part in content]
    return []


def message_text(message: BaseMessage) -> str:
    """Join text and refusal blocks into the text a user should see."""
    parts = []
    for block in _content_blocks(message):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "refusal":
            parts.append(str(block.get("refusal") or ""))
    return "".join(parts)


def message_refused(message: BaseMessage) -> bool:
    return any(isinstance(block, dict) and block.get("type") == "refusal" for block in _content_blocks(message))
