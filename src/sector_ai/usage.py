"""Bounded provider usage accounting for cache and latency telemetry.

Only counters and identifiers are retained. Prompt contents, tool arguments and
credentials are never logged here.
"""

import logging
from time import time

logger = logging.getLogger(__name__)

FIELDS = ("input_tokens", "output_tokens", "cache_read", "cache_write")
MAX_STATS = 32


def usage_details(message) -> dict:
    """Extract normalized usage fields from a message or streamed chunk.

    ``langchain-openai`` exposes cache reads/writes under
    ``usage_metadata.input_token_details``; some compatible servers only put the
    same counters in ``response_metadata``. Both spellings are accepted.
    """
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("input_token_details")
    if not isinstance(details, dict):
        details = {}
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    prompt_details = metadata.get("prompt_tokens_details") or metadata.get("input_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    values = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read": details.get("cache_read", prompt_details.get("cached_tokens")),
        "cache_write": details.get("cache_creation", prompt_details.get("cache_write_tokens")),
    }
    return {key: value for key, value in values.items() if type(value) is int and value >= 0}


def record_usage(context, workload: str, message) -> dict | None:
    """Accumulate one response's counters under model and workload.

    Returns the normalized details so callers can log them, or ``None`` when the
    backend reported no usage.
    """
    details = usage_details(message)
    if not details:
        return None
    try:
        model = context.get_model()
    except Exception:
        model = getattr(message, "response_metadata", {}).get("model", "unknown")
    key = (str(model), str(workload))
    stats = context.bot_data.setdefault("usage_stats", {})
    if not isinstance(stats, dict):
        stats = {}
        context.bot_data["usage_stats"] = stats
    if len(stats) >= MAX_STATS and key not in stats:
        stats.pop(next(iter(stats)))
    entry = stats.setdefault(
        key,
        {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
            "cache_requests": 0,
            "first_at": time(),
            "last_at": time(),
        },
    )
    entry["requests"] += 1
    for field in FIELDS:
        value = details.get(field)
        if isinstance(value, int):
            entry[field] += value
    if "cache_read" in details or "cache_write" in details:
        entry["cache_requests"] += 1
    entry["last_at"] = time()
    logger.debug("Model usage model=%s workload=%s %s", model, workload, details)
    return details
