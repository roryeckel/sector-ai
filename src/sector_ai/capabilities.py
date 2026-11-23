"""Generation, validation and delivery shared by commands and autonomous tools."""

import asyncio
import logging
import re
from dataclasses import dataclass
from io import BytesIO
from xml.etree import ElementTree

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from telegram.constants import PollLimit
from telegram.error import BadRequest, NetworkError

from .chat_metadata import (
    instruction_messages,
    message_metadata,
    metadata_pair,
    render_prompt_message,
    strip_internal_envelope,
)
from .history import plain_history
from .logging_utils import redact_for_user
from .message_text import message_refused, message_text
from .poll_tracking import track_delivered_poll
from .reactions import reaction_tool
from .tools import AgentTool
from .usage import record_usage

logger = logging.getLogger(__name__)


class PollContent(BaseModel):
    question: str = Field(min_length=1, max_length=PollLimit.MAX_QUESTION_LENGTH)
    options: list[str] = Field(min_length=PollLimit.MIN_OPTION_NUMBER, max_length=PollLimit.MAX_OPTION_NUMBER)

    def validate_options(self):
        if any(not option.strip() or len(option) > PollLimit.MAX_OPTION_LENGTH for option in self.options):
            raise ValueError("Poll options must be nonempty and within Telegram's length limit.")
        return self


class TopicContent(BaseModel):
    title: str
    topics: list[str]


@dataclass
class GeneratedOutput:
    kind: str
    content: dict
    caption: str
    png: BytesIO | None = None


def strip_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:svg|xml|html)?\s*\n?(.*?)\n?```", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def validate_svg(source: str) -> None:
    """Do not let the renderer fetch local files, external images or styles."""
    if "<!" in source or "@import" in source.lower():
        raise ValueError("SVG declarations and external styles are not allowed.")
    root = ElementTree.fromstring(source)
    if root.tag.split("}")[-1] != "svg":
        raise ValueError("Expected an SVG root element.")
    for element in root.iter():
        if element.tag.split("}")[-1] in {"image", "script", "foreignObject"}:
            raise ValueError("Embedded images, scripts and foreign objects are not allowed.")
        for name, value in element.attrib.items():
            if name.split("}")[-1] in {"href", "src"} and not value.startswith("#"):
                raise ValueError("External SVG references are not allowed.")
    for reference in re.findall(r"url\s*\((.*?)\)", source, flags=re.IGNORECASE | re.DOTALL):
        if not reference.strip(" \t\r\n'\"").startswith("#"):
            raise ValueError("External SVG URLs are not allowed.")


async def generate_output(kind: str, prompt: str, context, model, *, topic=False) -> GeneratedOutput:
    variables = await context.get_system_template_dict()
    # Generation is a text-only subrequest; never expose a tool-result role
    # without its call, or recursively bind tools to the generator itself.
    history = [render_prompt_message(message) for message in plain_history(context.chat_message_history)]
    shared = [render_prompt_message(message) for message in context.shared_chat_events()]
    if kind == "poll":
        parser = PydanticOutputParser(pydantic_object=TopicContent if topic else PollContent)
        template = context.settings.system_prompts.topic_poll if topic else context.settings.system_prompts.basic_poll
        variables.update(query=prompt, format_instructions=parser.get_format_instructions())
    else:
        parser = None
        template = getattr(context.settings.system_prompts, kind)
        variables["input"] = prompt
    parser_input = "query" if kind == "poll" else "input"
    task_messages = instruction_messages(
        PromptTemplate.from_template(template),
        variables,
        input_key=parser_input,
        input_text=prompt,
    )
    if not any(isinstance(message, HumanMessage) for message in task_messages):
        # Custom or topic prompts may not interpolate the input placeholder.
        task_messages.append(HumanMessage(content=f"Generate the requested {kind}: {prompt}"))
    prefix = []
    remainder = task_messages
    if task_messages and isinstance(task_messages[0], SystemMessage):
        prefix.append(task_messages[0])
        remainder = task_messages[1:]
    static, dynamic = metadata_pair(context)
    if static is not None:
        prefix.append(static)
    if context.context_summary:
        prefix.append(SystemMessage(content=f"Summary of earlier conversation:\n{context.context_summary}"))
    retained = list(history)
    suffix = [*remainder]
    if dynamic is not None:
        suffix.insert(0, dynamic)
    budget = await context.get_context_budget()
    # Keep the task instructions, drop the oldest conversation context first.
    while len(retained) > 1 and context._estimate_token_count([*prefix, *retained, *shared, *suffix]) > budget * 0.65:
        retained.pop(0)
    messages = [*prefix, *retained, *shared, *suffix]
    if context._estimate_token_count(messages) > budget * 0.8:
        raise ValueError("The generation request is too large for the model's context window.")
    workload = f"generate_{kind}" + ("_topic" if topic else "")
    generator = model.bind(response_format={"type": "json_object"}) if topic else model
    generator = context.bind_cache(generator, workload)
    for attempt in range(3 if kind == "svg" else 1):
        response = await generator.ainvoke(messages)
        record_usage(context, workload, response)
        if message_refused(response):
            raise ValueError(message_text(response) or "The model refused the request.")
        source = strip_fence(strip_internal_envelope(message_text(response)))
        if not source:
            raise ValueError("The model returned an empty response.")
        if len(source) > context.settings.tool_result_max_chars // 2:
            raise ValueError("Generated output exceeds the configured tool result size limit.")
        if kind == "poll":
            parsed = parser.parse(source)
            if topic:
                parsed = PollContent(question=parsed.title or "Topics", options=parsed.topics)
            parsed.validate_options()
            return GeneratedOutput(kind, parsed.model_dump(), prompt)
        if kind == "html":
            return GeneratedOutput(kind, {"html": source}, prompt)
        try:
            validate_svg(source)
            from .coding import render_svg_png

            png = await asyncio.to_thread(render_svg_png, source, context.chat_svg_size)
            return GeneratedOutput(kind, {"svg": source}, prompt, png)
        except Exception as error:
            if attempt == 2:
                raise ValueError("Failed to generate valid SVG snippet.") from error
            messages.append(HumanMessage(content=f"Invalid SVG: {error}. Return corrected SVG only."))
    raise ValueError("Generation failed")


def telegram_references(message) -> dict:
    references = {}
    identifier = getattr(message, "message_id", None)
    if type(identifier) is int:
        references["message_id"] = identifier
    timestamp = message_metadata(message).get("sent_at")
    if timestamp:
        references["sent_at"] = timestamp
    poll_id = getattr(getattr(message, "poll", None), "id", None)
    if isinstance(poll_id, str):
        references["poll_id"] = poll_id
    document_id = getattr(getattr(message, "document", None), "file_id", None)
    if isinstance(document_id, str):
        references["file_id"] = document_id
    photos = getattr(message, "photo", None)
    if isinstance(photos, (list, tuple)) and photos:
        file_id = getattr(photos[-1], "file_id", None)
        if isinstance(file_id, str):
            references["file_id"] = file_id
    return references


async def deliver_output(output: GeneratedOutput, update, context) -> dict:
    """One send attempt. A network error can mean Telegram accepted the send."""
    result = {"kind": output.kind, **output.content, "status": "delivery_pending"}
    try:
        message = update.effective_message
        if output.kind == "svg":
            sent = await message.reply_photo(photo=output.png)
        elif output.kind == "html":
            sent = await message.reply_document(
                filename="Website.html",
                caption=output.caption[:1024],
                document=BytesIO(output.content["html"].encode("utf-8")),
            )
        else:
            sent = await message.reply_poll(
                output.content["question"],
                output.content["options"],
                is_anonymous=False,
            )
            track_delivered_poll(sent, update, context)
        result.update(status="delivered", telegram=telegram_references(sent))
    except asyncio.CancelledError:
        # Persist this before propagating a run-level timeout/cancellation.
        result.update(status="delivery_uncertain", error="Delivery was interrupted; do not automatically resend.")
        context.conversation_data["interrupted_delivery"] = {
            "tool_call_id": context.conversation_data.get("active_tool_call_id"),
            "result": result,
        }
        raise
    except BadRequest as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram %s delivery failed: %s", output.kind, message)
        result.update(status="delivery_failed", error=message)
    except NetworkError as error:
        logger.warning(
            "Telegram %s delivery was uncertain: %s",
            output.kind,
            redact_for_user(str(error), context.settings),
        )
        result.update(status="delivery_uncertain", error="Telegram delivery is uncertain; do not automatically resend.")
    except Exception as error:
        message = redact_for_user(str(error), context.settings)
        logger.warning("Telegram %s delivery failed: %s", output.kind, message)
        result.update(status="delivery_failed", error=message)
    return result


def builtin_tools(update, context, model, *, topic=False) -> list[AgentTool]:
    tools = []
    for name, kind, description in [
        (
            "generate_svg",
            "svg",
            "Generate an SVG illustration and send its rendered PNG to this chat. Include relevant prior SVG source in the prompt when revising.",
        ),
        (
            "generate_html",
            "html",
            "Generate an HTML page and send it as a document to this chat. Include relevant prior HTML when revising.",
        ),
        (
            "create_poll",
            "poll",
            "Generate a poll question and options from a prompt, and post a non-anonymous Telegram poll to this chat.",
        ),
    ]:

        async def invoke(arguments, output_kind=kind):
            output = await generate_output(
                output_kind, arguments["prompt"], context, model, topic=topic and output_kind == "poll"
            )
            return await deliver_output(output, update, context)

        tools.append(
            AgentTool(
                name,
                description,
                {
                    "type": "object",
                    "properties": {"prompt": {"type": "string", "minLength": 1, "maxLength": 16000}},
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
                invoke,
                context.settings.tool_timeout,
            )
        )
    tools.append(reaction_tool(update, context))
    return tools
