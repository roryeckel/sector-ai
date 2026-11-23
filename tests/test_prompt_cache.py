"""Prefix stability, backend cache selection and usage accounting."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from sector_ai.admin import usage_cmd
from sector_ai.chat_metadata import instruction_messages, with_context
from sector_ai.tools import AgentTool, canonical_schema
from sector_ai.usage import record_usage, usage_details


def test_clock_only_change_keeps_the_retained_prefix(make_context, make_update):
    context = make_context("unused")
    context.chat_message_history = [HumanMessage(name="alice", content=f"alice: message {i}") for i in range(6)]
    context.context_summary = "Earlier decisions are unchanged."
    update = make_update(text="latest question")
    context.bind_update(update)

    first = context.get_templated_messages(tool_history=True).format_messages(USERNAME="bot", MAX_LEN=4096)
    context._context_time = "2030-01-01T00:00:00Z"
    second = context.get_templated_messages(tool_history=True).format_messages(USERNAME="bot", MAX_LEN=4096)

    assert isinstance(first[-1], SystemMessage)
    assert "current_time_utc" in first[-1].content
    assert first[:-1] == second[:-1]
    assert first[-1] != second[-1]
    assert "2030-01-01T00:00:00Z" in second[-1].content


def test_instruction_template_splits_task_text_from_instructions():
    template = PromptTemplate.from_template("Static {MAX_LEN}. Specification: {input}")
    messages = instruction_messages(
        template,
        {"MAX_LEN": 4096, "input": "draw a cat"},
        input_text="draw a cat",
    )

    assert [type(message) for message in messages] == [SystemMessage, HumanMessage]
    assert "4096" in messages[0].content
    assert messages[1].content == "draw a cat"


def test_instruction_template_keeps_a_trailing_system_suffix():
    template = PromptTemplate.from_template("Before {input}. After.")
    messages = instruction_messages(
        template,
        {"input": "body"},
        input_text="body",
    )

    assert [type(message) for message in messages] == [SystemMessage, HumanMessage, SystemMessage]
    assert messages[0].content == "Before"
    assert messages[1].content == "body"
    assert messages[2].content == ". After."


def test_with_context_puts_the_snapshot_before_the_task(make_context, make_update):
    context = make_context("unused")
    context.bind_update(make_update(text="hello"))
    messages = with_context([SystemMessage(content="instructions"), HumanMessage(content="input")], context)

    assert [type(message) for message in messages] == [SystemMessage, SystemMessage, SystemMessage, HumanMessage]
    assert "observed data" in messages[1].content
    assert "current_time_utc" in messages[2].content
    assert messages[-1].content == "input"


def test_cache_backend_auto_recognizes_known_hosts(make_context, make_settings):
    openai = make_context(
        "unused",
        settings=make_settings(
            llm_provider="openai",
            openai_base_url="https://api.openai.com/v1",
        ),
    )
    openrouter = make_context(
        "unused",
        settings=make_settings(
            llm_provider="openai",
            openai_base_url="https://openrouter.ai/api/v1",
        ),
    )
    local = make_context(
        "unused",
        settings=make_settings(
            llm_provider="openai",
            openai_base_url="http://localhost:1234/v1",
        ),
    )

    assert openai.cache_backend() == "openai"
    assert openrouter.cache_backend() == "openrouter"
    assert local.cache_backend() == "none"
    assert local.cache_request_options("agent") == {}
    assert openai.cache_request_options("agent")["prompt_cache_key"].endswith(":agent")
    openrouter_options = openrouter.cache_request_options("agent")["extra_body"]
    assert openrouter_options["session_id"].endswith(":agent")
    assert openrouter_options["cache_control"] == {"type": "ephemeral"}


def test_cache_controls_can_be_forced_or_disabled(make_context, make_settings):
    forced = make_context(
        "unused",
        settings=make_settings(
            llm_provider="openai",
            openai_base_url="http://localhost:1234/v1",
            prompt_cache_mode="openai",
            prompt_cache_retention="24h",
            prompt_cache_ttl="30m",
        ),
    )
    disabled = make_context(
        "unused",
        settings=make_settings(
            llm_provider="openai",
            openai_base_url="https://api.openai.com/v1",
            prompt_cache_mode="off",
        ),
    )

    options = forced.cache_request_options("agent")
    assert options["prompt_cache_retention"] == "24h"
    assert options["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert disabled.cache_request_options("agent") == {}


def test_openrouter_can_extend_automatic_cache_lifetime(make_context, make_settings):
    context = make_context(
        "unused",
        settings=make_settings(
            llm_provider="openai",
            openai_base_url="https://openrouter.ai/api/v1",
            prompt_cache_mode="openrouter",
            prompt_cache_ttl="1h",
        ),
    )

    extra = context.cache_request_options("agent")["extra_body"]
    assert extra["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_openai_serializes_cache_affinity_without_prompt_content(make_settings):
    model = ChatOpenAI(
        model="gpt-4.1",
        api_key="dummy",
        base_url="https://api.openai.com/v1",
        max_retries=0,
    )
    payload = model._get_request_payload(
        [HumanMessage(content="hello")],
        prompt_cache_key="sector-ai:1:0:agent",
    )
    openrouter = model._get_request_payload(
        [HumanMessage(content="hello")],
        extra_body={"session_id": "sector-ai:1:0:agent"},
    )

    assert payload["prompt_cache_key"] == "sector-ai:1:0:agent"
    assert payload["messages"][0]["content"] == "hello"
    assert openrouter["extra_body"] == {"session_id": "sector-ai:1:0:agent"}


def test_tool_schemas_are_canonical_and_order_independent():
    first = AgentTool(
        "demo",
        "Demo",
        {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}}},
        None,
        1,
    )
    second = AgentTool(
        "demo",
        "Demo",
        {"type": "object", "properties": {"a": {"type": "integer"}, "z": {"type": "string"}}},
        None,
        1,
    )

    assert first.schema == second.schema
    assert list(first.schema["function"]["parameters"]["properties"]) == ["a", "z"]
    assert canonical_schema({"b": 1, "a": {"d": 2, "c": 3}}) == {"a": {"c": 3, "d": 2}, "b": 1}


def test_usage_details_reads_openai_and_compatible_shapes():
    native = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_token_details": {"cache_read": 4, "cache_creation": 3},
        },
    )
    compatible = AIMessage(
        content="",
        response_metadata={"prompt_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 1}},
    )

    assert usage_details(native) == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read": 4,
        "cache_write": 3,
    }
    assert usage_details(compatible)["cache_read"] == 5
    assert usage_details(compatible)["cache_write"] == 1


def test_usage_summary_accumulates_and_computes_hit_rate(make_context):
    context = make_context("unused")
    message = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "input_token_details": {"cache_read": 40, "cache_creation": 20},
        },
    )

    record_usage(context, "agent", message)
    record_usage(context, "agent", message)
    row = context.usage_summary()[0]

    assert row["requests"] == 2
    assert row["input_tokens"] == 200
    assert row["cache_read"] == 80
    assert row["cache_write"] == 40
    assert round(row["hit_rate"], 1) == 40.0


async def test_usage_command_reports_counters_without_prompt_content(make_context, make_settings, make_update):
    context = make_context("unused", settings=make_settings(admin_usernames=["testuser"]))
    message = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 20,
            "output_tokens": 2,
            "total_tokens": 22,
            "input_token_details": {"cache_read": 10},
        },
    )
    record_usage(context, "agent", message)
    update = make_update(text="/usage")

    await usage_cmd(update, context)

    text = update.message.texts("reply_text")[-1]
    assert "fake-model" in text
    assert "agent:" in text
    assert "10 read" in text
    assert "unused" not in text
