import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sector_ai.history import history_text, plain_history, safe_cut


def transaction():
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_poll",
                "args": {"prompt": "Pick a color"},
                "id": "one",
                "type": "tool_call",
            }
        ],
    )
    result = ToolMessage(
        name="create_poll",
        content=json.dumps(
            {
                "question": "Color?",
                "options": ["Red", "Blue"],
                "status": "delivered",
            }
        ),
        tool_call_id="one",
    )
    return [call, result]


def test_safe_cut_keeps_call_with_result():
    messages = [HumanMessage(content="earlier"), *transaction()]
    assert safe_cut(messages, 2) == 1


def test_projection_preserves_arguments_and_results_without_tool_roles():
    projected = plain_history(transaction())
    assert all(isinstance(message, AIMessage) for message in projected)
    assert "Pick a color" in projected[0].content
    assert "Red" in projected[1].content


async def test_compaction_retains_whole_tool_group(make_context, make_settings):
    context = make_context("summary", settings=make_settings(context_length=8192, compaction_keep_messages=1))
    context.chat_message_history = [HumanMessage(content="earlier"), *transaction()]
    await context.compact()
    assert [type(message) for message in context.chat_message_history] == [AIMessage, ToolMessage]


async def test_compaction_summarizes_arguments_and_exact_poll_options(make_context, make_settings):
    context = make_context(
        "The poll offered Red and Blue.",
        settings=make_settings(
            context_length=8192,
            compaction_keep_messages=1,
        ),
    )
    context.chat_message_history = [*transaction(), HumanMessage(content="after")]
    await context.compact()
    transcript = context.bot_llm.received[0][-1].content
    assert "Pick a color" in transcript
    assert "Red" in transcript and "Blue" in transcript


def test_tool_argument_changes_invalidate_token_cache(make_context):
    context = make_context()
    messages = transaction()
    before = context._token_count_fingerprint(messages)
    messages[0].tool_calls[0]["args"]["prompt"] = "Pick a number"
    assert context._token_count_fingerprint(messages) != before
    assert "Pick a number" in history_text(messages[0])


def test_tool_schemas_are_included_in_token_estimates(make_context):
    context = make_context()
    messages = [HumanMessage(content="hi")]
    before = context._estimate_token_count(messages, include_tools=True)
    context.conversation_data["tool_schemas"] = [{"description": "long tool documentation " * 100}]
    assert context._estimate_token_count(messages, include_tools=True) > before + 100


def test_history_cap_does_not_orphan_tool_results(make_context, make_settings):
    context = make_context(settings=make_settings(compaction_max_messages=2))
    context.chat_message_history = [HumanMessage(content="old"), *transaction(), HumanMessage(content="new")]
    context._enforce_history_cap()
    assert isinstance(context.chat_message_history[0], AIMessage)
    assert context.chat_message_history[0].tool_calls
