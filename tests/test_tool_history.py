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


def test_a_truncated_receipt_keeps_its_note_and_ids(make_settings):
    import json as _json

    from sector_ai.tools import result_content

    settings = make_settings(tool_result_max_chars=2000)
    result = {
        "kind": "html",
        "html": "<p>" + '"x"' * 3000 + "</p>",
        "status": "delivered",
        "telegram": {"message_id": 5},
        "note": "Delivered as the Website.html document in this chat. Do not repeat or restate its contents.",
    }

    text = result_content(result, settings)
    content = _json.loads(text)

    assert len(text) <= 2000
    assert content["truncated"] is True
    assert content["status"] == "delivered"
    assert content["telegram"] == {"message_id": 5}
    assert content["note"] == result["note"]


def test_svg_validation_allows_comments_and_cdata_but_not_declarations():
    import pytest as _pytest

    from sector_ai.capabilities import strip_fence, validate_svg

    validate_svg(
        '<svg xmlns="http://www.w3.org/2000/svg"><!-- note --><style><![CDATA[rect{fill:red}]]></style>'
        '<rect width="1" height="1"/></svg>'
    )
    for declaration in ("<!DOCTYPE svg>", '<!ENTITY x SYSTEM "file:///etc/passwd">', "<!doctype svg>"):
        with _pytest.raises(ValueError):
            validate_svg(declaration + '<svg xmlns="http://www.w3.org/2000/svg"/>')
    assert strip_fence("```HTML\n<p>x</p>\n```") == "<p>x</p>"
    assert strip_fence("```\n<svg/>\n```") == "<svg/>"


def test_poll_votes_can_name_all_twelve_options():
    from types import SimpleNamespace

    from sector_ai.poll_tracking import poll_option_snapshot

    poll = SimpleNamespace(options=[SimpleNamespace(text=f"option {n}", voter_count=0) for n in range(12)])

    assert len(poll_option_snapshot(poll)) == 12


def test_clip_utf16_counts_emoji_twice_and_never_splits_one():
    from sector_ai.telegram_format import clip_utf16, utf16_len

    text = "🔥" * 3000
    clipped = clip_utf16(text, 4096)

    assert utf16_len(clipped) <= 4096
    assert clipped == "🔥" * 2048
    assert clip_utf16("👨‍👩", 3) == "👨"


def test_svg_validation_rejects_every_dangerous_form():
    import pytest

    from sector_ai.capabilities import validate_svg

    for source in [
        "<svg><style>@import url(x.css)</style></svg>",
        "<html/>",
        '<svg><use href="http://example.com/a.svg"/></svg>',
        '<svg><rect fill="url(http://example.com/a.svg)"/></svg>',
    ]:
        with pytest.raises(ValueError):
            validate_svg(source)


def test_telegram_references_reads_documents_and_photos():
    from types import SimpleNamespace

    from sector_ai.capabilities import telegram_references

    document = SimpleNamespace(message_id=3, document=SimpleNamespace(file_id="doc"), photo=[])
    photo = SimpleNamespace(message_id=4, document=None, photo=[SimpleNamespace(file_id="pic")])

    assert telegram_references(document)["file_id"] == "doc"
    assert telegram_references(photo)["file_id"] == "pic"
    assert telegram_references(SimpleNamespace(message_id=5))["message_id"] == 5


async def test_a_generic_send_failure_is_reported(make_context, make_update):
    from unittest.mock import AsyncMock

    from sector_ai.capabilities import GeneratedOutput, deliver_output

    context = make_context()
    update = make_update(text="hi")
    update.effective_message.reply_photo = AsyncMock(side_effect=RuntimeError("boom"))

    result = await deliver_output(GeneratedOutput("svg", {"svg": "<svg/>"}, "", None), update, context)

    assert result["status"] == "delivery_failed"
    assert "boom" in result["error"]


async def test_a_bookkeeping_failure_still_reports_the_artifact_as_delivered(make_context, make_update):
    from unittest.mock import Mock

    from sector_ai.capabilities import GeneratedOutput, deliver_output

    context = make_context()
    context.remember_sent = Mock(side_effect=RuntimeError("bookkeeping broke"))
    update = make_update(text="hi")

    result = await deliver_output(GeneratedOutput("html", {"html": "<p>x</p>"}, "", None), update, context)

    assert result["status"] == "delivered"
    assert result["note"]


def test_non_serialisable_tool_values_are_stringified(make_settings):
    from datetime import UTC, datetime

    from sector_ai.tools import result_content

    text = result_content({"when": datetime(2026, 1, 1, tzinfo=UTC)}, make_settings())

    assert "2026" in text


def test_a_receipt_with_a_huge_note_still_fits(make_settings):
    from sector_ai.tools import result_content

    text = result_content(
        {"status": "delivered", "html": "x" * 5000, "note": "n" * 900},
        make_settings(tool_result_max_chars=1024),
    )

    assert len(text) <= 1024
