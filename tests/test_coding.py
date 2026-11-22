"""Contract tests for /code, /html and /svg: what actually goes out."""

from io import BytesIO

import pytest
from PIL import Image

from sector_ai.coding import code_cmd, html_cmd, svg_cmd

VALID_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="50" height="50" fill="red"/></svg>'
)


async def test_code_cmd_sends_markdown(make_context, make_update):
    context = make_context("print(1)")
    update = make_update(text="/code print one")

    await code_cmd(update, context)

    assert update.message.calls_to("reply_markdown_v2")[0][0] == ("print(1)",)


async def test_code_cmd_reports_a_refusal(make_context, make_update):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = GenericFakeChatModel(
        messages=iter([AIMessage(content=[{"type": "refusal", "refusal": "I cannot write that."}])])
    )
    context = make_context(model=model)
    update = make_update(text="/code print one")

    await code_cmd(update, context)

    assert update.message.texts("reply_text")[0] == "I cannot write that."
    assert update.message.calls_to("reply_markdown_v2") == []


async def test_code_cmd_prompt_includes_user_input_and_template_vars(make_context, make_update):
    context = make_context("print(1)")
    update = make_update(text="/code print one")

    await code_cmd(update, context)

    sent = context.bot_llm.received[0][0].content
    assert "print one" in sent
    assert "4096" in sent  # MAX_LEN was substituted


async def test_code_cmd_retries_before_succeeding(make_context, make_update):
    context = make_context("first", "second", "third")
    update = make_update(text="/code print one", fail_times=2)

    await code_cmd(update, context)

    assert len(update.message.calls_to("reply_markdown_v2")) == 3
    assert context.bot_llm.call_count == 3
    assert "Failed" not in update.message.texts("reply_text")


async def test_code_cmd_gives_up_after_three_attempts(make_context, make_update):
    context = make_context("a", "b", "c")
    update = make_update(text="/code print one", fail_times=3)

    await code_cmd(update, context)

    assert context.bot_llm.call_count == 3
    assert update.message.texts("reply_text")[-1] == "Failed to generate valid code snippet."


async def test_html_cmd_sends_html_document(make_context, make_update):
    context = make_context("<html><body>hi</body></html>")
    update = make_update(text="/html a page")

    await html_cmd(update, context)

    _, kwargs = update.message.calls_to("reply_document")[0]
    assert kwargs["filename"] == "Website.html"
    assert kwargs["caption"] == "a page"
    assert isinstance(kwargs["document"], BytesIO)
    assert kwargs["document"].getvalue() == b"<html><body>hi</body></html>"


async def test_html_cmd_retries_before_succeeding(make_context, make_update):
    context = make_context("first page", "second page")
    update = make_update(text="/html a page", fail_times=1)

    await html_cmd(update, context)

    assert len(update.message.calls_to("reply_document")) == 2
    assert "Failed" not in update.message.texts("reply_text")


async def test_html_cmd_gives_up_after_three_attempts(make_context, make_update):
    context = make_context("a", "b", "c")
    update = make_update(text="/html a page", fail_times=3)

    await html_cmd(update, context)

    assert len(update.message.calls_to("reply_document")) == 3
    assert update.message.texts("reply_text")[-1] == "Failed to generate valid HTML snippet."


async def test_svg_cmd_sends_png_photo(make_context, make_update):
    context = make_context(VALID_SVG)
    update = make_update(text="/svg a red square")

    await svg_cmd(update, context)

    _, kwargs = update.message.calls_to("reply_photo")[0]
    photo = kwargs["photo"]
    assert isinstance(photo, BytesIO)
    assert photo.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


async def test_svg_cmd_scales_to_configured_size(make_context, make_update):
    context = make_context(VALID_SVG)
    update = make_update(text="/svg a red square")

    await svg_cmd(update, context)

    _, kwargs = update.message.calls_to("reply_photo")[0]
    with Image.open(BytesIO(kwargs["photo"].getvalue())) as image:
        assert image.size == (context.chat_svg_size * 2, context.chat_svg_size * 2)


async def test_svg_cmd_retries_invalid_output(make_context, make_update):
    context = make_context("not an svg", "still not svg", VALID_SVG)
    update = make_update(text="/svg a red square")

    await svg_cmd(update, context)

    assert context.bot_llm.call_count == 3
    assert len(update.message.calls_to("reply_photo")) == 1


async def test_svg_cmd_gives_up_after_three_attempts(make_context, make_update):
    context = make_context("nope", "nope", "nope")
    update = make_update(text="/svg a red square")

    await svg_cmd(update, context)

    assert update.message.calls_to("reply_photo") == []
    assert update.message.texts("reply_text")[-1] == "Failed to generate valid SVG snippet."


@pytest.mark.parametrize("command", ["/code", "/html", "/svg"])
async def test_commands_without_prompt(command, make_context, make_update):
    context = make_context("unused")
    update = make_update(text=command)

    module = {"/code": code_cmd, "/html": html_cmd, "/svg": svg_cmd}[command]
    await module(update, context)

    assert update.message.texts("reply_text") == ["No prompt provided."]
