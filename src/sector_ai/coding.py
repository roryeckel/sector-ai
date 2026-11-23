import logging
import math
from io import BytesIO

import pypdfium2 as pdfium
from langchain_core.prompts import PromptTemplate
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from telegram import Update
from telegram.error import BadRequest

from .chat_metadata import instruction_messages, strip_internal_envelope, with_context
from .history import record_content_command
from .message_text import message_refused, message_text
from .sector_context import SectorContext
from .usage import record_usage

logger = logging.getLogger(__name__)


def render_svg_png(svg: str, size: int) -> BytesIO:
    """Rasterise SVG markup into a PNG buffer, scaled to roughly ``size`` pixels."""
    drawing = svg2rlg(BytesIO(svg.encode("utf-8")))
    if not drawing:
        raise ValueError("Unable to parse SVG")
    if any(not math.isfinite(value) or value <= 0 or value > 100000 for value in (drawing.width, drawing.height)):
        raise ValueError("Invalid SVG dimensions")

    scale_factor = size / max(drawing.width, drawing.height)
    drawing.scale(scale_factor, scale_factor)
    drawing.width = size
    drawing.height = size

    # reportlab 5 dropped its built-in bitmap backend, and svglib's gradient support
    # relies on the PDF canvas, so render to PDF and rasterise that instead.
    pdf_buffer = BytesIO()
    renderPDF.drawToFile(drawing, pdf_buffer)
    pdf_buffer.seek(0)

    png_buffer = BytesIO()
    pdf = pdfium.PdfDocument(pdf_buffer)
    try:
        pdf[0].render(scale=2).to_pil().save(png_buffer, format="PNG")
    finally:
        pdf.close()

    png_buffer.seek(0)
    return png_buffer


@record_content_command
async def code_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.code, input_variables=["input", *system_template_dict.keys()]
    )
    code_prompt = update.message.text.split(maxsplit=1)
    if len(code_prompt) <= 1:
        await update.message.reply_text("No prompt provided.")
        return
    code_prompt = code_prompt[-1]
    logger.info(f"Code Prompt: {code_prompt}")
    messages = with_context(
        instruction_messages(
            template,
            {"input": code_prompt, **system_template_dict},
            input_key="input",
            input_text=code_prompt,
        ),
        context,
    )
    model = context.bind_cache(context.bot_llm, "code")
    for i in range(3):
        message = await model.ainvoke(messages)
        record_usage(context, "code", message)
        if message_refused(message):
            await update.message.reply_text(message_text(message) or "The model refused the request.")
            return
        code = strip_internal_envelope(message_text(message))
        logger.info(f"Code: {code}")
        try:
            sent = await update.message.reply_markdown_v2(code)
            context.save_assistant_message(code, sent)
            return
        except BadRequest as e:
            logger.exception("Code Error", exc_info=e)
            logger.exception(f"The code was: {code}")
            continue
    await update.message.reply_text("Failed to generate valid code snippet.")


async def html_cmd(update: Update, context: SectorContext) -> None:
    await artifact_command(update, context, "html")


async def svg_cmd(update: Update, context: SectorContext) -> None:
    await artifact_command(update, context, "svg")


async def artifact_command(update, context, command: str) -> None:
    parts = (update.effective_message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await update.effective_message.reply_text("No prompt provided.")
        return
    from .agent import run_agent

    await run_agent(update, context, command=command, prompt=parts[1])
