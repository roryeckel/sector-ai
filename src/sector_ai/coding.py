import asyncio
import logging
from io import BytesIO

import pypdfium2 as pdfium
from langchain_core.prompts import PromptTemplate
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from telegram import Update
from telegram.error import BadRequest

from .sector_context import SectorContext

logger = logging.getLogger(__name__)


def render_svg_png(svg: str, size: int) -> BytesIO:
    """Rasterise SVG markup into a PNG buffer, scaled to roughly ``size`` pixels."""
    drawing = svg2rlg(BytesIO(svg.encode("utf-8")))
    if not drawing:
        raise ValueError("Unable to parse SVG")

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


async def code_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.code, input_variables=["input", *system_template_dict.keys()]
    )
    chain = template | context.bot_ollama
    code_prompt = update.message.text.split(maxsplit=1)
    if len(code_prompt) <= 1:
        await update.message.reply_text("No prompt provided.")
        return
    code_prompt = code_prompt[-1]
    logger.info(f"Code Prompt: {code_prompt}")
    for i in range(3):
        code = (await chain.ainvoke({"input": code_prompt, **system_template_dict})).content
        logger.info(f"Code: {code}")
        try:
            await update.message.reply_markdown_v2(code)
            return
        except BadRequest as e:
            logger.exception("Code Error", exc_info=e)
            logger.exception(f"The code was: {code}")
            continue
    await update.message.reply_text("Failed to generate valid code snippet.")


async def html_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.html, input_variables=["input", *system_template_dict.keys()]
    )
    chain = template | context.bot_ollama
    html_prompt = update.message.text.split(maxsplit=1)
    if len(html_prompt) <= 1:
        await update.message.reply_text("No prompt provided.")
        return
    html_prompt = html_prompt[-1]
    logger.info(f"HTML Prompt: {html_prompt}")
    for i in range(3):
        html = (await chain.ainvoke({"input": html_prompt, **system_template_dict})).content
        logger.info(f"HTML: {html}")
        try:
            await update.message.reply_document(
                filename="Website.html", caption=html_prompt, document=BytesIO(html.encode("utf-8"))
            )
            return
        except BadRequest as e:
            logger.exception("HTML Error", exc_info=e)
            logger.exception(f"The HTML was: {html}")
            continue
    await update.message.reply_text("Failed to generate valid HTML snippet.")


async def svg_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.settings.system_prompts.svg, input_variables=["input", *system_template_dict.keys()]
    )
    chain = template | context.bot_ollama
    svg_prompt = update.message.text.split(maxsplit=1)
    if len(svg_prompt) <= 1:
        await update.message.reply_text("No prompt provided.")
        return
    svg_prompt = svg_prompt[-1]
    logger.info(f"SVG Prompt: {svg_prompt}")
    for i in range(3):
        svg = (await chain.ainvoke({"input": svg_prompt, **system_template_dict})).content.strip("```")
        logger.info(f"SVG: {svg}")
        try:
            png_file = await asyncio.to_thread(render_svg_png, svg, context.chat_svg_size)
            await update.message.reply_photo(photo=png_file)
            return
        except Exception as e:
            logger.exception("SVG Error", exc_info=e)
            logger.exception(f"The SVG was: {svg}")
            continue
    await update.message.reply_text("Failed to generate valid SVG snippet.")
