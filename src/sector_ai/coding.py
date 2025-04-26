import logging
from .sector_context import SectorContext
from telegram import Update
from telegram.error import BadRequest
from langchain_core.prompts import PromptTemplate
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from io import BytesIO
from telegram.error import BadRequest

logger = logging.getLogger(__name__)

async def code_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.config_code_system_prompt,
        input_variables=['input', *system_template_dict.keys()])
    chain = template | context.bot_ollama
    code_prompt = update.message.text.split(maxsplit=1)
    if len(code_prompt) <= 1:
        await update.message.reply_text('No prompt provided.')
        return
    code_prompt = code_prompt[-1]
    logger.info(f'Code Prompt: {code_prompt}')
    for i in range(3):
        code = chain.invoke({'input': code_prompt, **system_template_dict}).content
        logger.info(f'Code: {code}')
        try:
            await update.message.reply_markdown_v2(code)
            return
        except BadRequest as e:
            logger.exception(f'Code Error', exc_info=e)
            logger.exception(f'The code was: {code}')
            continue
    await update.message.reply_text('Failed to generate valid code snippet.')

async def html_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.config_html_system_prompt,
        input_variables=['input', *system_template_dict.keys()])
    chain = template | context.bot_ollama
    html_prompt = update.message.text.split(maxsplit=1)
    if len(html_prompt) <= 1:
        await update.message.reply_text('No prompt provided.')
        return
    html_prompt = html_prompt[-1]
    logger.info(f'HTML Prompt: {html_prompt}')
    for i in range(3):
        html = chain.invoke({'input': html_prompt, **system_template_dict}).content
        logger.info(f'HTML: {html}')
        try:
            await update.message.reply_document(
                filename='Website.html',
                caption=html_prompt,
                document=BytesIO(html.encode('utf-8')))
            return
        except BadRequest as e:
            logger.exception(f'HTML Error', exc_info=e)
            logger.exception(f'The HTML was: {html}')
            continue
    await update.message.reply_text('Failed to generate valid HTML snippet.')

async def svg_cmd(update: Update, context: SectorContext) -> None:
    system_template_dict = await context.get_system_template_dict()
    template = PromptTemplate(
        template=context.config_svg_system_prompt,
        input_variables=['input', *system_template_dict.keys()])
    chain = template | context.bot_ollama
    svg_prompt = update.message.text.split(maxsplit=1)
    if len(svg_prompt) <= 1:
        await update.message.reply_text('No prompt provided.')
        return
    svg_prompt = svg_prompt[-1]
    logger.info(f'SVG Prompt: {svg_prompt}')
    for i in range(3):
        svg = chain.invoke({'input': svg_prompt, **system_template_dict}).content.strip('```')
        logger.info(f'SVG: {svg}')
        try:
            drawing = svg2rlg(BytesIO(svg.encode('utf-8')))
            if not drawing:
                continue
            svg_size = context.chat_svg_size
            scale_factor = svg_size / max(drawing.width, drawing.height)
            drawing.scale(scale_factor, scale_factor)
            drawing.width = svg_size
            drawing.height = svg_size
            png_file = BytesIO()
            renderPM.drawToFile(drawing, png_file, fmt='PNG')
            png_file.seek(0)
            await update.message.reply_photo(photo=png_file)
            return
        except Exception as e:
            logger.exception(f'SVG Error', exc_info=e)
            logger.exception(f'The SVG was: {svg}')
            continue
    await update.message.reply_text('Failed to generate valid SVG snippet.')