"""Tests for the SVG -> PDF -> PNG rendering pipeline used by /svg."""

from io import BytesIO

import pytest
from PIL import Image

from sector_ai.coding import render_svg_png

SVG_CASES = {
    "basic_shape": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect x="10" y="10" width="50" height="50" fill="red"/></svg>'
    ),
    "text": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">'
        '<text x="100" y="50" text-anchor="middle" font-size="20">Hi</text></svg>'
    ),
    "transformed_group": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<g transform="translate(10,10)"><rect width="50" height="50" fill="green"/></g></svg>'
    ),
    # svglib 2.x paints gradients through the PDF canvas, which reportlab 5's
    # removed bitmap backend could not handle.
    "gradient": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<defs><linearGradient id="g"><stop offset="0%" stop-color="red"/>'
        '<stop offset="100%" stop-color="blue"/></linearGradient></defs>'
        '<circle cx="50" cy="50" r="40" fill="url(#g)"/></svg>'
    ),
}


@pytest.mark.parametrize("name", SVG_CASES)
def test_render_svg_png(name):
    png = render_svg_png(SVG_CASES[name], size=250)

    assert png.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(png.getvalue())) as image:
        assert image.size == (500, 500)


def test_render_svg_png_rejects_garbage():
    with pytest.raises(Exception):
        render_svg_png("not an svg", size=250)
