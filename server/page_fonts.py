"""Shared fonts-and-measure machinery for the appendix-page renderers.

An appendix renderer (`server/vocab_pages.py` today) sets whole inserted pages
with PyMuPDF's Story engine, in the same GoNotoKurrent face the interlinear
overlay glosses with — so Arabic arrives shaped and bidi-resolved — plus its
bold. Both faces are 15 MB whole and the Story engine embeds a fresh copy per
drawn box, so this module subsets them to the texts actually drawn (the
subsetter is shared from server/interlinear.py, which solved the same problem
for the gloss font) and offers the measuring `Story.place` pass that renderers
use to decide their page breaks. Every caller saves with `garbage=4`, which is
what folds the per-box copies of the subset faces into one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from server.interlinear import FONT_FILE
from server.interlinear import subset_font_bytes

logger = logging.getLogger("doctranslate.page_fonts")

BOLD_FONT_FILE = "GoNotoKurrent-Bold.ttf"


def _font_path(font_file: str) -> Path:
    """A face from babeldoc's font cache; the bold falls back to the regular."""
    from babeldoc.assets import assets

    try:
        path, _ = assets.get_font_and_metadata(font_file)
        return Path(path)
    except Exception:  # noqa: BLE001 - a missing bold is not worth a failure
        if font_file == FONT_FILE:
            raise
        logger.warning("page fonts: %s unavailable; using the regular face",
                       font_file)
        path, _ = assets.get_font_and_metadata(FONT_FILE)
        return Path(path)


class PageFonts:
    """The two subset faces plus a stylesheet — regular as `gloss`, bold as
    `glossbold`, `extra_css` appended after the shared base rules.

    One instance covers one document's appendix pages: the subset is taken
    once from every text the renderer will draw, then reused per page.
    """

    def __init__(self, texts: list[str], extra_css: str = "") -> None:
        self.archive = pymupdf.Archive()
        self.archive.add(subset_font_bytes(_font_path(FONT_FILE), texts),
                         "gloss.ttf")
        self.archive.add(subset_font_bytes(_font_path(BOLD_FONT_FILE), texts),
                         "gloss-bold.ttf")
        self.css = (
            "@font-face {font-family: gloss; src: url(gloss.ttf);}"
            "@font-face {font-family: glossbold; src: url(gloss-bold.ttf);}"
            "body {margin: 0; font-family: gloss;}"
            + extra_css
        )


def _escape(text: str) -> str:
    import html

    return html.escape(str(text))


def measure(html: str, fonts: PageFonts, width: float) -> float:
    """The height `html` needs at `width` (one Story.place pass).

    Measuring is how an appendix renderer decides its page breaks: each
    fragment is placed into an unbounded rect first, and only the filled
    height is believed.
    """
    story = pymupdf.Story(html=html, user_css=fonts.css, archive=fonts.archive)
    _, filled = story.place(pymupdf.Rect(0, 0, width, 100_000))

    return float(filled[3])
