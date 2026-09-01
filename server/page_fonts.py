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

It also owns `repair_arabic_text_layer`, the pass every renderer that draws
Arabic must run on the finished document before saving it — the Story engine
leaves a text layer that reads as glyph shapes rather than letters, and that
is not something a page can fix for itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from server.interlinear import FONT_FILE
from server.interlinear import subset_font_bytes

logger = logging.getLogger("doctranslate.page_fonts")

BOLD_FONT_FILE = "GoNotoKurrent-Bold.ttf"


def repair_arabic_text_layer(doc: pymupdf.Document) -> int:
    """Make the Arabic on `doc`'s pages extract as the letters it reads as.

    Call this on the finished document, after every page has been drawn and
    BEFORE saving it. Returns how many fonts were rewritten; a document with
    no Arabic in it is left untouched, so it is safe on any output.

    Why an appendix page needs it at all: the Story engine shapes Arabic
    through the font, so what lands on the page is contextual glyphs, and the
    ToUnicode map a viewer builds is a reverse of the font's cmap — which is
    where those shapes were looked up. The strip therefore renders correctly
    and extracts as presentation forms; worse, the glyphs the shaper reached
    through GSUB (every joined «ل», the hyphen inside an Arabic run) have no
    cmap entry to reverse at all, so the viewer falls back to reading the
    glyph id as a codepoint and «كلمات» extracts as «ﻛɅﻤﺎت». Neither is
    searchable, copyable, or readable by a screen reader, and catodemy's own
    corpus ingestion reads exactly this text.

    Nothing about the drawing changes: this only rewrites what each glyph
    claims to be.

    Never raises. The renderers that call it are inside a guard whose job is
    to keep a run alive, so an exception escaping here would take the whole
    appendix layer down to fix a text layer — the wrong trade every time. A
    failure leaves the document exactly as it was found and returns 0.
    """
    try:
        from babeldoc.format.pdf.document_il.backend.pdf_creater import (
            normalize_arabic_text_layer,
        )

        return normalize_arabic_text_layer(doc)
    except Exception:  # noqa: BLE001 - a text layer is never worth the run
        logger.exception("page fonts: could not repair the Arabic text layer; "
                         "shipping it as drawn")

        return 0


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
