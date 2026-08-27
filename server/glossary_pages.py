"""The «شرح المصطلحات» pages: glossary entries rendered onto appended pages.

`server/terms.py` picks a document's genuinely hard English terms and writes a
friendly Arabic explanation for each; this module is the one renderer of those
entries, shared by every place they can end up in a PDF — the mono result
(server/pipeline.py), the interlinear overlay (server/interlinear.py) and the
recomposed duals (server/compose.py). One renderer, so the pages look the same
whichever door the reader came through.

DESIGN (owner-approved sample): pages the same size as the document's own,
white, RTL. A bold blue title («شرح المصطلحات» + Terms), then one card per
term — light slate background, rounded corners, the term header as
"Wrapping — التغليف" in the title blue, a ~19 px explanation body, and a
smaller muted source line («وردت في: الشريحة 12 — "…"»). Cards that do not fit
the page flow onto further appended pages.

TEXT is PyMuPDF's Story engine via `insert_htmlbox` — the same machinery and
the same GoNotoKurrent face as the interlinear overlay, so Arabic arrives
shaped and bidi-resolved (and the Latin terms inside RTL lines land where they
should). The Story engine has no rounded corners, so each card's background is
drawn with `Page.draw_rect(radius=...)` first and the text laid over it; the
card's height comes from a measuring `Story.place` pass, which is also what
decides the page breaks. Both faces are subset before use (the full face is
15 MB per copy) with the subsetter shared from server/interlinear.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from server.interlinear import FONT_FILE
from server.interlinear import subset_font_bytes

logger = logging.getLogger("doctranslate.glossary_pages")

BOLD_FONT_FILE = "GoNotoKurrent-Bold.ttf"

# The approved palette.
_TITLE_COLOR = "#1d4ed8"
_BODY_COLOR = "#111827"
_SOURCE_COLOR = "#64748b"
_CARD_BG = (0xF1 / 255, 0xF5 / 255, 0xF9 / 255)  # #F1F5F9 as pymupdf floats

# Page geometry (points). The page itself is whatever size the document's
# pages are; these are the margins and card metrics inside it.
_MARGIN_X = 48.0
_MARGIN_TOP = 52.0
_MARGIN_BOTTOM = 48.0
_TITLE_GAP = 20.0          # below the title block
_CARD_PAD = 14.0           # card background around its text
_CARD_GAP = 14.0           # between cards
_CARD_RADIUS = 6.0

_FALLBACK_PAGE = (595.0, 842.0)  # A4, for the pathological empty-doc caller


def _font_path(font_file: str) -> Path:
    """A face from babeldoc's font cache; the bold falls back to the regular."""
    from babeldoc.assets import assets

    try:
        path, _ = assets.get_font_and_metadata(font_file)
        return Path(path)
    except Exception:  # noqa: BLE001 - a missing bold is not worth a failure
        if font_file == FONT_FILE:
            raise
        logger.warning("glossary: %s unavailable; using the regular face",
                       font_file)
        path, _ = assets.get_font_and_metadata(FONT_FILE)
        return Path(path)


class _CardFonts:
    """The two subset faces plus the stylesheet the cards are set in."""

    def __init__(self, texts: list[str]) -> None:
        self.archive = pymupdf.Archive()
        self.archive.add(subset_font_bytes(_font_path(FONT_FILE), texts),
                         "gloss.ttf")
        self.archive.add(subset_font_bytes(_font_path(BOLD_FONT_FILE), texts),
                         "gloss-bold.ttf")
        self.css = (
            "@font-face {font-family: gloss; src: url(gloss.ttf);}"
            "@font-face {font-family: glossbold; src: url(gloss-bold.ttf);}"
            "body {margin: 0; font-family: gloss;}"
            f"div.title {{font-family: glossbold; font-size: 24px;"
            f" color: {_TITLE_COLOR};}}"
            f"div.term {{font-family: glossbold; font-size: 20px;"
            f" color: {_TITLE_COLOR}; margin-bottom: 5px;}}"
            f"div.body {{font-size: 19px; color: {_BODY_COLOR};"
            " line-height: 1.9;}"
            f"div.source {{font-size: 13px; color: {_SOURCE_COLOR};"
            " margin-top: 7px;}"
        )


def _escape(text: str) -> str:
    import html

    return html.escape(str(text))


def _title_html() -> str:
    return '<div class="title" dir="rtl">شرح المصطلحات — Terms</div>'


def _card_html(entry: dict) -> str:
    term = _escape(entry.get("term") or "")
    arabic = _escape(entry.get("arabic") or "")
    header = f"{term} — {arabic}" if arabic else term

    parts = [f'<div class="term">{header}</div>',
             f'<div class="body">{_escape(entry.get("explanation") or "")}</div>']

    page, quote = entry.get("page"), entry.get("quote")
    if page or quote:
        where = f"الشريحة {page}" if page else "المستند"
        source = f"وردت في: {where}"
        if quote:
            source += f' — "{_escape(quote)}"'
        parts.append(f'<div class="source">{source}</div>')

    return f'<div dir="rtl">{"".join(parts)}</div>'


def _usable(entry: object) -> bool:
    return (isinstance(entry, dict)
            and str(entry.get("term") or "").strip() != ""
            and str(entry.get("explanation") or "").strip() != "")


def _measure(html: str, fonts: _CardFonts, width: float) -> float:
    """The height `html` needs at `width` (one Story.place pass)."""
    story = pymupdf.Story(html=html, user_css=fonts.css, archive=fonts.archive)
    _, filled = story.place(pymupdf.Rect(0, 0, width, 100_000))

    return float(filled[3])


def append_glossary_pages(doc: pymupdf.Document, entries: list,
                          page_size: tuple[float, float] | None = None) -> int:
    """Append the styled glossary pages to `doc`; returns pages added.

    `entries` is the sidecar's "glossary" list; unusable items (no term or no
    explanation) are skipped rather than rendered half-empty. No usable entry
    means no page and 0. Pages match `page_size` when given, else the size of
    `doc`'s last page — the page the glossary follows.

    The caller owns the save. Save with `garbage=4` (as every caller here
    does): like the interlinear overlay, each drawn box embeds its own copy of
    the subset font until garbage collection folds them into one.
    """
    cards = [entry for entry in entries or [] if _usable(entry)]

    if not cards:
        return 0

    if page_size is not None:
        width, height = float(page_size[0]), float(page_size[1])
    elif doc.page_count:
        rect = doc[doc.page_count - 1].rect
        width, height = rect.width, rect.height
    else:
        width, height = _FALLBACK_PAGE

    texts = ["شرح المصطلحات — Terms وردت في: الشريحة المستند 0123456789"]
    texts += [str(value) for card in cards
              for value in card.values() if value is not None]
    fonts = _CardFonts(texts)

    content_width = width - 2 * _MARGIN_X
    text_width = content_width - 2 * _CARD_PAD
    bottom = height - _MARGIN_BOTTOM

    page = doc.new_page(width=width, height=height)
    added = 1

    title = _title_html()
    title_height = _measure(title, fonts, content_width)
    page.insert_htmlbox(
        pymupdf.Rect(_MARGIN_X, _MARGIN_TOP, width - _MARGIN_X,
                     _MARGIN_TOP + title_height + 2),
        title, css=fonts.css, archive=fonts.archive)
    y = _MARGIN_TOP + title_height + _TITLE_GAP

    for card in cards:
        html = _card_html(card)
        text_height = _measure(html, fonts, text_width)
        card_height = text_height + 2 * _CARD_PAD

        if y + card_height > bottom and y > _MARGIN_TOP:
            # Overflow: this card opens a fresh page (no repeated title).
            page = doc.new_page(width=width, height=height)
            added += 1
            y = _MARGIN_TOP

        card_rect = pymupdf.Rect(_MARGIN_X, y, width - _MARGIN_X,
                                 min(y + card_height, bottom))
        radius = min(0.5, _CARD_RADIUS / max(min(card_rect.width,
                                                 card_rect.height), 1.0))
        page.draw_rect(card_rect, color=None, fill=_CARD_BG, radius=radius)
        # scale_low=0 lets a card taller than a whole page shrink to fit
        # rather than truncate; a normal card is measured to fit exactly.
        page.insert_htmlbox(
            pymupdf.Rect(card_rect.x0 + _CARD_PAD, card_rect.y0 + _CARD_PAD,
                         card_rect.x1 - _CARD_PAD, card_rect.y1 - _CARD_PAD),
            html, css=fonts.css, archive=fonts.archive, scale_low=0)
        y = card_rect.y1 + _CARD_GAP

    logger.info("glossary: %s card(s) over %s appended page(s)",
                len(cards), added)

    return added
