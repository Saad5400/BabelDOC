"""Ruled note space (أسطر) added around every page of a PDF.

A student printing a translated deck wants somewhere to write, and the slide
itself has no room to give — so this module grows the page instead. Each
requested side gains a band of blank space with faint ruled writing lines in
it, sized as a fraction of the page dimension being extended, and the content
never moves: the trick is the same one the vocab strips use
(server/vocab_pages.attach_vocab). The mediabox is pushed OUTWARD — bottom is
y0 - h, top is y1 + h, and since PDF x grows rightward the visual-left band is
x0 - w and the visual-right is x1 + w — so the original content keeps its
coordinates untouched and stays pixel-identical.

The lines are horizontal in EVERY band, side margins included: people write
Arabic (and everything else) in horizontal lines, even in a side margin. They
are deliberately faint — thin, muted slate, drawn at low opacity — guides for
a pen, not part of the document.

CORNERS, when two adjacent sides are both requested: the horizontal bands own
them. A top or bottom band spans the FULL new width, running under any side
band, while a left or right band spans only the original content's height.
One rule, applied always, so bottom+right never draws two half-rules fighting
over the same corner.

This runs AFTER whatever else shaped the page — a baked vocab strip, an
overlay's opened-up layout — and does not care: it extends whatever page it
is handed. The one page it leaves alone is a rotated one (non-zero /Rotate):
its visual edges are not its mediabox edges, and a band drawn on the wrong
visual side helps nobody.
"""

from __future__ import annotations

import logging
from io import BytesIO

import pymupdf

logger = logging.getLogger("doctranslate.notes_space")

SIDES = ("top", "bottom", "left", "right")

# Band thickness as a fraction of the page dimension being extended: the
# height for top/bottom, the width for left/right.
SIZES = {"sm": 0.18, "md": 0.30, "lg": 0.45}

# Runs synchronously in the request thread (compose.py's posture); drawing a
# handful of lines per page is cheap, but not free times ten thousand pages.
MAX_PAGES = 2000

_LINE_SPACING = 26.0
_LINE_WIDTH = 0.6
_LINE_COLOR = (0x94 / 255, 0xA3 / 255, 0xB8 / 255)  # #94A3B8, muted slate
_LINE_OPACITY = 0.3
# Line ends and last lines keep this far from the band's OUTER edges (the new
# page edges), and the first line sits a full spacing from the content edge —
# comfortably past the 14pt minimum clearance the content is owed.
_OUTER_INSET = 20.0
_CONTENT_CLEARANCE = 14.0


class NotesSpaceError(ValueError):
    """Invalid notes-space input (maps to HTTP 422)."""


def parse_sides(raw: str) -> list[str]:
    """The `sides` form value as a validated list, in SIDES order.

    Uploader-supplied, so parsed defensively: whitespace and empty segments
    are tolerated, duplicates collapse, anything that is not a known side —
    or a value with no side at all — is refused whole.
    """
    seen = set()

    for part in (raw or "").split(","):
        side = part.strip().lower()

        if not side:
            continue

        if side not in SIDES:
            raise NotesSpaceError(
                f"sides must be a comma-separated subset of "
                f"{', '.join(SIDES)}")

        seen.add(side)

    if not seen:
        raise NotesSpaceError("at least one side is required")

    return [side for side in SIDES if side in seen]


def _rule_across(page: pymupdf.Page, x0: float, x1: float,
                 content_edge: float, outer_edge: float) -> None:
    """Faint ruled lines filling a TOP or BOTTOM band.

    The band lies between `content_edge` (where the original page ends) and
    `outer_edge` (the new page edge), in page space. Lines march from the
    content outward: the first a full spacing from the content edge, the last
    at least the outer inset short of the page edge, every end inset from the
    band's outer left/right edges.
    """
    xa, xb = x0 + _OUTER_INSET, x1 - _OUTER_INSET

    if xb <= xa:
        return

    direction = 1.0 if outer_edge > content_edge else -1.0
    y = content_edge + direction * _LINE_SPACING

    while (outer_edge - y) * direction >= _OUTER_INSET:
        page.draw_line(pymupdf.Point(xa, y), pymupdf.Point(xb, y),
                       color=_LINE_COLOR, width=_LINE_WIDTH,
                       stroke_opacity=_LINE_OPACITY)
        y += direction * _LINE_SPACING


def _rule_beside(page: pymupdf.Page, outer_x: float, content_x: float,
                 y0: float, y1: float) -> None:
    """Faint ruled lines filling a LEFT or RIGHT band — still horizontal.

    Each line runs from the outer inset to the content clearance, so the pen
    gets the band's full height at writing-line rhythm without the rules ever
    touching the content edge. The first line sits a full spacing below the
    band's top, the last at least the outer inset above its bottom.
    """
    if outer_x < content_x:  # visual-left band
        xa, xb = outer_x + _OUTER_INSET, content_x - _CONTENT_CLEARANCE
    else:                    # visual-right band
        xa, xb = content_x + _CONTENT_CLEARANCE, outer_x - _OUTER_INSET

    if xb <= xa:
        return

    y = y0 + _LINE_SPACING

    while y <= y1 - _OUTER_INSET:
        page.draw_line(pymupdf.Point(xa, y), pymupdf.Point(xb, y),
                       color=_LINE_COLOR, width=_LINE_WIDTH,
                       stroke_opacity=_LINE_OPACITY)
        y += _LINE_SPACING


def _extend_page(page: pymupdf.Page, sides: list[str],
                 fraction: float) -> bool:
    """One page grown by its bands, ruled; False when it was left alone."""
    if page.rotation:
        return False

    rect = page.rect  # whatever the page is NOW — strip, overlay and all
    band_h = fraction * rect.height
    band_w = fraction * rect.width
    top = band_h if "top" in sides else 0.0
    bottom = band_h if "bottom" in sides else 0.0
    left = band_w if "left" in sides else 0.0
    right = band_w if "right" in sides else 0.0

    media = page.mediabox  # PDF space — y grows upward, so the bottom is y0
    page.set_mediabox(pymupdf.Rect(media.x0 - left, media.y0 - bottom,
                                   media.x1 + right, media.y1 + top))

    # Page space re-derives from the new mediabox: the content now sits inset
    # by (left, top) and the bands are the edges of the new rect.
    new = page.rect

    if top:
        _rule_across(page, new.x0, new.x1,
                     content_edge=top, outer_edge=new.y0)
    if bottom:
        _rule_across(page, new.x0, new.x1,
                     content_edge=new.y1 - bottom, outer_edge=new.y1)
    if left:
        _rule_beside(page, outer_x=new.x0, content_x=left,
                     y0=top, y1=top + rect.height)
    if right:
        _rule_beside(page, outer_x=new.x1, content_x=new.x1 - right,
                     y0=top, y1=top + rect.height)

    return True


def add_notes_space(pdf_bytes: bytes, sides: list[str],
                    size: str = "md") -> bytes:
    """`pdf_bytes` with every page grown by ruled note bands on `sides`.

    Raises NotesSpaceError on anything the caller got wrong — an unknown size
    or side, no sides at all, a PDF that cannot be opened. Rotated pages are
    skipped, not failed (see the module docstring); a document of nothing but
    rotated pages simply comes back re-saved and otherwise untouched.
    """
    if size not in SIZES:
        raise NotesSpaceError(f"size must be one of {', '.join(SIZES)}")

    if not sides or any(side not in SIDES for side in sides):
        raise NotesSpaceError(
            f"sides must be a non-empty subset of {', '.join(SIDES)}")

    try:
        doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - pymupdf raises many error types
        raise NotesSpaceError(f"file is not a readable PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise NotesSpaceError("file has no pages")

        if doc.page_count > MAX_PAGES:
            raise NotesSpaceError(
                f"file has {doc.page_count} pages (max {MAX_PAGES})")

        skipped = sum(1 for page in doc
                      if not _extend_page(page, sides, SIZES[size]))

        if skipped:
            logger.info("notes-space: %s rotated page(s) of %s left alone",
                        skipped, doc.page_count)

        out = BytesIO()
        # deflate only: nothing here embeds fonts, so there is no per-box
        # subset debris for garbage collection to fold.
        doc.save(out, deflate=True)

        return out.getvalue()
    finally:
        doc.close()
