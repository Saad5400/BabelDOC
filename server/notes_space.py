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

# A top/bottom band is measured in WRITING LINES, not as a fraction of the
# page: a fraction gives a taller band on a taller page, so one document whose
# pages differ in height (a mono with baked vocab strips does) gets a
# different amount of writing room on every sheet.
SIZES = {"sm": 4, "md": 8, "lg": 13}

# A left/right band is a fraction of the page WIDTH, because what a side band
# gives the reader is line LENGTH, and a line as long as the page beside it is
# the sensible ask on any page size. (Page width does not vary within a
# document the way height does.)
SIDE_FRACTIONS = {"sm": 0.18, "md": 0.30, "lg": 0.45}

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

# The only output of this module that matters is a SHEET OF PAPER, and a
# printer fits the whole page onto it: a page grown to twice A4 prints its
# ruled lines at half the spacing they were drawn at. 26 pt is 9.2 mm on a
# page that prints at full size, but `top,bottom` at the old `lg` grew an A4
# page to 595x1600, which prints at 53% — 4.8 mm between the rules, on a
# feature whose entire purpose is writing by hand.
#
# So the bands are sized against the PRINTED result. `_MIN_PRINTED_SPACING`
# is the floor an adult can write between; the bands are trimmed to whatever
# still prints at or above it, and if the page was already too big for that
# before any band was added (a side-by-side dual is), the rules are spaced
# further apart instead so that they still land on the floor.
_PAPER = (595.0, 842.0)  # A4
_MIN_PRINTED_SPACING = 18.0  # 6.3 mm on the sheet
_MIN_LINES = 2


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


def print_scale(width: float, height: float) -> float:
    """How much a printer shrinks a `width` x `height` page onto A4.

    Either orientation — nobody prints a landscape page down the short edge —
    and never an enlargement, since a page smaller than the paper is printed
    at its own size.
    """
    return max(min(paper_w / width, paper_h / height, 1.0)
               for paper_w, paper_h in (_PAPER, _PAPER[::-1]))


class _Plan:
    """The band geometry for ONE document: thickness, and the rule rhythm.

    Computed once for the whole file, from its largest page, so every sheet
    of a document gets the same amount of writing room — the fraction-of-the-
    page sizing gave a mono's tallest page 24% more room than its shortest.
    """

    def __init__(self, band_h: float, band_w: float, spacing: float,
                 lines: int, scale: float) -> None:
        self.band_h = band_h
        self.band_w = band_w
        self.spacing = spacing
        self.lines = lines
        # What a printer will do to the composed page — the number that makes
        # `spacing` mean something on paper. `spacing * scale` is what the
        # reader actually writes between.
        self.scale = scale


def _plan_for(width: float, height: float, sides: list[str], size: str,
              paper: tuple[float, float]) -> _Plan:
    """The bands `size` asks for, trimmed to what `paper` can still print.

    The budget is what the page may grow to and still print at the scale the
    rule spacing needs (`_MIN_PRINTED_SPACING / _LINE_SPACING`). A page that
    ALREADY exceeds that budget gets its bands in full: trimming them cannot
    buy back a scale the page had lost before this module touched it, and
    refusing the writing space the caller asked for would only make the
    feature useless on exactly the layouts (duals, opened-up overlays) people
    print most.
    """
    horizontal = sum(side in sides for side in ("top", "bottom"))
    vertical = sum(side in sides for side in ("left", "right"))
    floor = _MIN_PRINTED_SPACING / _LINE_SPACING

    band_w = SIDE_FRACTIONS[size] * width if vertical else 0.0
    room_w = paper[0] / floor - width

    if vertical and room_w > 0:
        band_w = min(band_w, room_w / vertical)

    lines = SIZES[size]
    room_h = paper[1] / floor - height

    if horizontal and room_h > 0:
        affordable = int((room_h / horizontal - _OUTER_INSET) // _LINE_SPACING)
        lines = max(_MIN_LINES, min(lines, affordable))

    band_h = lines * _LINE_SPACING + _OUTER_INSET if horizontal else 0.0

    scale = print_scale(width + vertical * band_w,
                        height + horizontal * band_h)
    spacing = max(_LINE_SPACING, _MIN_PRINTED_SPACING / scale)

    # Only reachable on a page that was over budget to begin with: widening
    # the rhythm past the band would draw a band with no rules in it.
    if horizontal:
        band_h = max(band_h, spacing + _OUTER_INSET)
        scale = print_scale(width + vertical * band_w,
                            height + horizontal * band_h)

    return _Plan(band_h, band_w, spacing, lines, scale)


def plan_bands(pages: list[tuple[float, float]], sides: list[str],
               size: str) -> _Plan:
    """The one band plan a document gets, from its most constrained page.

    A4 is tried both ways up and the more generous plan wins — a page grown
    only sideways prints better on landscape paper, one grown only downwards
    on portrait — preferring the plan that meets the printed-spacing floor,
    and among those the one that gives the reader more room.
    """
    width = max(w for w, _h in pages)
    height = max(h for _w, h in pages)
    floor = _MIN_PRINTED_SPACING / _LINE_SPACING

    candidates = [_plan_for(width, height, sides, size, paper)
                  for paper in (_PAPER, _PAPER[::-1])]

    return max(candidates,
               key=lambda plan: (plan.scale >= floor - 1e-9, plan.lines,
                                 plan.band_w, plan.scale))


def _rule_across(page: pymupdf.Page, x0: float, x1: float,
                 content_edge: float, outer_edge: float,
                 spacing: float) -> None:
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
    y = content_edge + direction * spacing

    while (outer_edge - y) * direction >= _OUTER_INSET:
        page.draw_line(pymupdf.Point(xa, y), pymupdf.Point(xb, y),
                       color=_LINE_COLOR, width=_LINE_WIDTH,
                       stroke_opacity=_LINE_OPACITY)
        y += direction * spacing


def _rule_beside(page: pymupdf.Page, outer_x: float, content_x: float,
                 y0: float, y1: float, spacing: float) -> None:
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

    y = y0 + spacing

    while y <= y1 - _OUTER_INSET:
        page.draw_line(pymupdf.Point(xa, y), pymupdf.Point(xb, y),
                       color=_LINE_COLOR, width=_LINE_WIDTH,
                       stroke_opacity=_LINE_OPACITY)
        y += spacing


def _extend_page(page: pymupdf.Page, sides: list[str],
                 plan: _Plan) -> bool:
    """One page grown by its bands, ruled; False when it was left alone.

    Every page of a document gets the SAME plan, whatever its own size: the
    band is writing room, and writing room that changes from sheet to sheet
    inside one printout is a defect, not a feature.
    """
    if page.rotation:
        return False

    rect = page.rect  # whatever the page is NOW — strip, overlay and all
    top = plan.band_h if "top" in sides else 0.0
    bottom = plan.band_h if "bottom" in sides else 0.0
    left = plan.band_w if "left" in sides else 0.0
    right = plan.band_w if "right" in sides else 0.0

    media = page.mediabox  # PDF space — y grows upward, so the bottom is y0
    page.set_mediabox(pymupdf.Rect(media.x0 - left, media.y0 - bottom,
                                   media.x1 + right, media.y1 + top))

    # Page space re-derives from the new mediabox: the content now sits inset
    # by (left, top) and the bands are the edges of the new rect.
    new = page.rect

    if top:
        _rule_across(page, new.x0, new.x1, content_edge=top,
                     outer_edge=new.y0, spacing=plan.spacing)
    if bottom:
        _rule_across(page, new.x0, new.x1, content_edge=new.y1 - bottom,
                     outer_edge=new.y1, spacing=plan.spacing)
    if left:
        _rule_beside(page, outer_x=new.x0, content_x=left,
                     y0=top, y1=top + rect.height, spacing=plan.spacing)
    if right:
        _rule_beside(page, outer_x=new.x1, content_x=new.x1 - right,
                     y0=top, y1=top + rect.height, spacing=plan.spacing)

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
        if doc.needs_pass:
            # An encrypted document OPENS: page_count is even right. It fails
            # on the first page access, several frames down, as a 500 — so
            # the check has to be here, where "the caller sent a locked PDF"
            # is still something we can say.
            raise NotesSpaceError(
                "file is password-protected; it has to be unlocked first")

        if doc.page_count == 0:
            raise NotesSpaceError("file has no pages")

        if doc.page_count > MAX_PAGES:
            raise NotesSpaceError(
                f"file has {doc.page_count} pages (max {MAX_PAGES})")

        sizes = [(page.rect.width, page.rect.height) for page in doc
                 if not page.rotation] or [(page.rect.width, page.rect.height)
                                           for page in doc]
        plan = plan_bands(sizes, sides, size)

        skipped = sum(1 for page in doc
                      if not _extend_page(page, sides, plan))

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
