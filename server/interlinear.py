"""Stateless overlay layouts: one paid translation, many ways to read it.

`compose.py` rebuilds the alternating / side-by-side duals by shuffling PDF
pages, which is all those layouts are. This module rebuilds the ones that need
to know what the translation SAYS and WHERE it belongs — and it gets that from
the run's sidecar (`document_il/midend/translation_sidecar.py`), never from a
second translation.

An INTERLINEAR layout keeps the original page and sets each paragraph's
translation small directly above it, the way a gloss sits above a line in an
interlinear text. The reader keeps the slide they already know and gets the
Arabic in the same glance — which neither the Arabic-only PDF (the original is
gone) nor the duals (the two are pages apart) can do.

There are two of them, and the difference between them is one question: may
the page change size?

`interlinear` says yes, and is the answer for almost every reader. The page is
CUT along horizontal lines and everything below each cut slides down, opening
a band exactly as tall as that line's gloss needs. Nothing is scaled, nothing
is reflowed, nothing is skipped, and the type is the requested fraction of the
source's every time. Its own long comment, at THE SPACED LAYOUT below, is where
the mechanics live: where a page may be cut, what moves when it is, and where a
gloss attaches.

`interlinear_compact` says no, and pays for it. A gloss gets the vertical band
between its paragraph and whatever sits above it, and is sized to that band:
the requested fraction of the source font size first, then stepped down, and
finally force-fitted by the renderer if a step still spills. A paragraph with
no usable band above it is SKIPPED rather than drawn over the reader's
document, and the count comes back to the caller — a bad fit should be visible,
not silent. On a real slide deck that costs a fifth of the glosses and shrinks
the rest, which is why it is no longer the default; it is still the right
answer when the original pagination has to be preserved, and it is the only one
that can gloss a page whose /Rotate the other cannot open along.

When one band cannot hold a gloss at a readable size, the compact layout
SPREADS it down the paragraph's own source lines instead of shrinking it into
illegibility — the sidecar carries where those lines sit. This is what makes a
slide's bullet list work there: BabelDOC merges a run of same-styled bullets
into one paragraph, so the whole list arrives as a single block whose only free
space is the sliver above the first bullet, and each bullet has an empty line
above it going unused. The split is PROPORTIONAL, not aligned — the translation
is one sentence and no word of it is claimed to belong to a particular source
line — but it reads in order, top to bottom, which one crushed 4 pt paragraph
does not. The spaced layout splits the same way, for the same reason, and there
it is the normal case rather than the rescue.

Text that lives INSIDE an embedded raster image — a diagram's labels, a
figure's captions — arrives as blocks marked `on_raster` with the image's
`region`, and gets the same treatment TURNED INSIDE OUT. Everywhere else the
page's ink is the veto and the whitespace is the canvas; inside an image there
is no whitespace, only artwork, so the gloss is allowed onto the artwork and
carries its own legibility with it: a rounded translucent PLATE under the
text. What it may still never cover is the image's own INK — the label it
glosses, the neighbouring labels, the arrows and borders between them — and
since none of that exists as objects the page could report, the region's
PIXELS are consulted instead: wherever the rendering changes sharply, something
was drawn, and the plate keeps off it. The band directly above the label is
tried first, then the band below, and a label with no quiet band either way is
skipped and counted, exactly like a paragraph with no room.

TEXT RENDERING is PyMuPDF's Story engine, which shapes Arabic into its
contextual forms and runs the bidi algorithm over it — including the Latin
technical terms the glossary deliberately keeps in Latin script. The sidecar
carries logical text, so this holds for any target language, RTL or not.
"""
from __future__ import annotations

import html
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy
import pymupdf

from server import config

logger = logging.getLogger("doctranslate.interlinear")

# Serialises MuPDF's global glyph-height switch — see {@link _text_ink}.
_GLYPH_HEIGHTS = threading.Lock()

# `interlinear` is the layout the product means by the word: the one that opens
# the page up so every gloss fits. `interlinear_compact` is the same idea under
# the constraint that the page may not change size — worth keeping for a reader
# who needs the original pagination, and the only thing that can gloss a page
# whose /Rotate this layout cannot open along.
COMPACT_STYLE = "interlinear_compact"
OVERLAY_STYLES = ("interlinear", COMPACT_STYLE)

# The sidecar shapes this module can read. A newer engine's sidecar is refused,
# not guessed at.
SUPPORTED_SIDECAR_VERSIONS = (1,)

# Overlays run synchronously in the request thread (compose.py's posture).
MAX_PAGES = 2000

# Anything that is not TEXT and covers more than this share of the page is
# scenery — a slide's background panel, a full-bleed photo, a decorative frame
# — and treating it as something a gloss must avoid would empty the page of
# glosses. Anything smaller is real furniture: a heading's rule, a table
# border, a callout box, an inline figure. The share is deliberately generous:
# a backdrop is nearly always full-bleed, so the guard only has to tell
# "behind everything" from "beside the text".
_MAX_BACKDROP_SHARE = 0.4

# How much legibility a spread gloss may give up against the single one and
# still be preferred. Type size is the tiebreaker because it is the thing the
# reader actually loses: a paragraph whose lines are tightly leaded can only
# take a spread by shrinking it far below the single gloss, and that ratio
# stays well under this. A merged bullet list, where every bullet already has
# an empty line above it, comes out even — and there the spread is the whole
# point.
_SPREAD_TOLERANCE = 0.9

# The raster lane's view of an image region: its pixels at twice the region's
# point size (144 dpi), which resolves a 1 pt hairline to a couple of pixels —
# enough to see every mark that matters without paying photo-sized renders.
_RASTER_SCALE = 2.0

# A grey step this big between neighbouring pixels is a mark, not a gradient:
# text and line art against any flat fill clear it by a wide margin, a
# background's soft blend stays under it.
_RASTER_EDGE_STEP = 20

# How far ink pushes back from itself, in mask pixels — the anti-aliased fringe
# around every glyph and line, plus a hair of standoff so a plate never looks
# welded to the mark it stopped at.
_RASTER_INK_DILATION = 2

# The share of a plate's pixels allowed to be ink anyway: exactly none is
# brittle (one stray speck of dither vetoes a whole band), so a speck is
# forgiven and anything that could be part of a letter is not.
_RASTER_INK_TOLERANCE = 0.003

# Corner rounding of the legibility plate, as the fraction of its short side
# pymupdf's draw_rect wants. Enough to read as a tag laid on the artwork
# rather than a patch torn out of it.
_PLATE_RADIUS = 0.25

# The pan-Unicode face BabelDOC itself embeds for Arabic output, so a gloss is
# set in the same letterforms as the Arabic-only PDF. It is also 15 MB, which
# is why {@link _GlossFont} subsets it before anything is drawn.
FONT_FILE = "GoNotoKurrent-Regular.ttf"

# Always kept in the subset: ASCII (Latin technical terms, digits, punctuation)
# and the few marks Arabic text picks up around them.
_ALWAYS_SUBSET = set(range(0x20, 0x7F)) | {ord(char) for char in "،؛؟–—‘’“”…"}

# Arabic Presentation Forms-A and -B: the contextual shapes the shaper
# substitutes in. GSUB closure keeps the GLYPHS whether or not these are asked
# for — but not their cmap entries, and without those the viewer has nothing to
# build a ToUnicode map from, so the finished PDF's Arabic extracts (and copies,
# and searches) as mojibake. 57 KB for a document whose text is still text.
_ARABIC_PRESENTATION_FORMS = (set(range(0xFB50, 0xFE00))
                              | set(range(0xFE70, 0xFF00)))
_ARABIC_BLOCK = range(0x0600, 0x0700)

_FONT_PATH: Path | None = None

# `color` is the one FREE-TEXT option and it lands inside a stylesheet, so it
# is pinned to the hex form rather than trusted to be a colour at all: anything
# else is pasted into the CSS as written, where a stray `}` stops styling the
# gloss and starts styling the document.
_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class OverlayError(ValueError):
    """Invalid overlay input (maps to HTTP 422)."""


@dataclass(frozen=True)
class OverlayOptions:
    """The knobs a caller may turn. The defaults are the tuned ones."""

    # THE rule: a gloss is a fraction of the SOURCE paragraph's font size, so
    # it stays in proportion to whatever it sits above — a heading's gloss is
    # bigger than a footnote's, exactly as the heading is. 0.68 puts a 15 pt
    # slide bullet's gloss at ~10 pt: unmistakably secondary, comfortably
    # readable.
    scale: float = 0.68
    # Backstops, not the rule. The floor is where a gloss stops being worth
    # drawing at all. The ceiling only catches the absurd (a 40 pt cover
    # title); set it near the body sizes it used to sit at and it silently
    # flattens every heading's gloss to the same size, which is precisely the
    # "fixed small text" the scale exists to avoid.
    min_font_size: float = 5.0
    max_font_size: float = 24.0
    line_height: float = 1.15
    # Clearance kept above the original and below whatever is above it, so a
    # gloss never looks welded to either. Deliberately small: it is charged
    # TWICE against a band that is often only a few points tall to begin with
    # (a tightly leaded bullet list runs ~7 pt between bullets), so a generous
    # margin here is paid for directly in gloss size — 1.5 pt cost a real
    # 19 pt deck a third of its type and 16 of its glosses outright.
    gap: float = 0.75
    # Distinct from body text on purpose: a gloss should read as an annotation
    # at a glance, not as a line the author wrote.
    color: str = "#1a3a5c"
    # VISUAL alignment of the gloss inside its band — 'left' lines it up with
    # the left edge of the paragraph it belongs to (which is what makes it read
    # as attached to that paragraph, whatever the gloss's own direction).
    align: str = "left"
    # How far a gloss may shrink below min_font_size when its band is still too
    # tight — the last resort before it is skipped altogether.
    squeeze: float = 0.8
    # The legibility plate under a raster gloss — the one gloss that is drawn
    # OVER artwork instead of into whitespace. Translucent on purpose: the
    # artwork should read as dimmed under the plate, not censored by it, and
    # 0.72 is where dark text stays comfortable on top of any flat fill the
    # test decks throw at it. White suits a dark default text colour; a caller
    # recolouring the gloss light can recolour the plate to match.
    plate_color: str = "#ffffff"
    plate_opacity: float = 0.72
    # Breathing room between the plate's edge and the text it carries.
    plate_padding: float = 1.5

    @classmethod
    def defaults(cls, style: str = "interlinear") -> OverlayOptions:
        """The tuned options for one layout.

        The two want different numbers for the same reason they are two
        layouts: `interlinear_compact` is rationing space the author left it,
        so its type is modest and its clearances are shaved to the point of
        being charged twice against a 7 pt band. `interlinear` makes the space
        it needs, so it can afford to be read comfortably and to breathe.
        """
        if style == COMPACT_STYLE:
            return cls()

        return cls(scale=0.78, gap=2.0, max_font_size=28.0)

    def validated(self) -> OverlayOptions:
        if not 0.1 <= self.scale <= 1.5:
            raise OverlayError("scale must be between 0.1 and 1.5")
        if not 1.0 <= self.min_font_size <= self.max_font_size <= 72.0:
            raise OverlayError(
                "font sizes must satisfy 1 <= min_font_size <= max_font_size <= 72")
        if self.align not in ("left", "right"):
            raise OverlayError("align must be one of left, right")
        if not _HEX_COLOR.match(self.color):
            raise OverlayError("color must be a hex colour such as #1a3a5c")
        if not 0.1 <= self.squeeze <= 1.0:
            raise OverlayError("squeeze must be between 0.1 and 1.0")
        if not 0.5 <= self.line_height <= 3.0:
            raise OverlayError("line_height must be between 0.5 and 3.0")
        if not _HEX_COLOR.match(self.plate_color):
            raise OverlayError("plate_color must be a hex colour such as #ffffff")
        if not 0.0 <= self.plate_opacity <= 1.0:
            raise OverlayError("plate_opacity must be between 0 and 1")
        if not 0.0 <= self.plate_padding <= 10.0:
            raise OverlayError("plate_padding must be between 0 and 10")
        return self


class _GlossFont:
    """The font used for one overlay: subset, in memory, plus its CSS.

    The full pan-Unicode face is 15 MB, and PyMuPDF's Story embeds a fresh copy
    of whatever it is given on EVERY box it draws. Subsetting it to the glyphs
    this document's glosses actually use turns that into ~100 KB, which is what
    keeps both the render time and the output file sane (a slide deck went from
    61 MB to under a megabyte). `garbage=4` on the save then collapses the
    per-box copies into one.

    Layout features are kept in the subset (`layout_features = ["*"]`) — they
    ARE the Arabic: without GSUB the shaper has no contextual forms to
    substitute and the text renders as disjoint letters.
    """

    def __init__(self, texts: list[str], options: OverlayOptions) -> None:
        self.archive = pymupdf.Archive()
        self.archive.add(subset_font_bytes(_font_path(), texts), "gloss.ttf")
        self.css = (
            "@font-face {font-family: gloss; src: url(gloss.ttf);}"
            " body {margin: 0;}"
            f" div.gloss {{font-family: gloss; color: {options.color};"
            f" line-height: {options.line_height};}}"
        )


def subset_font_bytes(source: Path, texts: list[str]) -> bytes:
    """`source` subset to what `texts` need — or whole, never refused.

    Shared with `server/page_fonts.py`, which sets the appendix pages in the
    same face (plus its bold) and has the same 15 MB problem to solve.
    """
    try:
        from fontTools import subset
    except ImportError:
        # Correct, just fat and slow. Never a reason to refuse the render.
        logger.warning("fontTools is unavailable; embedding the full gloss "
                       "font (large output)")
        return source.read_bytes()

    try:
        options = subset.Options()
        options.layout_features = ["*"]
        options.name_IDs = ["*"]
        options.notdef_outline = True
        options.drop_tables += ["DSIG"]

        font = subset.load_font(str(source), options)
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(unicodes=_subset_codepoints(texts))
        subsetter.subset(font)

        buffer = BytesIO()
        subset.save_font(font, buffer, options)

        return buffer.getvalue()
    except Exception:  # noqa: BLE001 - subsetting is an optimisation
        logger.exception("subsetting the gloss font failed; using it whole")

        return source.read_bytes()


def _subset_codepoints(texts: list[str]) -> set[int]:
    """Everything the gloss font must still be able to say after subsetting."""
    codepoints = {ord(char) for text in texts for char in text} | _ALWAYS_SUBSET

    if any(code in _ARABIC_BLOCK for code in codepoints):
        codepoints |= _ARABIC_PRESENTATION_FORMS

    return codepoints


def _font_path() -> Path:
    """{@link FONT_FILE} on disk, downloaded and cached by babeldoc.

    The Docker image bakes this cache, so in production it is already there; a
    dev box downloads it once.
    """
    global _FONT_PATH

    if _FONT_PATH is None:
        from babeldoc.assets import assets

        path, _ = assets.get_font_and_metadata(FONT_FILE)
        _FONT_PATH = Path(path)

    return _FONT_PATH


def _rtl_lang(lang: str | None) -> bool:
    lang = (lang or "").lower()

    return any(lang.startswith(code) for code in ("ar", "he", "fa", "ur", "ps", "yi"))


def _css_align(align: str, rtl: bool) -> str:
    """`align` (a VISUAL edge) as the keyword the Story engine wants.

    MuPDF resolves `text-align: left|right` against the writing direction, not
    against the page: inside `dir="rtl"` its `left` means "line start", which
    lands on the visual RIGHT. Callers of this module say where they want the
    text to appear; the flip lives here, once.
    """
    if not rtl:
        return align

    return "right" if align == "left" else "left"


def _gloss_html(text: str, font_size: float, direction: str, align: str) -> str:
    return (
        f'<div class="gloss" dir="{direction}"'
        f' style="font-size:{font_size:.2f}px;text-align:{align}">'
        f"{html.escape(text)}</div>"
    )


def _to_display(box: list[float], matrix: pymupdf.Matrix) -> pymupdf.Rect:
    """One sidecar box (PDF user space, y up) as a page rect (y down)."""
    x0, y0, x1, y1 = box

    return pymupdf.Rect(pymupdf.Point(x0, y1) * matrix,
                        pymupdf.Point(x1, y0) * matrix).normalize()


def _widest_span(anchor: pymupdf.Rect, column_right: float,
                 obstacles: list[pymupdf.Rect]) -> float:
    """How far right a gloss may run before it starts stealing.

    A gloss wider than its paragraph wraps less, and the whitespace to the
    right of a short bullet is usually free — but on a two-column page that
    same whitespace is where the NEXT column's own gloss has to go, and a
    gloss drawn there becomes an obstacle that squeezes it out. So the reach
    stops at the left edge of the next thing that starts to the right,
    whatever its height: horizontal neighbours are columns, and a gloss stays
    inside its own.
    """
    limit = column_right

    for obstacle in obstacles:
        if obstacle.x0 >= anchor.x1 + 2:
            limit = min(limit, obstacle.x0)

    return max(anchor.x1, limit)


def _ceiling(band_x0: float, band_x1: float, floor_y: float,
             obstacles: list[pymupdf.Rect]) -> float:
    """The lowest y a gloss spanning [band_x0, band_x1] may start at.

    Whatever overlaps the band horizontally and BEGINS above its floor sets the
    ceiling; nothing above it means the top of the page. An obstacle that
    begins above the floor and reaches past it returns a ceiling below the
    floor, which the caller reads as "no room" — the alternative, skipping it
    for not being strictly above, would hand the gloss the whole page and draw
    it straight through the line it was supposed to sit under.

    A paragraph never vetoes its own gloss: its box begins at the floor plus
    the clearance gap, so the `y0 < floor_y` test excludes it. The same test
    excludes everything else that starts at or below the floor, which is
    already out of the band's way.
    """
    ceiling = 0.0

    for obstacle in obstacles:
        if obstacle.y0 >= floor_y:
            continue
        if obstacle.x1 <= band_x0 or obstacle.x0 >= band_x1:
            continue

        ceiling = max(ceiling, obstacle.y1)

    return ceiling


def _split_proportionally(text: str, weights: list[float]) -> list[str]:
    """`text` cut into len(weights) chunks, each roughly its share of the whole.

    Word boundaries only, and every chunk is non-empty when there are enough
    words — a blank chunk would leave a source line silently unglossed.
    """
    words = text.split()
    count = len(weights)

    if count <= 1 or len(words) < count:
        return [text]

    total = sum(weights) or float(count)
    chunks: list[str] = []
    start = 0
    taken = 0.0

    for index, weight in enumerate(weights):
        taken += weight
        # Leave at least one word for each chunk still to come.
        end = min(round(len(words) * taken / total), len(words) - (count - index - 1))
        end = max(end, start + 1)
        chunks.append(" ".join(words[start:end]))
        start = end

    return chunks


class _Layout:
    """Measuring and drawing for one overlay run (shared font + direction)."""

    def __init__(self, font: _GlossFont, rtl: bool, options: OverlayOptions) -> None:
        self.font = font
        self.direction = "rtl" if rtl else "ltr"
        self.align = _css_align(options.align, rtl)
        self.options = options

    def html(self, text: str, font_size: float, align: str | None = None) -> str:
        # `align=None` is the configured edge; "center" is what a raster gloss
        # asks for, because a diagram's labels are centred and a gloss hanging
        # off one edge of its plate reads as a mistake. Centring is the same
        # keyword in both writing directions, so it skips the LTR/RTL flip.
        return _gloss_html(text, font_size, self.direction, align or self.align)

    def measure(self, text: str, font_size: float, width: float,
                align: str | None = None) -> float:
        """The height this gloss needs at the width it will be drawn in.

        A bare `Story.place()` rather than insert_htmlbox's `fit_scale`: one
        layout pass instead of a binary search, and every paragraph pays it.
        """
        story = pymupdf.Story(html=self.html(text, font_size, align),
                              user_css=self.font.css, archive=self.font.archive)
        _, filled = story.place(pymupdf.Rect(0, 0, width, 100_000))

        return float(filled[3])

    def fit(self, text: str, width: float, height: float,
            source_size: float) -> tuple[float, float] | None:
        """A font size whose gloss fits the band: `(font_size, height)`.

        Starts at the requested fraction of the source size and steps down; the
        last steps go below the legibility floor by `squeeze` before the gloss
        is given up on. Stepping down is a CONCESSION to the space available,
        never the intent — a paragraph with room gets its full proportional
        size.
        """
        if height <= 0:
            # An obstacle reaching past the paragraph's own top: there is no
            # band, so there is nothing to lay out and measure repeatedly.
            return None

        options = self.options
        size = min(max(source_size * options.scale, options.min_font_size),
                   options.max_font_size)
        floor = options.min_font_size * options.squeeze

        while True:
            needed = self.measure(text, size, width)

            # A hair of slack: the draw re-lays the same text out, and a
            # sub-point rounding difference must not become a failed fit.
            if needed + 0.5 <= height:
                return size, needed + 0.5

            if size <= floor:
                return None

            size = max(floor, size * 0.85)

    def draw(self, page: pymupdf.Page, rect: pymupdf.Rect, text: str,
             font_size: float, align: str | None = None) -> None:
        # scale_low lets the renderer absorb a sub-point disagreement with the
        # measuring pass instead of dropping the gloss; fit() means it is never
        # asked for real shrinking.
        page.insert_htmlbox(rect, self.html(text, font_size, align),
                            css=self.font.css, archive=self.font.archive,
                            scale_low=0.75)


def _placement(layout: _Layout, text: str, anchor: pymupdf.Rect,
               source_size: float, obstacles: list[pymupdf.Rect],
               column_right: float) -> tuple[pymupdf.Rect, float] | None:
    """Where and how big one gloss for `anchor` would be — nothing is drawn.

    Kept separate from drawing so a multi-chunk placement can be rehearsed in
    full and abandoned without leaving ink on the page.
    """
    options = layout.options
    floor_y = anchor.y0 - options.gap

    if floor_y <= 0:
        return None

    # Widest span first (fewer wrapped lines), then the paragraph's own span,
    # whose narrower reach often clears a higher ceiling. Deduplicated because
    # a paragraph with no free space beside it yields the same band twice, and
    # the second pass could only fail the fit ladder all over again.
    bands = dict.fromkeys(((anchor.x0, _widest_span(anchor, column_right, obstacles)),
                           (anchor.x0, anchor.x1)))

    for band_x0, band_x1 in bands:
        if band_x1 - band_x0 < 8:
            continue

        band_top = _ceiling(band_x0, band_x1, floor_y, obstacles) + options.gap
        fitted = layout.fit(text, band_x1 - band_x0, floor_y - band_top, source_size)

        if fitted is not None:
            font_size, height = fitted

            return pymupdf.Rect(band_x0, floor_y - height, band_x1, floor_y), font_size

    return None


def _spread_placements(layout: _Layout, text: str, lines: list[pymupdf.Rect],
                       anchor: pymupdf.Rect, source_size: float,
                       obstacles: list[pymupdf.Rect],
                       column_right: float) -> list[tuple[pymupdf.Rect, float, str]]:
    """Where one paragraph's gloss would go if spread over its own source lines.

    All or nothing — an empty list means "not this way". A paragraph glossed
    only down to its third line would read as a translation with holes in it,
    so a single chunk with nowhere to go abandons the whole arrangement, and
    nothing is drawn until the caller decides.
    """
    ordered = sorted(lines, key=lambda line: line.y0)
    chunks = _split_proportionally(text, [line.width for line in ordered])

    if len(chunks) < 2:
        return []

    # The paragraph's own box is what these lines are made of; leaving it in
    # the obstacle set would have every line's gloss blocked by its own
    # paragraph. The lines themselves take its place.
    scratch = [rect for rect in obstacles if rect is not anchor] + ordered
    placements = []

    for line, chunk in zip(ordered, chunks, strict=False):
        found = _placement(layout, chunk, line, source_size, scratch, column_right)

        if found is None:
            return []

        scratch.append(found[0])
        placements.append((*found, chunk))

    return placements


def _is_backdrop(rect: pymupdf.Rect, page_area: float) -> bool:
    """Is this non-text rect the thing the page is drawn ON, rather than on it?

    The test is size alone, and it has to be applied to every non-text kind:
    a slide's backdrop is a filled rectangle on one deck, a full-bleed JPEG on
    the next, and a BabelDOC figure box on the third. Guarding one kind and not
    the others just moves which decks come back with nothing drawn on them.
    """
    return abs(rect.get_area()) > page_area * _MAX_BACKDROP_SHARE


def _stroke_ink(drawing: dict) -> list[pymupdf.Rect]:
    """A STROKE-ONLY path as the marks it really leaves, not as its bounding box.

    A table's grid and a callout's outline are each ONE path whose bounding box
    is the whole table or the whole callout — while the inside of it is
    transparent, and is exactly where the text those things frame already sits.
    Read the bbox as ink and a slide's own furniture gets a veto over every
    gloss it encloses: a 580x260 pt outlined callout cost all four of its
    bullets their gloss, for a rectangle whose only ink is 1.2 pt wide.

    So a stroked rectangle contributes its four EDGES and never its inside.
    Nothing is conceded to get that: every mark that was actually drawn is
    still an obstacle. Paths that FILL (`f`/`fs`) keep their bounding box —
    a filled shape really is ink all the way across.
    """
    # Exactly the pen, not a courtesy margin around it: the fitting already
    # charges its own clearance on top, and every extra point here is taken
    # straight out of the gloss that has to fit between this stroke and the
    # text under it. A zero-width line is the PDF way of asking for the
    # thinnest mark the device can make, so it gets half a point.
    pad = max(float(drawing.get("width") or 0.0), 0.5) / 2
    edges: list[pymupdf.Rect] = []

    for item in drawing.get("items") or ():
        kind = item[0]

        if kind == "re":
            box = pymupdf.Rect(item[1]).normalize()
            edges += [pymupdf.Rect(box.x0, box.y0, box.x1, box.y0),
                      pymupdf.Rect(box.x0, box.y1, box.x1, box.y1),
                      pymupdf.Rect(box.x0, box.y0, box.x0, box.y1),
                      pymupdf.Rect(box.x1, box.y0, box.x1, box.y1)]
        elif kind == "l":
            edges.append(pymupdf.Rect(item[1], item[2]).normalize())
        elif kind == "qu":
            edges.append(item[1].rect)
        elif kind == "c":
            curve = pymupdf.Rect(item[1], item[1])
            for point in item[2:]:
                curve |= point
            edges.append(curve)
        else:
            # An item kind this PyMuPDF does not describe to us: fall back to
            # the whole path rather than pretend it drew nothing.
            return [pymupdf.Rect(drawing["rect"])]

    # Inflated by the pen: an axis-aligned stroke has a zero-height bounding
    # box, and until its width is given back it reads as EMPTY — which is how
    # a heading's rule, the first thing this was all meant to catch, was
    # silently dropped from the obstacle set entirely.
    return [edge + (-pad, -pad, pad, pad) for edge in edges]


def _page_ink(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """What is ACTUALLY drawn on the page, as the page itself reports it.

    The sidecar's boxes say where each translation BELONGS; they are not a
    reliable account of what is already there. BabelDOC's paragraph box is
    built from character metrics and sits BELOW the glyphs' real tops —
    consistently, and in proportion to the type size (a 40 pt title on a real
    deck overshot its box by 11 pt, a 19 pt bullet by 5). Trusting the box
    alone put a third of that deck's glosses on top of the ascenders they were
    supposed to sit above.

    The overlay has the original PDF in hand, so it can just ask: text spans,
    images, and the vector drawings a slide deck is full of — the rule under a
    heading, a table's borders, a callout's rounded box. Everything that is not
    text is passed through {@link _is_backdrop} first, because a page's
    BACKGROUND is drawn too and would otherwise veto every gloss on the page.
    """
    with _GLYPH_HEIGHTS:
        ink = [pymupdf.Rect(span["bbox"])
               for block in page.get_text("dict")["blocks"] if block["type"] == 0
               for line in block["lines"] for span in line["spans"]
               if span["text"].strip()]

    page_area = abs(page.rect.get_area()) or 1.0

    # One pass for every image PLACEMENT on the page, not one content-stream
    # scan per xref: a deck that reuses a logo on 200 slides would otherwise
    # pay for it 200 times per page.
    try:
        images = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    except (ValueError, RuntimeError):
        # A malformed image reference is not worth failing an overlay over;
        # the sidecar's figure boxes still cover the space.
        images = []

    drawings: list[pymupdf.Rect] = []

    try:
        for drawing in page.get_drawings():
            rect = pymupdf.Rect(drawing["rect"])
            # Stroke-only paths are worth taking apart ({@link _stroke_ink});
            # anything that fills is ink across its whole box.
            #
            # `is_empty` is what keeps this from quietly costing coverage: an
            # AXIS-ALIGNED stroke — a heading's rule, a table's row lines —
            # reports a zero-height box and is dropped by the filter below, so
            # it is not an obstacle today. Taking such a path apart would give
            # its width back and make it one for the first time, which is
            # arguably right and is measurably expensive (a table with a rule
            # 6 pt above each row loses every row's gloss). That is a tuning
            # decision to make against a real document, so this only ever
            # SHRINKS the obstacle set: a path that blocks nothing today
            # carries on blocking nothing.
            drawings += (_stroke_ink(drawing)
                         if drawing.get("type") == "s" and not rect.is_empty
                         else [rect])
    except (ValueError, RuntimeError):
        # Same posture: an unreadable content stream costs the drawing guard,
        # never the render.
        drawings = []

    ink.extend(rect for rect in images + drawings
               if rect.is_valid and not _is_backdrop(rect, page_area))

    return [rect for rect in ink if rect.is_valid and not rect.is_empty]


def _cover_ink(anchor: pymupdf.Rect, source_size: float,
               ink: list[pymupdf.Rect]) -> pymupdf.Rect:
    """`anchor` with its top raised to clear the glyphs it actually holds.

    Only the TOP moves, and only upward, and never by more than the source's
    own font size — the shortfall being corrected is an ascent artifact, so
    anything larger is a neighbour bleeding in rather than this paragraph's
    own ink, and must not be allowed to push the gloss off the page.
    """
    top = anchor.y0
    limit = anchor.y0 - max(source_size, 1.0)

    for rect in ink:
        if rect.x1 <= anchor.x0 or rect.x0 >= anchor.x1:
            continue
        if rect.y0 >= anchor.y1 or rect.y1 <= anchor.y0:
            continue

        top = min(top, max(rect.y0, limit))

    return pymupdf.Rect(anchor.x0, top, anchor.x1, anchor.y1)


def _hex_rgb(color: str) -> tuple[float, float, float]:
    """A validated hex colour as the 0..1 triple pymupdf's drawing wants."""
    digits = color.lstrip("#")

    if len(digits) == 3:
        digits = "".join(char * 2 for char in digits)

    return tuple(int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))


class _RegionInk:
    """One image region's pixels, read as ink the way _page_ink reads objects.

    Inside a raster image nothing exists as an object the page could report:
    the labels, the arrows, the borders are all just pixels. So the region is
    rendered once, and every sharp step between neighbouring pixels is taken as
    a mark somebody drew — text against a fill, a border against the page, an
    arrow against both — while the flat fills and soft gradients a plate is
    allowed to dim stay clear. The marks are dilated by their own anti-aliased
    fringe and summed into an integral image, so the fitting ladder can ask
    about every candidate plate for the price of one render.

    Rendered from the page as it stands MID-OVERLAY, deliberately: anything the
    normal pass already drew near the region shows up as ink and keeps plates
    off it, without this module keeping a second account of its own output.
    """

    def __init__(self, page: pymupdf.Page, clip: pymupdf.Rect) -> None:
        self.clip = clip
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(_RASTER_SCALE, _RASTER_SCALE), clip=clip,
            colorspace=pymupdf.csGRAY, alpha=False)
        grey = (numpy.frombuffer(pix.samples, numpy.uint8)
                .reshape(pix.height, pix.stride)[:, :pix.width]
                .astype(numpy.int16))

        # A step in either direction marks BOTH pixels astride it: the edge
        # belongs to the mark and to the fringe it bleeds into.
        edge = numpy.zeros(grey.shape, dtype=bool)
        step_x = numpy.abs(numpy.diff(grey, axis=1)) > _RASTER_EDGE_STEP
        step_y = numpy.abs(numpy.diff(grey, axis=0)) > _RASTER_EDGE_STEP
        edge[:, :-1] |= step_x
        edge[:, 1:] |= step_x
        edge[:-1, :] |= step_y
        edge[1:, :] |= step_y

        # Grown into a COPY each round: |= between overlapping views of one
        # array cascades in memory order and smears a single edge pixel across
        # the whole row.
        for _ in range(_RASTER_INK_DILATION):
            grown = edge.copy()
            grown[:, :-1] |= edge[:, 1:]
            grown[:, 1:] |= edge[:, :-1]
            grown[:-1, :] |= edge[1:, :]
            grown[1:, :] |= edge[:-1, :]
            edge = grown

        self._integral = numpy.pad(edge.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def inky(self, rect: pymupdf.Rect) -> bool:
        """Does a plate here sit on something that was drawn?"""
        rows, cols = self._integral.shape[0] - 1, self._integral.shape[1] - 1
        x0 = max(int((rect.x0 - self.clip.x0) * _RASTER_SCALE), 0)
        y0 = max(int((rect.y0 - self.clip.y0) * _RASTER_SCALE), 0)
        x1 = min(int((rect.x1 - self.clip.x0) * _RASTER_SCALE) + 1, cols)
        y1 = min(int((rect.y1 - self.clip.y0) * _RASTER_SCALE) + 1, rows)
        area = (x1 - x0) * (y1 - y0)

        if area <= 0:
            # Entirely off the rendered mask: nothing is known about what is
            # there, and unknown pixels are not a place to draw.
            return True

        count = (self._integral[y1, x1] - self._integral[y0, x1]
                 - self._integral[y1, x0] + self._integral[y0, x0])

        return count > area * _RASTER_INK_TOLERANCE


def _raster_placement(layout: _Layout, text: str, anchor: pymupdf.Rect,
                      region: pymupdf.Rect, source_size: float,
                      ink: _RegionInk,
                      taken: list[pymupdf.Rect]) -> tuple[pymupdf.Rect, float] | None:
    """Where one raster gloss's plate would go — nothing is drawn.

    The band directly above the label first — a gloss reads as belonging to
    what it sits over — then the band below it. Each band gets the same fitting
    ladder as a normal gloss, but the judge is different: not "is there
    whitespace", which inside an image there never is, but "are these pixels
    quiet". A size step that would land the plate on a border or a neighbour
    shrinks past it; a label with no quiet pixels either way is given up on,
    and the caller counts it.

    Only the plate comes back: the text's own rect is derived from it at draw
    time by peeling the padding off.
    """
    options = layout.options
    pad = options.plate_padding
    # The label's box is an OCR measurement, and the mask has grown every mark
    # by its dilation on top of that — so a plate standing off by the ordinary
    # gap still lands on the label's own fringe and is vetoed by it. The
    # fringe's width in points is known exactly, and the plate stands off by
    # that much more.
    gap = options.gap + (_RASTER_INK_DILATION + 1) / _RASTER_SCALE

    # The label's own span first; then a wider reach for the label whose
    # translation runs longer than it does. The ink test is what decides
    # whether the borrowed width was actually free.
    stretch = anchor.width * 0.3
    spans = dict.fromkeys((
        (max(region.x0, anchor.x0), min(region.x1, anchor.x1)),
        (max(region.x0, anchor.x0 - stretch), min(region.x1, anchor.x1 + stretch)),
    ))

    start = min(max(source_size * options.scale, options.min_font_size),
                options.max_font_size)
    floor = options.min_font_size * options.squeeze

    for below in (False, True):
        for span_x0, span_x1 in spans:
            if span_x1 - span_x0 - 2 * pad < 8:
                continue

            size = start

            while True:
                # The same hair of slack as fit(): the draw re-lays the text
                # out, and a rounding disagreement must not spill the plate.
                needed = layout.measure(text, size, span_x1 - span_x0 - 2 * pad,
                                        align="center") + 0.5
                height = needed + 2 * pad

                if below:
                    plate = pymupdf.Rect(span_x0, anchor.y1 + gap,
                                         span_x1, anchor.y1 + gap + height)
                else:
                    plate = pymupdf.Rect(span_x0, anchor.y0 - gap - height,
                                         span_x1, anchor.y0 - gap)

                if (plate.y0 >= region.y0 and plate.y1 <= region.y1
                        and not any(plate.intersects(other) for other in taken)
                        and not ink.inky(plate)):
                    return plate, size

                if size <= floor:
                    break

                size = max(floor, size * 0.85)

    return None


def _render_raster(page: pymupdf.Page, blocks: list[dict], layout: _Layout,
                   matrix: pymupdf.Matrix, page_rect: pymupdf.Rect,
                   remap: Callable[[pymupdf.Rect, pymupdf.Rect],
                                   tuple[pymupdf.Rect, pymupdf.Rect]] | None = None,
                   ) -> tuple[int, int]:
    """Draw one page's raster glosses. Returns `(drawn, skipped)`.

    Each block rides its own declared region and never consults the page's
    object-level obstacle set: the region's pixels are the whole truth about
    what may not be covered, and the region's edges are the fence the gloss
    may not leave. Plates already placed are the one thing the pixels cannot
    know about, so they are carried alongside.

    `remap` is the spaced layout's hand: it has rebuilt the page taller and
    moved the artwork down it, so the label and its region are mapped onto the
    new page before anything is judged or drawn.
    """
    options = layout.options
    plate_rgb = _hex_rgb(options.plate_color)
    masks: dict[tuple[float, float, float, float], _RegionInk] = {}
    taken: list[pymupdf.Rect] = []
    drawn = skipped = 0

    for block in blocks:
        anchor = _to_display(block["box"], matrix)
        region = _to_display(block["region"], matrix).normalize()

        if remap is not None:
            anchor, region = remap(anchor, region)

        region = (region & page_rect).normalize()

        if region.is_empty or not region.is_valid:
            skipped += 1
            continue

        key = tuple(region)

        if key not in masks:
            masks[key] = _RegionInk(page, region)

        source_size = float(block.get("font_size") or 0) or anchor.height
        found = _raster_placement(layout, block["target"].strip(), anchor,
                                  region, source_size, masks[key], taken)

        if found is None:
            skipped += 1
            continue

        plate, size = found
        pad = options.plate_padding
        # Plate first, text second: the text must sit ON the plate, and both
        # over the artwork.
        page.draw_rect(plate, color=None, fill=plate_rgb,
                       fill_opacity=options.plate_opacity, radius=_PLATE_RADIUS)
        layout.draw(page, plate + (pad, pad, -pad, -pad),
                    block["target"].strip(), size, align="center")
        taken.append(plate)
        drawn += 1

    return drawn, skipped


def _says_nothing(block: dict) -> bool:
    """Would this block's gloss just repeat the line it sits over?

    A translation run returns a target for every block it was given, and for a
    great many of them the target is the source back again: a slide number, a
    date, a truth table's T/F cells, a table of Java keywords, a formula. On
    the corpus that is 2284 of 6106 blocks — 37 % — and every one of them is
    drawn as a gloss saying exactly what is already printed under it, in the
    spaced layout after opening a band to hold it. So the page grows to
    restate itself.

    Whitespace-normalised equality only. Anything the translator actually
    changed — a different word, a different script, punctuation moved — still
    gets its gloss; this is not a heuristic about what is worth translating,
    it is the observation that a gloss identical to its source is not a gloss.
    """
    target = " ".join((block.get("target") or "").split())

    return bool(target) and target == " ".join((block.get("source") or "").split())


def _gloss_blocks(page_data: dict) -> tuple[list[dict], list[dict]]:
    """One page's blocks worth glossing, as `(normal, raster)`.

    The raster lane — blocks the engine marked as living inside an embedded
    image — is split out BEFORE the normal pass so that pass's anchors,
    obstacles and column edge are computed from exactly the blocks it always
    saw: an old sidecar renders exactly what it always did.
    """
    blocks = [block for block in (page_data.get("blocks") or [])
              if block.get("box") and (block.get("target") or "").strip()
              and not _says_nothing(block)]

    return ([block for block in blocks
             if not (block.get("on_raster") and block.get("region"))],
            [block for block in blocks
             if block.get("on_raster") and block.get("region")])


def _render_page(page: pymupdf.Page, page_data: dict,
                 layout: _Layout) -> tuple[int, int, int, int]:
    """Draw one page's glosses.

    Returns `(drawn, skipped, raster_drawn, raster_skipped)`.
    """
    blocks, raster = _gloss_blocks(page_data)

    if not blocks and not raster:
        return 0, 0, 0, 0

    # The sidecar's coordinates are unrotated PDF user space. Neutralising
    # /Rotate for the duration makes page.transformation_matrix the plain
    # y-flip that maps them, and anything drawn while it is 0 rotates back with
    # the page's own content once it is restored.
    rotation = page.rotation

    if rotation:
        page.set_rotation(0)

    try:
        matrix = page.transformation_matrix
        page_rect = page.rect
        ink = _page_ink(page)
        # One Rect per paragraph, reused as both its anchor and its entry in
        # the obstacle set — _spread has to be able to tell a paragraph's own
        # box apart from every other rect in there. Each is grown to cover the
        # glyphs the sidecar's box falls short of.
        anchors = [
            _cover_ink(_to_display(block["box"], matrix),
                       float(block.get("font_size") or 0)
                       or _to_display(block["box"], matrix).height, ink)
            for block in blocks
        ]
        # The page's own ink joins the obstacle set: it is the truthful account
        # of what a gloss may not cover, and it also catches everything the
        # sidecar never knew about (a caption, a page number, a stray label).
        # The sidecar's own figure boxes take the same backdrop test as the
        # page's — BabelDOC records a full-bleed background as a figure too,
        # and guarding the ink but not the sidecar only moves the veto.
        page_area = abs(page_rect.get_area()) or 1.0
        figures = [rect for rect in
                   (_to_display(box, matrix)
                    for box in (page_data.get("obstacles") or []) if box)
                   if not _is_backdrop(rect, page_area)]
        obstacles = anchors + ink + figures
        # The page's text column: a narrow bullet's gloss may run out to the
        # widest block's edge instead of wrapping three times inside a stub.
        column_right = min(max((rect.x1 for rect in obstacles), default=page_rect.x1),
                           page_rect.x1)

        drawn = skipped = 0

        for block, anchor in zip(blocks, anchors, strict=False):
            text = block["target"].strip()
            source_size = float(block.get("font_size") or 0) or anchor.height
            lines = [_cover_ink(_to_display(line, matrix), source_size, ink)
                     for line in (block.get("lines") or [])]

            single = _placement(layout, text, anchor, source_size, obstacles,
                                column_right)
            spread = (_spread_placements(layout, text, lines, anchor, source_size,
                                         obstacles, column_right)
                      if len(lines) > 1 else [])

            # Both arrangements are rehearsed and the one that reads better
            # wins, measured by the size its smallest chunk had to settle for.
            # A running paragraph's leading cannot take a spread without
            # shrinking it, so the single gloss keeps those; a bullet list's
            # already-empty lines take it at full size, and there the spread is
            # the whole point.
            if spread and (single is None
                           or min(size for _rect, size, _chunk in spread)
                           >= single[1] * _SPREAD_TOLERANCE):
                for rect, font_size, chunk in spread:
                    layout.draw(page, rect, chunk, font_size)
                    obstacles.append(rect)

                # The source lines are obstacles from here on too: without them
                # the paragraph above would gloss into the gap just filled.
                obstacles.extend(lines)
                drawn += 1
                continue

            if single is None:
                skipped += 1
                continue

            rect, font_size = single
            layout.draw(page, rect, text, font_size)
            obstacles.append(rect)
            drawn += 1

        # The raster lane runs LAST: its ink masks are rendered from the page
        # as it now stands, so every normal gloss just drawn is already part of
        # what a plate must keep off.
        raster_drawn = raster_skipped = 0

        if raster:
            raster_drawn, raster_skipped = _render_raster(page, raster, layout,
                                                          matrix, page_rect)

        return drawn, skipped, raster_drawn, raster_skipped
    finally:
        if rotation:
            page.set_rotation(rotation)


# ---------------------------------------------------------------------------
# THE SPACED LAYOUT
#
# `interlinear` never moves the original, so a gloss only gets the whitespace
# the author happened to leave — on a real slide deck that is a few points
# between bullets, which is why its type comes out small and why a paragraph
# with a tight neighbour above it gets no gloss at all.
#
# `interlinear_spaced` gives up the one thing that was costing all of it: the
# page's height. The page is CUT along horizontal lines and the parts below
# each cut slide down, opening a band of exactly the height that line's gloss
# needs. Nothing is scaled, nothing is reflowed, nothing overlaps, and the
# gloss is set at its full proportional size every time.
#
# Three things make that safe to do:
#
# WHERE TO CUT. A cut has to land where the page is vertically constant, since
# the band it opens is painted by stretching a hairline strip taken from the
# cut itself. Through a solid fill, a vertical rule or plain background that is
# invisible; through a glyph or a photograph it is a smear. So the cut is
# snapped to a line that no glyph, image, curve or horizontal rule crosses —
# the interior of a filled box is explicitly fine, which is what lets a code
# block, a callout or a title bar simply grow taller.
#
# WHAT MOVES WITH IT. A page is not always one flow: a slide with a screenshot
# beside its bullets must not have the screenshot sliced by the bullets' cuts.
# The page is first divided into REGIONS — columns split at full-height
# corridors, rows split at full-width gaps, recursively — and each region is
# opened up independently. Columns grow to their tallest member, rows stack.
# Nothing crosses a corridor by construction, so nothing can be torn.
#
# WHERE A GLOSS ATTACHES. Its own line, if a cut can be opened directly above
# it with nothing in between. Failing that the whole paragraph is glossed once
# above its first line, then once above its whole region — a diagram's row of
# labelled boxes cannot be opened up between the boxes, but it can be opened up
# above them, each gloss still standing over the box it belongs to. Only when
# none of that is available does it fall back to `interlinear`'s fit-into-the-
# whitespace placement, and only then can a gloss be skipped.
# ---------------------------------------------------------------------------

# Tolerance for "on the boundary". Ink that merely touches a cut line does not
# cross it: a line's own glyphs start exactly where the cut above them ends.
_CUT_EPS = 0.05

# The strip of page stretched to paint an opened band. Thin enough to be one
# uniform slice of whatever is behind the text, thick enough to survive being
# rasterised at print resolution.
_STRIP_HEIGHT = 0.6

# How far above its line a cut may be looked for, as a fraction of the source
# type size. The gloss has to read as belonging to the line under it, so a cut
# further away than this is not used even when the page would allow it — the
# paragraph- and region-level attachments are the deliberate way to go further.
_CUT_LOOKUP = 0.75

# The cut check's view of the page: its pixels at this many per point. A cut
# asks one question — does the page CHANGE from one row to the next here — and
# 2 px/pt resolves the anti-aliased fringe of a hairline rule to a couple of
# rows, which is all the question needs.
_CUT_SCALE = 2.0

# A grey step this big between vertically neighbouring pixels is something
# drawn, not a gradient. Same reasoning and same number as the raster lane's
# {@link _RASTER_EDGE_STEP}: type and line art clear it by a wide margin, a
# background's soft blend stays under it.
_CUT_EDGE_STEP = 20

# How far up its gap a cut may be walked while looking for clean air, in
# points. A face whose metrics under-report its ascent (run44's subset `35`
# reports a 28 pt title as starting 16 pt below where its glyphs really do)
# leaves the free gap running down into the letters, and the walk is what
# finds the top of them again. Bounded because a gap on a mostly-blank page
# can be hundreds of points tall and every step costs a mask query.
_CUT_WALK = 24.0

# Region splitting. A corridor is the strongest evidence a page has two
# independent flows, so columns are looked for first — but only in a region
# tall enough to hold more than one line of text. Splitting a single line at
# the gap between two of its words would let its own halves drift apart
# vertically, which is the one way this layout could damage a page.
_COL_GAP = 6.0
_ROW_GAP = 5.0
_MIN_COLUMN_HEIGHT = 72.0
_MIN_COLUMN_MARKS = 2
_MIN_COLUMN_SHARE = 0.05
_MAX_REGION_DEPTH = 4

# A single band may not open wider than this share of the original page. Only
# a pathological case reaches it (a whole paragraph glossed into a narrow
# column), and there the gloss is stepped down rather than allowed to push a
# page to five times its height.
_MAX_BAND_SHARE = 0.6


def _text_ink(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Every glyph run on the page as the ink it really leaves.

    A span's bbox is by default the full em band of its font — for Helvetica
    that is 1.075 above the baseline and 0.299 below, against glyphs that
    reach 0.78 and 0.22 — and consecutive lines of large type routinely
    OVERLAP in those terms while there is visibly clear air between them. Cuts
    are looked for in exactly that air, so the padded box would report a solid
    wall of ink down the page and no line would ever get a band.

    `set_small_glyph_heights` is MuPDF's answer and is a GLOBAL switch, so it
    is held for one page's parse at a time and put back. Every other reader of
    a page's text in this module takes the same lock, so nobody is handed the
    other mode's boxes by surprise.
    """
    with _GLYPH_HEIGHTS:
        pymupdf.TOOLS.set_small_glyph_heights(True)

        try:
            blocks = page.get_text("dict")["blocks"]

            return [pymupdf.Rect(span["bbox"])
                    for block in blocks if block["type"] == 0
                    for line in block["lines"] for span in line["spans"]
                    if span["text"].strip()]
        finally:
            pymupdf.TOOLS.set_small_glyph_heights(False)


class _CutRows:
    """Where the page is vertically constant — the only place it may be cut.

    Every account this module takes of what is drawn is an approximation of
    one question: may the page be sliced here and a hairline of it stretched
    over the band that opens? A span's bbox answers it from the FONT's
    metrics, and a font's metrics can lie. run44's title face — the subset
    `AAAAAB+35` — reports its 28 pt title as starting at y 63.0 while the
    glyphs reach up to y 47.2, so the free gap above the title ran 16 pt down
    into the letters, the cut landed inside them, the sliced ascenders were
    left behind above the band, the 0.6 pt strip smeared them down it as gold
    bars, and the gloss was drawn on the wreckage. 16 of that document's 17
    pages.

    So the page is rasterised once and asked directly: a row of pixels that
    differs from the row under it is somewhere the page changes, and a strip
    taken there cannot be stretched invisibly. A vertical rule passing through
    is unchanged from row to row and stays allowed, exactly as
    {@link _cut_items} always intended.
    """

    def __init__(self, page: pymupdf.Page) -> None:
        self.rect = page.rect
        pix = page.get_pixmap(matrix=pymupdf.Matrix(_CUT_SCALE, _CUT_SCALE),
                              colorspace=pymupdf.csGRAY, alpha=False)
        grey = (numpy.frombuffer(pix.samples, numpy.uint8)
                .reshape(pix.height, pix.stride)[:, :pix.width]
                .astype(numpy.int16))
        step = numpy.abs(numpy.diff(grey, axis=0)) > _CUT_EDGE_STEP
        # Row `y` of the integral counts the steps in source rows < y, so a
        # query over [y0, y1) is one subtraction whatever the height.
        self._integral = numpy.pad(step.cumsum(0).cumsum(1).astype(numpy.int64),
                                   ((1, 0), (1, 0)))

    def constant(self, x0: float, x1: float, y0: float, y1: float) -> bool:
        """Is the page unchanging down [y0, y1] across [x0, x1]?"""
        rows, cols = self._integral.shape[0] - 1, self._integral.shape[1] - 1
        left = min(max(int((x0 - self.rect.x0) * _CUT_SCALE), 0), cols)
        right = min(max(int((x1 - self.rect.x0) * _CUT_SCALE) + 1, left), cols)
        # A step between rows n and n+1 sits at index n, so a strip spanning
        # source rows [a, b] is asked about steps [a - 1, b] — the boundaries
        # into and out of it are part of what stretching it would smear.
        top = min(max(int((y0 - self.rect.y0) * _CUT_SCALE) - 1, 0), rows)
        bottom = min(max(int((y1 - self.rect.y0) * _CUT_SCALE) + 1, top), rows)

        if right <= left or bottom <= top:
            # Off the rendered page: nothing is known, and unknown is not a
            # place to cut.
            return False

        return not (self._integral[bottom, right] - self._integral[top, right]
                    - self._integral[bottom, left] + self._integral[top, left])


def _cut_items(drawing: dict) -> list[pymupdf.Rect]:
    """A path as the parts of it a horizontal cut would DAMAGE.

    The interior of an upright rectangle is not one of them, filled or not: a
    cut through the middle of a code block's panel, a callout, a table cell or
    a title bar reopens as more of the same colour, and the box simply comes
    out taller. What cannot survive being stretched is anything whose
    appearance varies down the page — a curve (a rounded corner, a circle), a
    diagonal (a connector, an arrow), and the horizontal edges that make the
    box a box rather than a stripe. Vertical rules stretch perfectly and are
    left out entirely.
    """
    pad = max(float(drawing.get("width") or 0.0), 0.5) / 2
    marks: list[pymupdf.Rect] = []

    for item in drawing.get("items") or ():
        kind = item[0]

        if kind == "re":
            box = pymupdf.Rect(item[1]).normalize()
            marks += [pymupdf.Rect(box.x0, box.y0 - pad, box.x1, box.y0 + pad),
                      pymupdf.Rect(box.x0, box.y1 - pad, box.x1, box.y1 + pad)]
        elif kind == "l":
            start, end = item[1], item[2]

            if abs(start.x - end.x) <= 0.6:
                continue  # a vertical rule: stretches into a longer rule

            marks.append(pymupdf.Rect(min(start.x, end.x), min(start.y, end.y) - pad,
                                      max(start.x, end.x), max(start.y, end.y) + pad))
        elif kind == "qu":
            marks.append(pymupdf.Rect(item[1].rect))
        elif kind == "c":
            curve = pymupdf.Rect(item[1], item[1])

            for point in item[2:]:
                curve |= point

            marks.append(curve + (-pad, -pad, pad, pad))
        else:
            # Undescribed geometry: assume the whole path is at risk.
            return [pymupdf.Rect(drawing["rect"])]

    return marks


def _page_marks(page: pymupdf.Page) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect],
                                             list[pymupdf.Rect]]:
    """`(ink, blockers, backdrops)` for one page, from a single parse of it.

    `ink` is what a gloss may not be drawn over — the same account
    {@link _page_ink} takes, but on the spans' real heights ({@link
    _text_ink}) because this layout works in the air between lines.
    `blockers` is the narrower question of what a CUT may not pass through.
    `backdrops` is what the page is drawn ON — set aside by both, and reported
    because a region that is nothing BUT backdrop has no clear air to open a
    band in and has to be handled differently ({@link _paint_leaf}).
    """
    ink = _text_ink(page)
    blockers = list(ink)
    backdrops: list[pymupdf.Rect] = []

    page_area = abs(page.rect.get_area()) or 1.0

    try:
        images = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    except (ValueError, RuntimeError):
        images = []

    for rect in images:
        if not rect.is_valid:
            continue

        if _is_backdrop(rect, page_area):
            backdrops.append(rect)
        else:
            ink.append(rect)
            blockers.append(rect)

    try:
        drawings = page.get_drawings()
    except (ValueError, RuntimeError):
        drawings = []

    for drawing in drawings:
        rect = pymupdf.Rect(drawing["rect"])

        if not rect.is_valid:
            continue

        if _is_backdrop(rect, page_area):
            backdrops.append(rect)
            continue

        ink += (_stroke_ink(drawing)
                if drawing.get("type") == "s" and not rect.is_empty else [rect])
        blockers += _cut_items(drawing)

    return ([rect for rect in ink if rect.is_valid and not rect.is_empty],
            [rect for rect in blockers if rect.is_valid and rect.height > 0],
            backdrops)


def _merge_spans(spans: list[tuple[float, float]],
                 gap: float) -> list[tuple[float, float]]:
    """`spans` merged into runs, closing anything narrower than `gap`."""
    ordered = sorted(spans)

    if not ordered:
        return []

    runs = [list(ordered[0])]

    for low, high in ordered[1:]:
        if low - runs[-1][1] < gap:
            runs[-1][1] = max(runs[-1][1], high)
        else:
            runs.append([low, high])

    return [(low, high) for low, high in runs]


def _cut_lines(runs: list[tuple[float, float]], low: float,
               high: float) -> list[float]:
    """The midpoints of the gaps between `runs`, inside [low, high]."""
    return [(previous[1] + following[0]) / 2
            for previous, following in zip(runs, runs[1:], strict=False)
            if low < (previous[1] + following[0]) / 2 < high]


@dataclass
class _Band:
    """One cut, the space opened at it, and the glosses that go in it."""

    cut: float
    # Where the hairline of background that paints the band is taken from —
    # the cut itself, unless the cut is somewhere the page has no clear air.
    source: float = 0.0
    strip: float = _STRIP_HEIGHT
    height: float = 0.0
    pending: list[tuple[pymupdf.Rect, str, float]] = field(default_factory=list)
    glosses: list[tuple[float, float, float, float, float, str]] = (
        field(default_factory=list))

    def __post_init__(self) -> None:
        if not self.source:
            self.source = self.cut


@dataclass
class _Region:
    """One independently-openable part of a page.

    A leaf owns a rectangle of the page and the cuts made inside it. A node
    owns children that are either stacked (`rows`, growth adds up) or side by
    side (`columns`, growth is the tallest one). Children always TILE their
    parent, so every point of the page belongs to exactly one leaf and is
    copied to the new page exactly once.
    """

    rect: pymupdf.Rect
    axis: str | None = None
    children: list[_Region] = field(default_factory=list)
    column: pymupdf.Rect | None = None
    bands: list[_Band] = field(default_factory=list)
    offset: float = 0.0
    # The right-hand edge a gloss in this region may run to.
    right: float = 0.0
    # Is this region wall-to-wall backdrop, with nothing drawn on it?
    covered: bool = False
    # Set instead of a padding band for such a region: see {@link _pad_region}.
    stretch: float = 0.0

    @property
    def growth(self) -> float:
        if self.axis == "columns":
            return max((child.growth for child in self.children), default=0.0)
        if self.axis == "rows":
            return sum(child.growth for child in self.children)

        return sum(band.height for band in self.bands)

    def leaves(self) -> list[_Region]:
        if not self.children:
            return [self]

        return [leaf for child in self.children for leaf in child.leaves()]


def _free_gaps(blockers: list[pymupdf.Rect],
               rect: pymupdf.Rect) -> list[tuple[float, float]]:
    """The y ranges inside `rect` that nothing crosses, in order."""
    covered = _merge_spans(
        [(max(mark.y0, rect.y0), min(mark.y1, rect.y1)) for mark in blockers
         if mark.x1 > rect.x0 + _CUT_EPS and mark.x0 < rect.x1 - _CUT_EPS
         and mark.y1 > rect.y0 and mark.y0 < rect.y1], 0.0)

    gaps = []
    y = rect.y0

    for low, high in covered:
        if low > y:
            gaps.append((y, low))

        y = max(y, high)

    if rect.y1 > y:
        gaps.append((y, rect.y1))

    return gaps


def _substantial(runs: list[tuple[float, float]],
                 marks: list[pymupdf.Rect]) -> list[tuple[float, float]]:
    """`runs` with the ones too slight to be a column of their own folded in.

    A page number in the outer margin, a rotated sidebar label, a lone bullet
    glyph: each sits behind a genuine corridor and each would otherwise become
    a column that grows on its own schedule, which is how a slide's page
    number ends up floating halfway up the page while the footer it belongs to
    moved to the bottom.

    Weight, not head count: a screenshot beside the bullets is ONE mark and is
    half the page, and it needs its column more than anything else on it.
    """
    if len(runs) < 2:
        return runs

    def weigh(low: float, high: float) -> tuple[int, float]:
        held = [mark for mark in marks if mark.x1 > low and mark.x0 < high]

        return len(held), sum(abs(mark.get_area()) for mark in held)

    weighed = [weigh(low, high) for low, high in runs]
    total = sum(area for _count, area in weighed) or 1.0
    kept = [run for run, (count, area) in zip(runs, weighed, strict=False)
            if count >= _MIN_COLUMN_MARKS or area >= total * _MIN_COLUMN_SHARE]

    if len(kept) < 2:
        return [(runs[0][0], runs[-1][1])]

    # A folded-in run keeps its space: it is absorbed by whichever kept column
    # it sits beside, so the columns still tile the region.
    return kept


def _split_region(rect: pymupdf.Rect, blockers: list[pymupdf.Rect], depth: int,
                  column: pymupdf.Rect) -> _Region:
    """`rect` divided into the parts of it that can be opened independently."""
    inside = [mark for mark in blockers
              if mark.x1 > rect.x0 + _CUT_EPS and mark.x0 < rect.x1 - _CUT_EPS
              and mark.y1 > rect.y0 + _CUT_EPS and mark.y0 < rect.y1 - _CUT_EPS]

    if depth < _MAX_REGION_DEPTH and inside:
        # Columns first: a corridor running the full height of the region is
        # the strongest evidence that its two sides are independent flows.
        # Height-guarded, so a wide word gap in one line is never mistaken for
        # one — see {@link _MIN_COLUMN_HEIGHT}.
        if rect.height >= _MIN_COLUMN_HEIGHT:
            runs = _substantial(
                _merge_spans([(mark.x0, mark.x1) for mark in inside], _COL_GAP),
                inside)
            edges = _cut_lines(runs, rect.x0, rect.x1)

            if edges:
                bounds = [rect.x0, *edges, rect.x1]
                children = [
                    _split_region(pymupdf.Rect(low, rect.y0, high, rect.y1),
                                  blockers, depth + 1,
                                  pymupdf.Rect(low, rect.y0, high, rect.y1))
                    for low, high in zip(bounds, bounds[1:], strict=False)]

                return _Region(rect, "columns", children, column)

        runs = _merge_spans([(mark.y0, mark.y1) for mark in inside], _ROW_GAP)
        edges = _cut_lines(runs, rect.y0, rect.y1)

        if edges:
            bounds = [rect.y0, *edges, rect.y1]
            children = [
                _split_region(pymupdf.Rect(rect.x0, low, rect.x1, high),
                              blockers, depth + 1, column)
                for low, high in zip(bounds, bounds[1:], strict=False)]

            return _Region(rect, "rows", children, column)

    return _Region(rect, None, [], column)


def _cut_above(line: pymupdf.Rect, gaps: list[tuple[float, float]],
               blockers: list[pymupdf.Rect], limit: float, strict: bool,
               clean: Callable[[float, float], bool] | None = None,
               ) -> tuple[float, float] | None:
    """`(cut, strip)` — where the region may be opened for `line`, or None.

    `strict` is what makes a gloss belong to its line: the cut must be
    reachable from the line with nothing of the page in between, so a gloss is
    never hung above someone else's paragraph. The paragraph- and region-level
    attachments drop it deliberately, having already decided to sit further up.

    `clean(cut, strip)` is the page's own pixels vetting the result
    ({@link _CutRows}). The gap this works in is built from object bounds, and
    where those under-report their ink the gap runs down inside the glyphs —
    so the cut is walked back UP the gap until the page agrees there is
    nothing there. Walking up costs the gloss a little of its closeness to the
    line; cutting through the line costs the reader the line.
    """
    top = line.y0
    found = None

    for low, high in gaps:
        # Gaps run down the region, so the last one that STARTS above the line
        # is the clear air immediately over it. A gap may end a hair below that
        # top — the sidecar's box and the glyphs' own bounds disagree by
        # fractions of a point — which is why the cut is clamped rather than
        # the gap rejected.
        if low >= top - _CUT_EPS:
            break
        if top - high <= limit:
            found = (low, high)

    if found is None:
        return None

    low, high = found
    ceiling = min(high, top)
    strip = min(_STRIP_HEIGHT, ceiling - low)

    if strip < 0.1:
        return None

    cut = ceiling - strip / 2

    if clean is not None:
        floor = max(low + strip / 2, ceiling - strip / 2 - _CUT_WALK)

        while not clean(cut, strip):
            cut -= strip

            if cut < floor:
                return None

    if strict:
        for mark in blockers:
            if mark.x1 <= line.x0 or mark.x0 >= line.x1:
                continue
            if mark.y1 > cut + _CUT_EPS and mark.y0 < top - _CUT_EPS:
                return None

    return cut, strip


def _leaf_for(rect: pymupdf.Rect, leaves: list[_Region]) -> _Region | None:
    """The leaf `rect` belongs to — the one it overlaps most."""
    best = None
    best_area = 0.0

    for leaf in leaves:
        overlap = pymupdf.Rect(rect) & leaf.rect

        if overlap.is_valid and overlap.get_area() > best_area:
            best, best_area = leaf, overlap.get_area()

    return best or (leaves[0] if leaves else None)


def _band_at(leaf: _Region, cut: float, strip: float) -> _Band:
    """The band opened at `cut` in `leaf`, creating or reusing it.

    Two lines whose cuts land within a point of each other share one band
    rather than opening two — on a page whose columns were not split (a table's
    row, a diagram's row of boxes) that is what keeps their glosses on one
    level instead of stepping down the page.
    """
    for band in leaf.bands:
        if abs(band.cut - cut) <= 1.0:
            band.strip = min(band.strip, strip)

            return band

    band = _Band(cut=cut, strip=strip)
    leaf.bands.append(band)

    return band


def _ink_bounds(box: pymupdf.Rect, ink: list[pymupdf.Rect]) -> pymupdf.Rect:
    """`box` replaced by the marks it actually holds.

    The sidecar's boxes come from BabelDOC's character metrics and are only
    approximately where the glyphs are — consistently short at the top, and by
    enough that consecutive lines of the same paragraph OVERLAP each other in
    those terms. This layout works in the air between lines, so it cannot use a
    box that reports a line as starting inside the one above it: a cut would be
    looked for where there is no air, and the gloss would be given up on.

    {@link _cover_ink} is the compact layout's answer and grows a box to cover
    whatever it overlaps — which here would swallow the neighbour it was
    supposed to be told apart from. Ownership is decided by CENTRE instead: a
    mark belongs to the box its middle falls in, and to no other.
    """
    owned = [mark for mark in ink
             if box.y0 <= (mark.y0 + mark.y1) / 2 <= box.y1
             and mark.x1 > box.x0 and mark.x0 < box.x1]

    if not owned:
        return box

    bounds = pymupdf.Rect(owned[0])

    for mark in owned[1:]:
        bounds |= mark

    return bounds


def _spaced_units(block: dict, matrix: pymupdf.Matrix,
                  ink: list[pymupdf.Rect]) -> tuple[pymupdf.Rect, list, float]:
    """One paragraph as `(anchor, [(line, chunk)], source_size)`."""
    anchor = _to_display(block["box"], matrix)
    source_size = float(block.get("font_size") or 0) or anchor.height
    text = block["target"].strip()
    lines = sorted((_ink_bounds(_to_display(line, matrix), ink)
                    for line in (block.get("lines") or []) if line),
                   key=lambda rect: rect.y0)
    anchor = _ink_bounds(anchor, ink)

    if not lines:
        return anchor, [(anchor, text)], source_size

    chunks = _split_proportionally(text, [line.width for line in lines])

    if len(chunks) < len(lines):
        # Too few words to go round: one gloss for the paragraph.
        return anchor, [(lines[0], text)], source_size

    return anchor, list(zip(lines, chunks, strict=False)), source_size


def _strip_at(y: float, gaps: list[tuple[float, float]]) -> tuple[float, float]:
    """`(source_y, height)` of a hairline of pure background to paint with.

    Straddling `y` when `y` is in clear air, and otherwise the middle of the
    roomiest clear air this region has — the band still comes out in the
    region's own background rather than in the new page's white.
    """
    for low, high in gaps:
        if low - _CUT_EPS <= y <= high + _CUT_EPS and high - low >= 0.1:
            strip = min(_STRIP_HEIGHT, high - low)

            return min(max(y, low + strip / 2), high - strip / 2), strip

    widest = max(gaps, key=lambda gap: gap[1] - gap[0], default=None)

    if widest is None or widest[1] - widest[0] < 0.1:
        return y, 0.0

    strip = min(_STRIP_HEIGHT, widest[1] - widest[0])

    return (widest[0] + widest[1]) / 2, strip


def _assign_offsets(region: _Region, offset: float) -> None:
    """Where every region lands on the taller page."""
    region.offset = offset

    if region.axis == "columns":
        for child in region.children:
            _assign_offsets(child, offset)
    elif region.axis == "rows":
        for child in region.children:
            _assign_offsets(child, offset)
            offset += child.growth


def _pad_region(region: _Region, target: float,
                gaps: dict[int, list[tuple[float, float]]],
                backdrops: list[pymupdf.Rect]) -> None:
    """Open `region` by `target` even where its own glosses did not need it.

    Side-by-side columns have to end level: the one that grew least gets the
    difference as empty background at its foot, so a screenshot beside a
    glossed bullet list keeps the page rectangular instead of leaving a step
    in it.
    """
    if region.axis == "columns":
        for child in region.children:
            _pad_region(child, target, gaps, backdrops)

        return

    if region.axis == "rows":
        slack = max(target - region.growth, 0.0)

        for index, child in enumerate(region.children):
            extra = slack if index == len(region.children) - 1 else 0.0
            _pad_region(child, child.growth + extra, gaps, backdrops)

        return

    slack = target - region.growth

    if slack <= 0.01:
        return

    if region.covered and not region.bands:
        region.stretch = slack

        return

    # The foot of the region — but never below where its background stops. A
    # slide's panel ends a few points above the paper's edge, and a padding
    # band opened in that last white strip paints the strip: a white gash down
    # the side of a grey slide, for exactly as far as the column beside it grew.
    foot = min(region.rect.y1, min((backdrop.y1 for backdrop in backdrops
                                    if backdrop.y1 > region.rect.y0
                                    and backdrop.x1 > region.rect.x0
                                    and backdrop.x0 < region.rect.x1),
                                   default=region.rect.y1))
    source, strip = _strip_at(foot, gaps.get(id(region), []))
    region.bands.append(_Band(cut=foot, strip=strip, height=slack, source=source))


def _shift(leaf: _Region, y: float) -> float:
    """How far down the new page a point at source `y` in `leaf` moves."""
    return leaf.offset + sum(band.height for band in leaf.bands
                             if band.cut <= y + _CUT_EPS)


def _copy_strip(target: pymupdf.Page, source: pymupdf.Document, index: int,
                clip: pymupdf.Rect, offset: float) -> None:
    """One horizontal slice of the original, unscaled, `offset` further down.

    PyMuPDF embeds the source page once and refers back to it, so a page cut
    into fifty slices costs fifty short references, not fifty copies of a
    slide.
    """
    if clip.height <= _CUT_EPS or clip.width <= _CUT_EPS:
        return

    target.show_pdf_page(clip + (0, offset, 0, offset), source, index,
                         clip=clip, keep_proportion=False)


def _paint_band(target: pymupdf.Page, source: pymupdf.Document, index: int,
                band: _Band, rect: pymupdf.Rect, offset: float) -> None:
    """Fill an opened band with the background it was opened in.

    A hairline taken from the cut itself, stretched over the band. Behind text
    that is a flat colour, a gradient stripe or a photograph's local tone — in
    every case what the reader would have seen if the author had left the space
    there, and never a white gash across a dark slide.
    """
    if band.height <= _CUT_EPS or band.strip <= 0:
        return

    top = band.cut + offset

    target.show_pdf_page(
        pymupdf.Rect(rect.x0, top, rect.x1, top + band.height), source, index,
        clip=pymupdf.Rect(rect.x0, band.source - band.strip / 2,
                          rect.x1, band.source + band.strip / 2),
        keep_proportion=False)


def _paint_leaf(target: pymupdf.Page, source: pymupdf.Document, index: int,
                leaf: _Region, layout: _Layout) -> None:
    """One region rebuilt on the taller page: its slices, bands and glosses."""
    rect = leaf.rect
    offset = leaf.offset
    y = rect.y0

    if leaf.stretch:
        # Wall-to-wall backdrop and nothing to gloss — a title slide's
        # full-bleed photograph beside the text that grew. It has no clear air
        # in it to open a band with, so the whole region is drawn taller
        # instead: a photograph 20% taller reads as a photograph, where a
        # hairline of it smeared across a band reads as damage.
        target.show_pdf_page(
            pymupdf.Rect(rect.x0, rect.y0 + offset, rect.x1,
                         rect.y1 + offset + leaf.stretch),
            source, index, clip=rect, keep_proportion=False)

        return

    for band in sorted(leaf.bands, key=lambda item: item.cut):
        _copy_strip(target, source, index,
                    pymupdf.Rect(rect.x0, y, rect.x1, band.cut), offset)
        _paint_band(target, source, index, band, rect, offset)

        top = band.cut + offset

        for x0, x1, above, height, font_size, text in band.glosses:
            layout.draw(target,
                        pymupdf.Rect(x0, top + above, x1, top + above + height + 0.5),
                        text, font_size)

        y = band.cut
        offset += band.height

    _copy_strip(target, source, index,
                pymupdf.Rect(rect.x0, y, rect.x1, rect.y1), offset)


def _levels(entries: list) -> list[list]:
    """`entries` dealt onto rows that do not overlap each other horizontally.

    Almost always one row: a band belongs to the line under it and holds that
    line's gloss. It is more when several paragraphs attach to the same cut —
    a diagram's row of labelled boxes, whose glosses sit side by side over the
    boxes they name, and the degenerate case where two of them would sit ON
    each other, which is what the rows are really for. Reading order (down the
    page, then across) decides who goes first.
    """
    rows: list[list] = []

    for entry in sorted(entries, key=lambda item: (item[0].y0, item[0].x0)):
        for row in rows:
            if all(entry[0].x0 >= placed[0].x1 or entry[0].x1 <= placed[0].x0
                   for placed in row):
                row.append(entry)
                break
        else:
            rows.append([entry])

    return rows


def _size_bands(leaves: list[_Region], layout: _Layout, page_height: float) -> None:
    """Decide every gloss's type size and width, and how far each band opens.

    Sizing is the point of this layout: the size is the requested fraction of
    the source's, full stop, and the page is opened to whatever that needs.
    The only stepping down left is {@link _MAX_BAND_SHARE}, which exists so one
    pathological paragraph cannot triple a page on its own.
    """
    options = layout.options
    cap = page_height * _MAX_BAND_SHARE
    floor = options.min_font_size * options.squeeze

    for leaf in leaves:
        for band in leaf.bands:
            top = options.gap

            for row in _levels(band.pending):
                row.sort(key=lambda entry: entry[0].x0)
                measured = []

                for position, (line, text, source_size) in enumerate(row):
                    # A gloss reaches to the next gloss on its own row, or to
                    # the edge of its column — wider means fewer wrapped lines,
                    # which is the difference between opening the page by one
                    # line and opening it by three.
                    edge = (row[position + 1][0].x0 - options.gap
                            if position + 1 < len(row) else leaf.right)
                    x1 = max(min(edge, leaf.right), line.x1)
                    width = max(x1 - line.x0, 8.0)

                    font_size = min(max(source_size * options.scale,
                                        options.min_font_size),
                                    options.max_font_size)
                    needed = layout.measure(text, font_size, width)

                    while needed + top + 2 * options.gap > cap and font_size > floor:
                        font_size = max(floor, font_size * 0.85)
                        needed = layout.measure(text, font_size, width)

                    measured.append((line.x0, x1, needed, font_size, text))

                height = max(needed for _x0, _x1, needed, _size, _text in measured)

                # Every gloss hangs from the BOTTOM of its row, so it stays
                # against the line it belongs to however tall its neighbour had
                # to become — one three-word column header next to one that
                # wrapped five times must not be left floating.
                band.glosses += [(x0, x1, top + height - needed, needed, size, text)
                                 for x0, x1, needed, size, text in measured]
                top += height + options.gap

            band.height = max(band.height, top)


def _spaced_page(target: pymupdf.Document, source: pymupdf.Document, index: int,
                 page_data: dict,
                 layout: _Layout) -> tuple[int, int, int, int]:
    """Rebuild one page, opened up.

    Returns `(drawn, skipped, raster_drawn, raster_skipped)`.
    """
    page = source[index]
    rect = page.rect
    options = layout.options
    ink, blockers, backdrops = _page_marks(page)
    root = _split_region(rect, blockers, 0, rect)
    leaves = root.leaves()
    gaps = {id(leaf): _free_gaps(blockers, leaf.rect) for leaf in leaves}
    # The page's own pixels, which every cut is checked against before it is
    # committed. One render per page, shared by every leaf on it.
    rows = _CutRows(page)

    def cleaner(leaf: _Region) -> Callable[[float, float], bool]:
        # The strip is copied across the whole leaf, so the whole leaf's width
        # is what has to be constant — not just the line's own span.
        def clean(cut: float, strip: float) -> bool:
            return rows.constant(leaf.rect.x0, leaf.rect.x1,
                                 cut - strip / 2, cut + strip / 2)

        return clean

    clean_for = {id(leaf): cleaner(leaf) for leaf in leaves}
    reach: dict[int, float] = {}

    for leaf in leaves:
        column = leaf.column or leaf.rect
        key = id(column)

        if key not in reach:
            reach[key] = max((mark.x1 for mark in blockers
                              if mark.x1 > column.x0 and mark.x0 < column.x1),
                             default=column.x1)

        leaf.right = min(leaf.rect.x1, max(reach[key], leaf.rect.x0 + 8.0))
        # "Nothing here but the background" — not merely "the background
        # reaches here", which on any page with a full-bleed fill is every
        # region on it.
        leaf.covered = (
            not any((leaf.rect & mark).is_valid and not (leaf.rect & mark).is_empty
                    for mark in ink)
            and any((leaf.rect & backdrop).is_valid
                    and abs((leaf.rect & backdrop).get_area())
                    > 0.9 * abs(leaf.rect.get_area())
                    for backdrop in backdrops))

    # The raster lane's blocks live INSIDE an embedded image. No cut can pass
    # through an image (it is a blocker), so opening the page cannot make room
    # for them — they keep the plate treatment, drawn on the rebuilt page after
    # everything has landed where it is going to stay.
    blocks, raster = _gloss_blocks(page_data)
    matrix = page.transformation_matrix
    anchors: list[pymupdf.Rect] = []
    fallbacks: list[tuple[pymupdf.Rect, str, float]] = []
    drawn = skipped = 0

    for block in blocks:
        anchor, units, source_size = _spaced_units(block, matrix, ink)
        text = block["target"].strip()
        anchors.append(anchor)
        limit = max(2.0, _CUT_LOOKUP * source_size)
        found = []

        for line, _chunk in units:
            leaf = _leaf_for(line, leaves)
            found.append((leaf, _cut_above(line, gaps[id(leaf)], blockers,
                                           limit, True, clean_for[id(leaf)])))

        if all(cut for _leaf, cut in found):
            # Line by line: the gloss sits directly over the words it renders.
            attached = [(leaf, cut, line, chunk)
                        for (leaf, cut), (line, chunk) in zip(found, units, strict=False)]
        elif found[0][1] is not None:
            # One line of it could not be reached — gloss the paragraph once,
            # above its first line, rather than leave the translation in pieces.
            attached = [(found[0][0], found[0][1], units[0][0], text)]
        else:
            # Nowhere inside the paragraph's own surroundings: go above the
            # whole region. A diagram's row of boxes cannot be opened up
            # between the boxes; it can be opened up over them.
            line = units[0][0]
            leaf = _leaf_for(line, leaves)
            cut = _cut_above(line, gaps[id(leaf)], blockers,
                             line.y0 - leaf.rect.y0 + 1.0, False,
                             clean_for[id(leaf)])
            attached = [(leaf, cut, line, text)] if cut else None

        if attached is None:
            fallbacks.append((anchor, text, source_size))
            continue

        for leaf, (cut, strip), line, chunk in attached:
            _band_at(leaf, cut, strip).pending.append((line, chunk, source_size))

        drawn += 1

    _size_bands(leaves, layout, rect.height)
    _pad_region(root, root.growth, gaps, backdrops)
    _assign_offsets(root, 0.0)

    new_page = target.new_page(width=rect.width, height=rect.height + root.growth)

    for leaf in leaves:
        _paint_leaf(new_page, source, index, leaf, layout)

    if fallbacks:
        # Whatever is left is placed the way `interlinear` places everything:
        # into the whitespace that was already there. Every cut and every
        # region boundary joins the obstacle set, so a gloss fitted this way
        # cannot straddle one and be torn in half by the move.
        figures = [figure for figure in
                   (_to_display(box, matrix)
                    for box in (page_data.get("obstacles") or []) if box)
                   if not _is_backdrop(figure, abs(rect.get_area()) or 1.0)]
        seams = [pymupdf.Rect(rect.x0, edge, rect.x1, edge)
                 for leaf in leaves
                 for edge in (leaf.rect.y0, *(band.cut for band in leaf.bands))]
        obstacles = anchors + ink + figures + seams
        column_right = min(max((mark.x1 for mark in obstacles), default=rect.x1),
                           rect.x1)

        for anchor, text, source_size in fallbacks:
            placed = _placement(layout, text, anchor, source_size, obstacles,
                                column_right)

            if placed is None:
                skipped += 1
                continue

            box, font_size = placed
            leaf = _leaf_for(anchor, leaves)
            shift = _shift(leaf, anchor.y0)
            layout.draw(new_page, box + (0, shift, 0, shift), text, font_size)
            obstacles.append(box)
            drawn += 1

    raster_drawn = raster_skipped = 0

    if raster:
        def remap(anchor: pymupdf.Rect,
                  region: pymupdf.Rect) -> tuple[pymupdf.Rect, pymupdf.Rect]:
            # The map is the ANCHOR's, not the region's. A furniture image is
            # a blocker no cut passes through, so anchor and region share one
            # rigid shift — but a full-page backdrop is not a blocker: the
            # cuts slice straight through it and each label inside it moves
            # with its own strip. The label's own leaf knows that move
            # exactly. The region is only a fence, so its edges are mapped
            # through the same leaf one by one, letting a sliced backdrop's
            # fence stretch over everything it now spans.
            leaf = _leaf_for(anchor, leaves)

            if leaf.stretch:
                grow = ((leaf.rect.height + leaf.stretch)
                        / (leaf.rect.height or 1.0))
                base = leaf.rect.y0 + leaf.offset - grow * leaf.rect.y0

                def place(y: float) -> float:
                    return grow * y + base
            else:
                def place(y: float) -> float:
                    return y + _shift(leaf, y)

            dy = place(anchor.y0) - anchor.y0

            return (anchor + (0, dy, 0, dy),
                    pymupdf.Rect(region.x0, place(region.y0),
                                 region.x1, place(region.y1)))

        raster_drawn, raster_skipped = _render_raster(
            new_page, raster, layout, matrix, new_page.rect, remap)

    return drawn, skipped, raster_drawn, raster_skipped


def render_spaced(original_bytes: bytes, sidecar: dict,
                  layout: _Layout) -> tuple[pymupdf.Document, dict]:
    """The whole document rebuilt page by page, opened up for its glosses."""
    source = pymupdf.open(stream=BytesIO(original_bytes), filetype="pdf")

    try:
        by_number = {page["page_number"]: page for page in sidecar["pages"]
                     if isinstance(page, dict)
                     and isinstance(page.get("page_number"), int)}

        for number in by_number:
            if not 0 <= number < source.page_count:
                raise OverlayError(
                    f"the sidecar names page {number!r}, which the original "
                    f"({source.page_count} pages) does not have — the sidecar "
                    "and the PDF are not from the same run")

        target = pymupdf.open()
        drawn = skipped = raster_drawn = raster_skipped = touched = 0

        try:
            for index in range(source.page_count):
                page_data = by_number.get(index)

                if page_data is None:
                    # A page the run had nothing to say about is carried over
                    # whole, keeping its links and annotations.
                    target.insert_pdf(source, from_page=index, to_page=index)
                    continue

                if source[index].rotation:
                    # /Rotate turns the page in the viewer without moving a
                    # single mark in the file, so the axis this layout opens
                    # the page along is not the axis the reader sees lines
                    # stacked on — a cut meant to pass between two lines would
                    # pass THROUGH every one of them. The page is carried over
                    # and glossed the way `interlinear_compact` does it, which
                    # works in the rotated frame and leaves the page alone.
                    target.insert_pdf(source, from_page=index, to_page=index)
                    page_counts = _render_page(
                        target[target.page_count - 1], page_data, layout)
                else:
                    page_counts = _spaced_page(target, source, index,
                                               page_data, layout)
                drawn += page_counts[0]
                skipped += page_counts[1]
                raster_drawn += page_counts[2]
                raster_skipped += page_counts[3]
                touched += 1

            return target, {"pages": touched, "drawn": drawn, "skipped": skipped,
                            "raster_drawn": raster_drawn,
                            "raster_skipped": raster_skipped}
        except Exception:
            target.close()
            raise
    finally:
        source.close()


def _validate(sidecar: Any) -> dict:
    if not isinstance(sidecar, dict):
        raise OverlayError("the sidecar must be a JSON object")

    version = sidecar.get("version")

    if version not in SUPPORTED_SIDECAR_VERSIONS:
        raise OverlayError(f"unsupported sidecar version {version!r}; this "
                           f"engine reads {SUPPORTED_SIDECAR_VERSIONS}")

    if not isinstance(sidecar.get("pages"), list):
        raise OverlayError("the sidecar has no pages")

    return sidecar


def _targets(sidecar: dict) -> list[str]:
    return [block["target"] for page in sidecar["pages"]
            if isinstance(page, dict)
            for block in (page.get("blocks") or [])
            if isinstance(block, dict) and block.get("target")]


def _open_original(original_bytes: bytes) -> pymupdf.Document:
    try:
        doc = pymupdf.open(stream=BytesIO(original_bytes), filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - pymupdf raises many error types
        raise OverlayError(f"the original is not a readable PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise OverlayError("the original has no pages")

    if doc.page_count > MAX_PAGES:
        pages = doc.page_count
        doc.close()
        raise OverlayError(f"the original has {pages} pages (max {MAX_PAGES})")

    return doc


def render_compact(original_bytes: bytes, sidecar: dict,
                   layout: _Layout) -> tuple[pymupdf.Document, dict]:
    """The original, untouched, with each gloss drawn in the space above it."""
    doc = _open_original(original_bytes)

    try:
        drawn = skipped = raster_drawn = raster_skipped = pages = 0

        for page_data in sidecar["pages"]:
            if not isinstance(page_data, dict):
                continue

            index = page_data.get("page_number")

            if not isinstance(index, int) or not 0 <= index < doc.page_count:
                # A sidecar page with no counterpart means the two files are
                # not from the same run — better said out loud than half drawn.
                raise OverlayError(
                    f"the sidecar names page {index!r}, which the original "
                    f"({doc.page_count} pages) does not have — the sidecar and "
                    "the PDF are not from the same run")

            page_counts = _render_page(doc[index], page_data, layout)
            drawn += page_counts[0]
            skipped += page_counts[1]
            raster_drawn += page_counts[2]
            raster_skipped += page_counts[3]
            pages += 1

        return doc, {"pages": pages, "drawn": drawn, "skipped": skipped,
                     "raster_drawn": raster_drawn,
                     "raster_skipped": raster_skipped}
    except Exception:
        doc.close()
        raise


def render_overlay(original_bytes: bytes, sidecar: Any,
                   style: str = "interlinear",
                   options: OverlayOptions | None = None,
                   vocab: bool = True) -> tuple[bytes, dict]:
    """Draw `style` over the original PDF from its translation sidecar.

    Returns the PDF bytes and a small report — pages touched, glosses drawn,
    glosses that had nowhere to go, with the raster lane's counts on their own
    keys — so a caller can tell a good fit from a document that quietly got
    nothing.

    `vocab=False` is the caller opting out of the «كلمات هذه الصفحة» layer
    for this render: the sidecar's "vocab" is simply not attached. The
    config.VOCAB_PAGES kill switch keeps overriding everything to off.
    """
    if style not in OVERLAY_STYLES:
        raise OverlayError(f"style must be one of {OVERLAY_STYLES}")

    sidecar = _validate(sidecar)
    options = (options or OverlayOptions.defaults(style)).validated()
    targets = _targets(sidecar)

    if not targets:
        raise OverlayError("the sidecar carries no translated text")

    rtl = _rtl_lang(sidecar.get("lang_out"))
    layout = _Layout(_GlossFont(targets, options), rtl, options)
    build = render_compact if style == COMPACT_STYLE else render_spaced
    doc, report = build(original_bytes, sidecar, layout)

    try:
        # A sidecar that carries "vocab" (server/vocab.py wrote it after the
        # mono run) gets the same «كلمات هذه الصفحة» treatment the mono
        # result has: each original page grows a compact bottom strip of its
        # own new words (a page the strip cannot serve — rotated, too small —
        # falls back to an inserted page after it) — whichever lane (spaced
        # or compact) built it. Best-effort: an overlay must not fail over
        # its vocab layer. Lazy import — vocab_pages draws on the shared
        # page-fonts machinery, which imports this module for the subsetter.
        vocab_added = 0

        if (config.VOCAB_PAGES and vocab
                and isinstance(sidecar.get("vocab"), dict)):
            from server import vocab_pages

            anchors = {}

            for key in sidecar["vocab"]:
                try:
                    number = int(key)
                except (TypeError, ValueError):
                    continue

                if 0 <= number < doc.page_count:
                    anchors[number] = number

            try:
                # How many content pages got their words (strip or fallback
                # page) — not a page count, since a strip adds no page.
                vocab_added = len(vocab_pages.attach_vocab(
                    doc, sidecar["vocab"], anchors))
            except Exception:  # noqa: BLE001 - the vocab layer is optional
                logger.exception("interlinear: attaching vocab failed; "
                                 "returning the overlay without it")

        report["vocab_pages"] = vocab_added

        out = BytesIO()
        # garbage=4 is what collapses the one-font-copy-per-box the Story
        # engine leaves behind into a single embedded subset.
        doc.save(out, garbage=4, deflate=True)

        if report["skipped"] or report["raster_skipped"]:
            logger.info("%s: %s gloss(es) and %s raster gloss(es) had no room "
                        "and were skipped (%s + %s drawn over %s page(s))",
                        style, report["skipped"], report["raster_skipped"],
                        report["drawn"], report["raster_drawn"], report["pages"])

        return out.getvalue(), report
    finally:
        doc.close()
