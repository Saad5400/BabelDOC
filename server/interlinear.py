"""Stateless overlay layouts: one paid translation, many ways to read it.

`compose.py` rebuilds the alternating / side-by-side duals by shuffling PDF
pages, which is all those layouts are. This module rebuilds the one that needs
to know what the translation SAYS and WHERE it belongs — and it gets that from
the run's sidecar (`document_il/midend/translation_sidecar.py`), never from a
second translation.

`interlinear` leaves the original page completely untouched and draws each
paragraph's translation small in the whitespace directly above it, the way a
gloss sits above a line in an interlinear text. The reader keeps the slide they
already know and gets the Arabic in the same glance — which neither the
Arabic-only PDF (the original is gone) nor the duals (the two are pages apart)
can do.

FITTING is the whole problem, because the original may not move. A gloss gets
the vertical band between its paragraph and whatever sits above it, and is
sized to that band: the requested fraction of the source font size first, then
stepped down, and finally force-fitted by the renderer if a step still spills.
A paragraph with no usable band above it is SKIPPED rather than drawn over the
reader's document, and the count comes back to the caller — a bad fit should be
visible, not silent.

When one band cannot hold a gloss at a readable size, the gloss is SPREAD down
the paragraph's own source lines instead of shrunk into illegibility — the
sidecar carries where those lines sit. This is what makes a slide's bullet list
work: BabelDOC merges a run of same-styled bullets into one paragraph, so the
whole list arrives as a single block whose only free space is the sliver above
the first bullet, and each bullet has an empty line above it going unused. The
split is PROPORTIONAL, not aligned — the translation is one sentence and no
word of it is claimed to belong to a particular source line — but it reads in
order, top to bottom, which one crushed 4 pt paragraph does not.

TEXT RENDERING is PyMuPDF's Story engine, which shapes Arabic into its
contextual forms and runs the bidi algorithm over it — including the Latin
technical terms the glossary deliberately keeps in Latin script. The sidecar
carries logical text, so this holds for any target language, RTL or not.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf

from server import config

logger = logging.getLogger("doctranslate.interlinear")

OVERLAY_STYLES = ("interlinear",)

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

    Shared with `server/glossary_pages.py`, which sets its cards in the same
    face (plus its bold) and has the same 15 MB problem to solve.
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

    def html(self, text: str, font_size: float) -> str:
        return _gloss_html(text, font_size, self.direction, self.align)

    def measure(self, text: str, font_size: float, width: float) -> float:
        """The height this gloss needs at the width it will be drawn in.

        A bare `Story.place()` rather than insert_htmlbox's `fit_scale`: one
        layout pass instead of a binary search, and every paragraph pays it.
        """
        story = pymupdf.Story(html=self.html(text, font_size),
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
             font_size: float) -> None:
        # scale_low lets the renderer absorb a sub-point disagreement with the
        # measuring pass instead of dropping the gloss; fit() means it is never
        # asked for real shrinking.
        page.insert_htmlbox(rect, self.html(text, font_size), css=self.font.css,
                            archive=self.font.archive, scale_low=0.75)


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


def _render_page(page: pymupdf.Page, page_data: dict,
                 layout: _Layout) -> tuple[int, int]:
    """Draw one page's glosses. Returns `(drawn, skipped)`."""
    blocks = [block for block in (page_data.get("blocks") or [])
              if block.get("box") and (block.get("target") or "").strip()]

    if not blocks:
        return 0, 0

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

        return drawn, skipped
    finally:
        if rotation:
            page.set_rotation(rotation)


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


def render_overlay(original_bytes: bytes, sidecar: Any,
                   style: str = "interlinear",
                   options: OverlayOptions | None = None) -> tuple[bytes, dict]:
    """Draw `style` over the original PDF from its translation sidecar.

    Returns the PDF bytes and a small report — pages touched, glosses drawn,
    glosses that had nowhere to go — so a caller can tell a good fit from a
    document that quietly got nothing.
    """
    if style not in OVERLAY_STYLES:
        raise OverlayError(f"style must be one of {OVERLAY_STYLES}")

    sidecar = _validate(sidecar)
    options = (options or OverlayOptions()).validated()

    try:
        doc = pymupdf.open(stream=BytesIO(original_bytes), filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - pymupdf raises many error types
        raise OverlayError(f"the original is not a readable PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise OverlayError("the original has no pages")
        if doc.page_count > MAX_PAGES:
            raise OverlayError(
                f"the original has {doc.page_count} pages (max {MAX_PAGES})")

        targets = _targets(sidecar)

        if not targets:
            raise OverlayError("the sidecar carries no translated text")

        rtl = _rtl_lang(sidecar.get("lang_out"))
        layout = _Layout(_GlossFont(targets, options), rtl, options)

        drawn = skipped = pages = 0

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

            page_drawn, page_skipped = _render_page(doc[index], page_data, layout)
            drawn += page_drawn
            skipped += page_skipped
            pages += 1

        # A sidecar that carries "glossary" entries (server/terms.py wrote them
        # after the mono run) gets the same «شرح المصطلحات» pages the mono
        # result has, appended after the overlaid document. Best-effort: an
        # overlay must not fail over its appendix. Lazy import — glossary_pages
        # imports this module for the font subsetter.
        glossary_added = 0

        if config.GLOSSARY_PAGES and sidecar.get("glossary"):
            from server import glossary_pages

            try:
                glossary_added = glossary_pages.append_glossary_pages(
                    doc, sidecar["glossary"])
            except Exception:  # noqa: BLE001 - the appendix is optional
                logger.exception("interlinear: appending glossary pages failed;"
                                 " returning the overlay without them")

        out = BytesIO()
        # garbage=4 is what collapses the one-font-copy-per-box the Story
        # engine leaves behind into a single embedded subset.
        doc.save(out, garbage=4, deflate=True)

        if skipped:
            logger.info("interlinear: %s gloss(es) had no room and were skipped "
                        "(%s drawn over %s page(s))", skipped, drawn, pages)

        return out.getvalue(), {"pages": pages, "drawn": drawn,
                                "skipped": skipped,
                                "glossary_pages": glossary_added}
    finally:
        doc.close()
