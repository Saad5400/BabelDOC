#!/usr/bin/env python
"""Should this text-inside-an-image be glossed at all? — one decision.

Text that lives inside an embedded raster (a diagram's labels, a screenshot's
UI chrome) reaches the reader through two very different lanes:

  * the MONO lane — server/image_prep.py injects an invisible OCR run over
    the pixels, BabelDOC's paragraph_finder turns it into an `image_text`
    paragraph, paints a background-matched MASK over the source glyphs and
    typesets the translation in their place; and
  * the INTERLINEAR lane — the same blocks arrive in the sidecar marked
    `on_raster`, and server/interlinear.py lays a translucent PLATE on the
    artwork with the gloss on it.

Both lanes destroy pixels to make room for words, and both were doing it in
places where the words were not worth the pixels: a university wordmark read
as `ll` / `ola_cola` and "translated" into `أولا _ كولا`, a blurred stock
photo of code read as `FalseFalse`, a Windows PATH dialog read as twelve
overlapping fragments whose masks and glosses shredded each other.

The judgement is the same in both lanes, so it is made once, here, and both
lanes are thin callers of {@link gloss_plan}. The MONO lane consults it in
image_prep — the single point where OCR text becomes a run at all, so a
region refused there never reaches paragraph_finder, the sidecar, or either
renderer. The INTERLINEAR lane consults it again at draw time, because a
sidecar built before this gate existed is still downloadable and must not
render the garbage it recorded; a region refused there is counted as
`raster_skipped`, exactly like a label with nowhere to go.

Two facts are lane-specific and therefore optional: OCR CONFIDENCE and image
SHARPNESS are known only where the pixels are (image_prep). Passing them is
strictly better; passing None makes those rules inert rather than wrong.

Every threshold is a module constant with the measurement behind it, so the
one place to retune the gate is the top of this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field

# ---------------------------------------------------------------------------
# thresholds
#
# Numbers come from the five production regions that produced the defects
# (job 136 p1's logo, job 154 p1's blurred photo and pp.33/37's code
# screenshots, job 143 p6's PATH dialog) measured against the regions on the
# same pages that render correctly. Each is set in the middle of the gap
# between the two populations, never at the edge of one.

# --- text quality -----------------------------------------------------------

# Mean tesseract confidence over a segment's words. ocr_prep already kills
# individual words under 60; a SEGMENT that only just clears that is a
# reading of something that is not text ('‘%SYSTEMROOT' scored 65).
GATE_MIN_CONF = 70.0

# A token is "wordish" when it could be a word in any language written in
# Latin script: three letters or more, at least one vowel, no letter tripled.
# `ola` and `cola` pass; `ll`, `ol`, `Rrrr`, `bcdf` do not.
GATE_WORDISH_MIN_LEN = 3
GATE_VOWELS = frozenset("aeiouyAEIOUY")

# With three or more tokens, this share of them must be wordish. Below it the
# "text" is a scatter of glyph shards that happen to sit on a line.
GATE_MIN_WORDISH_FRACTION = 0.4

# Letters already in the target script. A bilingual wordmark's Arabic half,
# or a caption the deck already translated, must not be round-tripped.
GATE_TARGET_SCRIPT_SHARE = 0.3
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_TOKEN = re.compile(r"[A-Za-z]+")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
# One token written in two scripts at once is an OCR fusion of a glyph and a
# mark, never a word.
_MIXED_SCRIPT_TOKEN = re.compile(
    r"[A-Za-z][\u0600-\u06FF]|[\u0600-\u06FF][A-Za-z]"
)

# --- the image as a whole ---------------------------------------------------

# A placement this small, sitting wholly inside a page corner, is furniture:
# a wordmark, a faculty crest, a sponsor's badge. Job 136's Umm Al-Qura logo
# is 1.5 % of the slide and entirely in the top-right corner; the smallest
# region on the same deck that carries real content is 6 %.
GATE_LOGO_PAGE_FRACTION = 0.02
# Corner bands, as a share of the page. Same values as image_prep's own
# CORNER_X / CORNER_Y, which describe the same idea for full-bleed scans.
GATE_LOGO_CORNER_X = 0.20
GATE_LOGO_CORNER_Y = 0.22

# When this share of an image's readings are themselves junk, the pixels
# under them are not text and the survivors are misreadings of the same
# artwork. Two rejects is the floor: one bad neighbour must not condemn a
# real label. (136 p1: `ll` and `ol` rejected out of four readings.)
GATE_JUNK_REGION_FRACTION = 0.5
GATE_JUNK_REGION_MIN = 2

# Variance of the discrete Laplacian over the region's own render. Job 154
# p1's blurred decorative photo of code scores 37; every region in the
# corpus that carries real text scores 232 or more (143 p6's two dialogs
# 232/245, 154's code screenshots 490-944, 136's logo 429). The threshold
# sits in the middle of that gap. Only meaningful at the region's NATIVE
# resolution — see {@link laplacian_variance}.
GATE_MIN_SHARPNESS = 100.0

# One line of a code screenshot, as OCR sees it: statement punctuation, a
# comment marker, a declaration keyword, a function CALL, a Windows path or
# an environment variable. The call and path forms were added after job 154
# pp.33/37 (`print(5*text)`, `float(input("enter`) and job 143 p6
# (`C:\Program`, `%SYSTEMROOT`) went through the old pattern untouched.
GATE_CODE_LINE = re.compile(
    r"[;{}]"
    r"|\(\s*\)"
    r"|//|/\*|\*/"
    r"|\\"                                    # a path separator, or an escape
    r"|%[A-Za-z]"                             # %SYSTEMROOT%, %PATH%
    r"|\b[A-Za-z_]\w*\s*\("                   # print(, float(, input(
    r"|^\s*(public|private|protected|static|void|int|double|float|class"
    r"|return|new|import|package)\b"
    r"|\b\w+\s*:\s*(int|double|float|void|string|char|bool|boolean)\b",
    re.IGNORECASE,
)
# Fewer code-ish lines than this never condemns an image, nor does a minority
# of them: a diagram may legitimately label one box `getName()`.
GATE_CODE_MIN_LINES = 2
GATE_CODE_LINE_FRACTION = 0.4

# --- density ----------------------------------------------------------------

# Two readings belong to the same visual ROW when they overlap vertically by
# this share of the shorter one. Side-by-side fragments of one line of a
# screenshot must be judged as one row, not as a stack of rows.
GATE_ROW_OVERLAP = 0.5

# What a gloss costs in vertical space: the mask/plate grows the row by
# GATE_MASK_PAD_PT plus GATE_MASK_VPAD_FRACTION of its height on each side
# (paragraph_finder.IMAGE_TEXT_MASK_PAD / _MASK_VPAD_FRACTION — the mono
# lane's actual mask geometry). A row therefore needs a pitch of at least
# (1 + 2 * 0.35) of its own height before the next one, or the two glosses
# overlap by construction. Rows tighter than that are PACKED.
GATE_MASK_PAD_PT = 2.0
GATE_MASK_VPAD_FRACTION = 0.35
GATE_CROWD_PITCH_FACTOR = 1.0 + 2 * GATE_MASK_VPAD_FRACTION

# An image where most rows are packed cannot be glossed legibly at all, and a
# terminal or a code screenshot left untouched reads far better than one made
# illegible. Measured on the corpus: 0 % for the dialogs and captions that
# render correctly, 60 % for a rendered TABLE that comes out mostly right
# (only its stacked header cells collide, and the collision rule below thins
# those), then 70 % for the NetBeans project dialog and 80 % for 154 p37's
# code screenshot — both of which came out unreadable. The line sits in that
# gap: partial crowding is thinned, wholesale crowding is refused.
GATE_MAX_PACKED_ROW_FRACTION = 0.65
# Below this many rows "half the rows are packed" is one coin flip, so the
# rule stands down and the other rules decide (a two-line terminal pane is
# glossed).
GATE_DENSITY_MIN_ROWS = 4

# Backstop for the shape density misses: masks that would cover this much of
# the image have eaten the image, whatever the row arithmetic says. Measured
# on the PADDED boxes, because that is what a mask actually covers: 154 p33's
# two-line terminal pane comes to 52 % (and is still worth glossing), so the
# line sits above it and only an image that is wall-to-wall text trips it.
GATE_MAX_GLOSS_AREA_FRACTION = 0.6


# ---------------------------------------------------------------------------
# the items a lane hands over


@dataclass
class RasterText:
    """One reading of text inside an image, as either lane can describe it.

    `box` is (x0, y0, x1, y1) in any single consistent coordinate system whose
    unit is the POINT — page space (y down, image_prep) and display space
    (y down, interlinear) both qualify. Only distances are used, never the
    direction of y, so a y-up space works too as long as the region is given
    in the same one.
    """

    text: str | None
    box: tuple[float, float, float, float]
    conf: float | None = None
    payload: object = None          # whatever the caller needs handed back
    reason: str | None = field(default=None, compare=False)

    @property
    def rect(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.box
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def height(self) -> float:
        _, y0, _, y1 = self.rect
        return y1 - y0

    def padded(self) -> tuple[float, float, float, float]:
        """The rect a mask/plate for this reading actually covers."""
        x0, y0, x1, y1 = self.rect
        vpad = max(GATE_MASK_PAD_PT, (y1 - y0) * GATE_MASK_VPAD_FRACTION)
        return (x0 - GATE_MASK_PAD_PT, y0 - vpad,
                x1 + GATE_MASK_PAD_PT, y1 + vpad)


@dataclass
class GlossPlan:
    """What to do with one image region.

    `reason` is None when the image may be glossed, and otherwise names the
    rule that refused it — the whole image, every reading in it. `keep` is
    the readings worth glossing when it is None; `dropped` is everything the
    per-reading rules or the collision pass threw away, each carrying its own
    `reason`, for logging.
    """

    reason: str | None
    keep: list[RasterText]
    dropped: list[RasterText]

    def __bool__(self) -> bool:
        return self.reason is None and bool(self.keep)


# ---------------------------------------------------------------------------
# per-reading quality


def _wordish(token: str) -> bool:
    if len(token) < GATE_WORDISH_MIN_LEN:
        return False
    if not any(char in GATE_VOWELS for char in token):
        return False
    lowered = token.lower()
    return not any(lowered[i] == lowered[i + 1] == lowered[i + 2]
                   for i in range(len(lowered) - 2))


def text_reject_reason(text: str | None, conf: float | None = None) -> str | None:
    """Why this reading is not language, or None when it is.

    The rules, in the order a reader would apply them: is it text at all, is
    it already in the target script, and does it read as words.

    `text is None` means the caller does not know what was recognised — an
    old sidecar records no source for some blocks — and, like an unknown
    confidence, it makes these rules stand down rather than guess. An empty
    STRING is a different statement and is refused.
    """
    if text is None:
        return None

    stripped = text.strip()

    if not stripped:
        return "empty"

    if conf is not None and conf < GATE_MIN_CONF:
        return "conf"

    letters = _LETTER.findall(stripped)
    arabic = _ARABIC.findall(stripped)

    if _MIXED_SCRIPT_TOKEN.search(stripped):
        # Judged before the script share: a Latin letter welded to an Arabic
        # one is noise in BOTH directions, and calling it "already Arabic"
        # would be flattering it.
        return "mixed-script"

    if letters and len(arabic) / len(letters) >= GATE_TARGET_SCRIPT_SHARE:
        # Already Arabic — the other half of a bilingual wordmark, or a
        # caption the deck itself set in both languages.
        return "target-language"

    tokens = _LATIN_TOKEN.findall(stripped)

    if not tokens:
        # Digits and symbols only. A measurement, an axis tick or a version
        # number: nothing to translate, but nothing to be wrong about either,
        # so it is only refused when it is not even that.
        return None if any(char.isdigit() for char in stripped) else "junk"

    wordish = [token for token in tokens if _wordish(token)]

    if not wordish:
        return "gibberish"

    if len(tokens) >= 3 and len(wordish) / len(tokens) < GATE_MIN_WORDISH_FRACTION:
        return "gibberish"

    return None


# ---------------------------------------------------------------------------
# geometry: rows, packing, collisions


def _overlaps(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def rows(items: list[RasterText]) -> list[tuple[float, float]]:
    """The readings' distinct visual rows, as (top, bottom), top-first.

    Fragments of one line of a screenshot arrive as separate readings with
    slightly different tops; they are one row, and counting them as several
    would make every screenshot look infinitely dense.
    """
    bands: list[list[float]] = []

    for item in sorted(items, key=lambda i: i.rect[1]):
        _, top, _, bottom = item.rect
        for band in bands:
            overlap = min(bottom, band[1]) - max(top, band[0])
            shorter = min(bottom - top, band[1] - band[0])
            if shorter > 0 and overlap / shorter >= GATE_ROW_OVERLAP:
                band[0] = min(band[0], top)
                band[1] = max(band[1], bottom)
                break
        else:
            bands.append([top, bottom])

    bands.sort()
    return [(top, bottom) for top, bottom in bands]


def packed_row_fraction(items: list[RasterText]) -> tuple[float, int]:
    """(share of rows with no room for a gloss under them, row count)."""
    bands = rows(items)

    if len(bands) < 2:
        return 0.0, len(bands)

    packed = 0
    for (top, bottom), (next_top, _) in zip(bands, bands[1:], strict=False):
        height = max(bottom - top, 1e-6)
        if next_top - top < GATE_CROWD_PITCH_FACTOR * height:
            packed += 1

    return packed / (len(bands) - 1), len(bands)


def drop_collisions(items: list[RasterText]) -> tuple[list[RasterText],
                                                      list[RasterText]]:
    """Keep only readings whose mask can be drawn without hiding another one.

    A gloss that covers the neighbouring label has taken more than it gave —
    and two glosses on the same pixels are simply two illegible ones. Longest
    reading wins the contested space: it is the one that carries the most
    meaning, and it is usually the one OCR was most sure of.
    """
    kept: list[RasterText] = []
    dropped: list[RasterText] = []

    for item in sorted(items, key=lambda i: (-len((i.text or "").strip()), i.rect)):
        mask = item.padded()
        if any(_overlaps(mask, other.rect) or _overlaps(item.rect, other.padded())
               for other in kept):
            item.reason = "collision"
            dropped.append(item)
        else:
            kept.append(item)

    kept.sort(key=lambda i: (i.rect[1], i.rect[0]))
    return kept, dropped


# ---------------------------------------------------------------------------
# per-image rules


def region_is_code(texts: list[str | None]) -> bool:
    """Is this image a code screenshot / terminal / path list?

    Judged on the readings that SURVIVED the quality rules, so a photo
    caption beside junk does not tip the scale. Refusing the WHOLE image is
    deliberate: translating a screenshot's prose while masking half its
    statements produces exactly the shredded hybrid this rule exists to
    prevent, and policy keeps code verbatim anyway.
    """
    lines = [text for text in texts if text and text.strip()]

    if len(lines) < GATE_CODE_MIN_LINES:
        return False

    hits = sum(1 for text in lines if GATE_CODE_LINE.search(text))

    return (hits >= GATE_CODE_MIN_LINES
            and hits / len(lines) >= GATE_CODE_LINE_FRACTION)


def _is_page_corner(region, page) -> bool:
    """Does the region lie wholly inside one corner band of the page?"""
    px0, py0, px1, py1 = page
    width, height = px1 - px0, py1 - py0
    x0, y0, x1, y1 = region
    horizontal = (x1 <= px0 + GATE_LOGO_CORNER_X * width
                  or x0 >= px1 - GATE_LOGO_CORNER_X * width)
    vertical = (y1 <= py0 + GATE_LOGO_CORNER_Y * height
                or y0 >= py1 - GATE_LOGO_CORNER_Y * height)
    return horizontal and vertical


def region_reject_reason(items: list[RasterText],
                         region: tuple[float, float, float, float],
                         page: tuple[float, float, float, float] | None = None,
                         sharpness: float | None = None,
                         rejected: int = 0) -> str | None:
    """Why this whole image must be left alone, or None to gloss it.

    `items` are the readings that PASSED {@link text_reject_reason};
    `rejected` is how many did not. `sharpness` and `page` are optional: a
    lane that cannot supply them makes those rules inert, not wrong.
    """
    if not items and not rejected:
        # Nothing was read here at all: artwork, and no decision to make.
        return "empty"

    x0, y0, x1, y1 = (min(region[0], region[2]), min(region[1], region[3]),
                      max(region[0], region[2]), max(region[1], region[3]))
    area = max((x1 - x0) * (y1 - y0), 1e-6)

    if page is not None:
        page_area = abs((page[2] - page[0]) * (page[3] - page[1])) or 1.0
        if (area / page_area <= GATE_LOGO_PAGE_FRACTION
                and _is_page_corner((x0, y0, x1, y1), page)):
            return "logo"

    if sharpness is not None and sharpness < GATE_MIN_SHARPNESS:
        return "blurred"

    if (rejected >= GATE_JUNK_REGION_MIN
            and rejected >= GATE_JUNK_REGION_FRACTION * (rejected + len(items))):
        return "junk"

    if not items:
        # Every reading was refused on its own merits, just not enough of
        # them at once to say the image is junk.
        return "unreadable"

    if region_is_code([item.text for item in items]):
        return "code"

    packed, row_count = packed_row_fraction(items)

    if (row_count >= GATE_DENSITY_MIN_ROWS
            and packed >= GATE_MAX_PACKED_ROW_FRACTION):
        return "dense"

    covered = sum((mx1 - mx0) * (my1 - my0)
                  for mx0, my0, mx1, my1 in (item.padded() for item in items))

    if covered / area > GATE_MAX_GLOSS_AREA_FRACTION:
        return "crowded"

    return None


# ---------------------------------------------------------------------------
# THE decision


def gloss_plan(items: list[RasterText],
               region: tuple[float, float, float, float],
               page: tuple[float, float, float, float] | None = None,
               sharpness: float | None = None) -> GlossPlan:
    """The one decision both lanes make about one image region.

    Readings that are not language are dropped; then the image as a whole is
    judged on what is left; then the survivors are thinned until no gloss can
    cover another.
    """
    good: list[RasterText] = []
    dropped: list[RasterText] = []

    for item in items:
        reason = text_reject_reason(item.text, item.conf)
        if reason is None:
            good.append(item)
        else:
            item.reason = reason
            dropped.append(item)

    refusal = region_reject_reason(good, region, page, sharpness,
                                   rejected=len(dropped))

    if refusal is not None:
        for item in good:
            item.reason = refusal
        return GlossPlan(refusal, [], dropped + good)

    kept, collided = drop_collisions(good)

    return GlossPlan(None, kept, dropped + collided)


# ---------------------------------------------------------------------------
# sharpness (only the lane holding the pixels can measure it)


def laplacian_variance(pixmap) -> float:
    """Blur score of a pymupdf Pixmap: variance of its discrete Laplacian.

    Meaningful only at the image's own resolution — image_prep renders a
    region at its NATIVE dpi (clamped to 300..600), so an upscaled render's
    interpolation blur is not mistaken for the photographer's.
    """
    # Local: the gate's rules are pure arithmetic, and only this one
    # measurement needs the array library.
    import numpy

    height, width, channels = pixmap.height, pixmap.width, pixmap.n
    samples = numpy.frombuffer(pixmap.samples, numpy.uint8)

    if samples.size == height * pixmap.stride:      # padded rows
        grid = samples.reshape(height, pixmap.stride)[:, :width * channels]
    else:
        grid = samples.reshape(height, width * channels)

    grey = grid.reshape(height, width, channels)[:, :, :3].mean(axis=2)
    grey = grey.astype(numpy.float32)

    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return float("inf")

    laplacian = (4 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1]
                 - grey[1:-1, :-2] - grey[1:-1, 2:])

    return float(laplacian.var())
