"""Matching phrase highlights: the sidecar's aligned pairs, drawn as chips.

Phrase-pair capture (babeldoc/.../midend/phrase_pairs.py) leaves each sidecar
block with `{s, t, s_rects, t_rects}` entries, listed in SOURCE order and
aligned by MEANING — the translation may put the same phrases in a different
order, and each entry's rects are its own wherever they landed. This module is
the drawing side of that data: pair i of a paragraph gets colour i on BOTH sides,
as a soft background chip — filled at FILL_OPACITY ON TOP of the existing page
content, the way a text highlighter marks a printed page, which is also what
makes it work on scanned/opaque-background pages. No border, small rounded
corners, a little padding around the text rect. The palette cycles per
paragraph, so matching colours always mean matching phrases within one
paragraph and never claim anything across paragraphs.

Two call sites share these helpers:

- `/v1/compose` (server/compose.py): a PyMuPDF pre-pass over both INPUT PDFs
  before pypdf assembles the dual — `s_rects` onto the original's pages,
  `t_rects` onto the translated's ({@link highlight_pairs}). Drawing before
  assembly is what makes side_by_side free: the placement scaling that fits
  each input page into its half scales the chips along with everything else.
- `/v1/overlay` (server/interlinear.py): `s_rects` chips on the original page,
  via the same {@link draw_phrase_rects}. The Arabic side there is the gloss
  the Story engine lays out fresh, so `t_rects` (mono-layout space) do not
  apply — {@link GlossHighlighter} instead wraps the gloss text's phrase
  segments in coloured `<span>`s inside the HTML the overlay already builds.

Sidecars arrive UPLOADER-SUPPLIED at both endpoints, so the work is bounded
(`MAX_PAIRS` per paragraph — phrase_pairs' own cap — and
{@link MAX_RECTS_PER_PAGE} per page) and every coordinate is sanitized:
zero/negative area, wrong types, non-finite numbers are skipped silently, and
every drawn rect is clamped to its page. And everything is best-effort:
highlights are a bonus on top of a translation the caller already has, so no
failure here may cost them their PDF.

Kill switch: env `PHRASE_HIGHLIGHTS=0` (server/config.py) turns the drawing
off at both call sites. It is independent of `PHRASE_PAIRS`, which gates
capturing pairs into the sidecar in the first place.
"""

from __future__ import annotations

import html
import logging
import math
from io import BytesIO

import pymupdf
from babeldoc.format.pdf.document_il.midend.phrase_pairs import MAX_PAIRS
from babeldoc.format.pdf.document_il.midend.phrase_pairs import tile_permutation

logger = logging.getLogger("doctranslate.phrase_highlights")

# The owner-approved chip palette, cycled per paragraph: phrase i (on both
# sides) gets PALETTE[i % len]. Eight hues since the word-level granularity
# ruling (2026-08-27), ordered so neighbours in the cycle sit far apart on
# the wheel: amber, sky, green, pink, violet, teal, peach, lime. Subtlety is
# ONE knob — FILL_OPACITY — by owner ruling; do not also desaturate these.
PALETTE = ("#FDE68A", "#BAE6FD", "#BBF7D0", "#FBCFE8",
           "#DDD6FE", "#5EEAD4", "#FDBA74", "#D9F99D")

# Chips draw like a text highlighter: soft, borderless, on top.
FILL_OPACITY = 0.30
PADDING = 1.5  # points of breathing room around the text rect
CORNER_RADIUS = 2.0  # points

# An uploader-supplied sidecar bounds the drawing it can ask for.
MAX_RECTS_PER_PAGE = 400


def chip_color(index: int) -> str:
    """Phrase `index`'s colour, as hex — the palette, cycled."""
    return PALETTE[index % len(PALETTE)]


def span_color(index: int) -> str:
    """`chip_color` pre-blended toward white by FILL_OPACITY.

    The Story engine paints span backgrounds solid, while drawn chips get
    `fill_opacity` over white paper — blending here keeps the overlay's
    Arabic side the same visual weight as the drawn chips beside it.
    """
    code = chip_color(index).lstrip("#")
    blended = (round(255 - (255 - int(code[i:i + 2], 16)) * FILL_OPACITY)
               for i in (0, 2, 4))
    return "#" + "".join(f"{c:02x}" for c in blended)


def _chip_fill(index: int) -> tuple[float, float, float]:
    """The same colour as the 0..1 RGB triple PyMuPDF's fill wants."""
    code = chip_color(index)

    return tuple(int(code[i:i + 2], 16) / 255 for i in (1, 3, 5))


def sane_rect(raw) -> list[float] | None:
    """`raw` as `[x0, y0, x1, y1]` floats with positive area, or None.

    bools are numbers to isinstance and garbage to geometry; NaN and inf are
    numbers to both and poison to clamping — all of them answer None, silently,
    which is the contract for uploader-supplied coordinates.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None

    values: list[float] = []

    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        values.append(value)

    x0, y0, x1, y1 = values

    if x1 <= x0 or y1 <= y0:
        return None

    return values


def pair_chip_rects(pairs, key: str) -> list[tuple[list[list[float]], int]]:
    """One block's drawable chips: `(a phrase's sane rects, its colour index)`.

    `key` picks the side ("s_rects" / "t_rects"). Bad shapes are skipped, never
    raised on; a phrase missing the requested side simply contributes nothing —
    the other side still highlights. The list is capped at phrase_pairs' own
    MAX_PAIRS, past which a "pairs" value is a runaway, not a segmentation.
    """
    if not isinstance(pairs, list):
        return []

    chips: list[tuple[list[list[float]], int]] = []

    for index, pair in enumerate(pairs[:MAX_PAIRS]):
        if not isinstance(pair, dict):
            continue

        raw = pair.get(key)

        if not isinstance(raw, list):
            continue

        rects = [rect for rect in map(sane_rect, raw) if rect is not None]

        if rects:
            chips.append((rects, index))

    return chips


def draw_phrase_rects(page: pymupdf.Page, rects, color_index: int) -> int:
    """Draw one phrase's chips on `page`. `rects` are DISPLAY-space (y down).

    Each rect is padded by {@link PADDING}, clamped to the page, and skipped
    when nothing usable remains. Returns how many were drawn; a single bad
    rect never costs the rest.
    """
    drawn = 0
    fill = _chip_fill(color_index)

    for rect in rects:
        try:
            padded = pymupdf.Rect(rect) + (-PADDING, -PADDING, PADDING, PADDING)
            chip = padded & page.rect

            if not chip.is_valid or chip.is_empty or chip.is_infinite:
                continue

            # draw_rect's radius is a fraction of the shorter side; ask for
            # CORNER_RADIUS points, capped at the fully-round half.
            radius = min(0.5, CORNER_RADIUS / min(chip.width, chip.height))
            page.draw_rect(chip, color=None, fill=fill,
                           fill_opacity=FILL_OPACITY, radius=radius)
            drawn += 1
        except Exception:  # noqa: BLE001 - chips are a bonus, never a risk
            logger.exception("a phrase highlight rect failed; skipping it")

    return drawn


def _to_display(box: list[float], matrix: pymupdf.Matrix) -> pymupdf.Rect:
    """One sidecar box (PDF user space, y up) as a page rect (y down).

    The same mapping server/interlinear.py uses for its glosses; kept local
    because interlinear imports this module, not the other way round.
    """
    x0, y0, x1, y1 = box

    return pymupdf.Rect(pymupdf.Point(x0, y1) * matrix,
                        pymupdf.Point(x1, y0) * matrix).normalize()


def _draw_page_chips(page: pymupdf.Page, chips) -> int:
    """All of one page's chips, mapped from sidecar space and drawn.

    Sidecar coordinates are unrotated PDF user space; neutralising /Rotate for
    the duration makes `page.transformation_matrix` the plain y-flip that maps
    them, and the chips rotate back with the page's own content afterwards
    (the same dance server/interlinear.py does).
    """
    rotation = page.rotation

    if rotation:
        page.set_rotation(0)

    try:
        matrix = page.transformation_matrix
        drawn = 0

        for rects, color_index in chips:
            drawn += draw_phrase_rects(
                page, [_to_display(rect, matrix) for rect in rects], color_index)

        return drawn
    finally:
        if rotation:
            page.set_rotation(rotation)


def _chips_by_page(sidecar: dict, key: str) -> dict[int, list]:
    """Every page's drawable chips for one side, bounded per page."""
    pages = sidecar.get("pages")

    if not isinstance(pages, list):
        return {}

    by_page: dict[int, list] = {}

    for page_data in pages:
        if not isinstance(page_data, dict):
            continue

        index = page_data.get("page_number")

        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            continue

        chips = by_page.setdefault(index, [])
        budget = MAX_RECTS_PER_PAGE - sum(len(rects) for rects, _ in chips)
        blocks = page_data.get("blocks")

        for block in blocks if isinstance(blocks, list) else ():
            if budget <= 0:
                break

            if not isinstance(block, dict):
                continue

            for rects, color_index in pair_chip_rects(block.get("pairs"), key):
                rects = rects[:budget]

                if not rects:
                    break

                budget -= len(rects)
                chips.append((rects, color_index))

    return {index: chips for index, chips in by_page.items() if chips}


def highlight_pairs(pdf_bytes: bytes, sidecar, key: str) -> bytes:
    """`pdf_bytes` with the sidecar's `key`-side phrase chips drawn on.

    The /v1/compose pre-pass. Returns the INPUT BYTES UNTOUCHED whenever there
    is nothing — or no way — to draw: no pairs on the sidecar, an unreadable or
    encrypted PDF, any failure at all. Drawing never adds, removes or resizes a
    page, so the page count compose's glossary-tail accounting
    (`_content_page_count`) relies on survives the re-save exactly.
    """
    try:
        if not isinstance(sidecar, dict):
            return pdf_bytes

        by_page = _chips_by_page(sidecar, key)

        if not by_page:
            return pdf_bytes

        doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")

        try:
            if doc.needs_pass:
                return pdf_bytes

            # A sidecar page the PDF does not have is simply not drawn on:
            # at this endpoint the sidecar is the uploader's claim, and the
            # chips are not worth refusing a compose over.
            drawn = sum(_draw_page_chips(doc[index], chips)
                        for index, chips in by_page.items()
                        if 0 <= index < doc.page_count)

            if not drawn:
                return pdf_bytes

            out = BytesIO()
            doc.save(out)

            return out.getvalue()
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 - highlights are a bonus, never a risk
        logger.exception("phrase highlights (%s) failed; using the input "
                         "untouched", key)

        return pdf_bytes


class GlossHighlighter:
    """Colours one paragraph's gloss text by its phrase segmentation.

    The pairs are listed in SOURCE order and their "t" strings are an EXACT
    segmentation of the block's translation in whatever order the translation
    actually uses (whitespace-normalized) — Arabic may reorder the source's
    phrases. `phrase_pairs.tile_permutation` (the same tiler the capture
    pipeline uses) says which pair each successive stretch of the gloss
    belongs to, and each stretch is coloured with its SOURCE pair's index —
    matching colours stay bound to matching MEANING, wherever the phrase
    landed. The matching is word-wise, which is what makes it
    whitespace-flexible. The overlay may draw a gloss as ONE box or SPREAD it
    down the source lines in word chunks; {@link html} is therefore a cursor:
    called once with the full text, or repeatedly with consecutive chunks in
    drawing order, it hands back each piece's ESCAPED inner HTML with phrase
    `<span>`s — or None on any mismatch, which the caller renders as the plain
    unhighlighted gloss. Phrases are matched against the RAW text and escaping
    happens here, around the matched segments, so a `<` or `&` inside a phrase
    stays text and never becomes markup.
    """

    def __init__(self, target_text, pairs) -> None:
        self._words = self._word_colors(target_text, pairs)
        self._cursor = 0

    @staticmethod
    def _word_colors(target_text, pairs) -> list[tuple[str, int]] | None:
        """Each word of the target, in TARGET order, tagged with the colour
        index of the SOURCE pair whose phrase it belongs to."""
        if not isinstance(target_text, str) or not isinstance(pairs, list):
            return None

        if not 1 <= len(pairs) <= MAX_PAIRS:
            return None

        for pair in pairs:
            if not isinstance(pair, dict) or not isinstance(pair.get("t"), str):
                return None

        phrase_tokens = [pair["t"].split() for pair in pairs]
        permutation = tile_permutation(phrase_tokens, target_text.split())

        if permutation is None:
            return None

        return [(word, index)
                for index in permutation
                for word in phrase_tokens[index]]

    @property
    def usable(self) -> bool:
        return self._words is not None

    def html(self, text: str) -> str | None:
        """`text`'s escaped inner HTML with phrase spans, or None on mismatch.

        Consumes `text`'s words from the cursor — the whole gloss in one call,
        or its spread chunks across consecutive calls. A None answer never
        advances the cursor, so one odd chunk cannot desynchronise the rest.
        """
        if self._words is None:
            return None

        tokens = text.split()
        end = self._cursor + len(tokens)
        window = self._words[self._cursor:end]

        if not tokens or [word for word, _ in window] != tokens:
            return None

        self._cursor = end

        segments: list[tuple[int, list[str]]] = []

        for word, color_index in window:
            if segments and segments[-1][0] == color_index:
                segments[-1][1].append(word)
            else:
                segments.append((color_index, [word]))

        return " ".join(
            f'<span style="background-color:{span_color(color_index)}">'
            f'{html.escape(" ".join(words))}</span>'
            for color_index, words in segments)


def gloss_highlighter(target_text, pairs) -> GlossHighlighter | None:
    """A usable highlighter for this block, or None (absent/invalid pairs)."""
    highlighter = GlossHighlighter(target_text, pairs)

    return highlighter if highlighter.usable else None
