"""The translation SIDECAR: one run's translated text, kept as data.

A translation costs money; a layout does not. Everything the pipeline knows
about a finished translation — which paragraph on which page said what, in the
source and in the target, and how big it was drawn — lives in the IL for the
few seconds between {@link ILTranslator} and {@link Typesetting}, and then is
thrown away with the process. The only durable artifact is a rendered PDF, so
every new way of *presenting* the same translation used to mean paying for the
same translation again.

The sidecar is that knowledge written down: a small JSON document, emitted by
the same run that produced the PDF, from which any number of layouts can be
rebuilt later for free — the interlinear overlay in `server/interlinear.py`
today, a bilingual export or a study sheet tomorrow.

WHEN it is captured is the whole trick. `PdfParagraph.unicode` holds the SOURCE
text until ILTranslator overwrites it with the translation, and the paragraph's
compositions still hold the SOURCE characters until it replaces those too;
`PdfParagraph.box` holds the paragraph's box as it sits in the ORIGINAL page
until Typesetting starts moving and growing boxes to fit the target text. So the
source side is snapshotted before translation ({@link snapshot_source}) and
everything else is read after translation but BEFORE typesetting ({@link
build_sidecar}) — exactly the window in which both halves are true at once.

SOURCE LINES are part of that snapshot and are not a nicety. BabelDOC merges a
run of same-styled bullets into ONE paragraph, so a slide's whole bullet list
can arrive as a single block whose only free space is the thin band above the
first bullet. Knowing where the source's own lines sit lets a renderer spread
that block's translation down the list instead of crushing it into one gap.

COORDINATES are PDF user space (y grows upward), unrotated, exactly as the IL
carries them. BabelDOC normalises every page's MediaBox to `[0 0 w h]` before
parsing (`high_level.fix_media_box`), which changes the visible window but never
moves the content, so a point here is the same point in the untouched original
the caller pairs the sidecar with. A renderer maps them with the target page's
own `transformation_matrix` (and must neutralise page rotation first — see
`server/interlinear.py`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string

logger = logging.getLogger(__name__)

# Bump when the shape changes incompatibly. Readers refuse what they do not
# know rather than guessing at a layout from a future engine.
SIDECAR_VERSION = 1

# Paragraph boxes below this many points in either dimension are layout noise
# (a stray glyph, a rule mistaken for text): nothing readable can be glossed
# above them and a renderer would only produce specks.
_MIN_BOX_SIDE = 2.0


def _box(box: il_version_1.Box | None) -> list[float] | None:
    """One IL box as `[x0, y0, x1, y1]`, or None when it is unusable."""
    if box is None:
        return None

    x0, y0, x1, y1 = box.x, box.y, box.x2, box.y2

    if None in (x0, y0, x1, y1):
        return None

    # The IL is not guaranteed to keep x<x2 / y<y2 after the layout passes.
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))

    if x1 - x0 < _MIN_BOX_SIDE or y1 - y0 < _MIN_BOX_SIDE:
        return None

    return [x0, y0, x1, y1]


def _characters(paragraph: il_version_1.PdfParagraph) -> list[il_version_1.PdfCharacter]:
    """Every source character of a paragraph, in reading order."""
    characters: list[il_version_1.PdfCharacter] = []

    for composition in paragraph.pdf_paragraph_composition:
        for holder in (composition.pdf_line, composition.pdf_formula,
                       composition.pdf_same_style_characters):
            if holder is not None:
                characters.extend(holder.pdf_character)
                break
        else:
            # A bare character composition; the unicode-run kind carries no
            # characters at all (it only appears after translation anyway).
            if composition.pdf_character is not None:
                characters.append(composition.pdf_character)

    return characters


def _source_lines(paragraph: il_version_1.PdfParagraph) -> list[list[float]]:
    """The paragraph's own lines, as boxes, in reading order.

    Rebuilt from the source characters rather than read off the compositions:
    by the time this runs StylesAndFormulas has regrouped those by STYLE, so a
    line can be several compositions and a composition can span several lines.

    The break rule is geometric — a character whose TOP is below the current
    line's bottom starts a new line — which keeps sub/superscripts and inline
    formulas on the line they belong to. BabelDOC's own `Layout.is_newline`
    would be the obvious thing to reuse, but it declines to judge any character
    without a `pdf_character_id`, and the pipeline synthesises exactly such
    characters (the dummy spaces); one of those falling on a line boundary
    would silently weld two lines together.
    """
    lines: list[list[float]] = []
    current: list[float] | None = None

    for character in _characters(paragraph):
        box = character.box

        if box is None or None in (box.x, box.y, box.x2, box.y2):
            continue

        if current is None or box.y2 < current[1]:
            current = [box.x, box.y, box.x2, box.y2]
            lines.append(current)
        else:
            current[0] = min(current[0], box.x)
            current[1] = min(current[1], box.y)
            current[2] = max(current[2], box.x2)
            current[3] = max(current[3], box.y2)

    return [line for line in lines
            if line[2] - line[0] >= _MIN_BOX_SIDE and line[3] - line[1] >= _MIN_BOX_SIDE]


# Where a run boundary must NOT become a space: punctuation that hugs the word
# before it, and brackets that hug the word after.
_HUGS_LEFT = ")]}»،؛؟,.;:!?%"
_HUGS_RIGHT = "([{«"


def _composition_text(composition: il_version_1.PdfParagraphComposition) -> str:
    """One composition's text: a translated run, or a formula's own characters."""
    run = composition.pdf_same_style_unicode_characters

    if run is not None:
        return run.unicode or ""

    for holder in (composition.pdf_line, composition.pdf_formula,
                   composition.pdf_same_style_characters):
        if holder is not None:
            return get_char_unicode_string(list(holder.pdf_character))

    if composition.pdf_character is not None:
        return get_char_unicode_string([composition.pdf_character])

    return ""


def _target_text(paragraph: il_version_1.PdfParagraph) -> str:
    """The paragraph's translation as a READER would see it.

    NOT `paragraph.unicode`, which is the translator's raw output and still
    carries the pipeline's own scaffolding — `{v1}` placeholders standing in
    for formulas and inline glyphs (a bullet, a symbol) and `<style id='N'>`
    tags marking the runs whose formatting must be preserved. Typesetting
    consumes all of that on its way to the page; a sidecar that captured it
    verbatim would hand the overlay renderer literal «{v1}مقدمة إلى NumPy» to
    draw.

    The compositions ILTranslator leaves behind are that same text already
    parsed — translated runs as plain unicode, each placeholder resolved back
    to the characters it stood for.

    Reading them back needs one repair the typeset page never does. A styled
    run carries no space of its own, because on the page the runs are POSITIONED
    rather than concatenated; flatten them into a string and
    «<style>المحاضرة 9:</style>مقدمة إلى<style>NumPy</style>» comes out as
    «المحاضرة 9:مقدمة إلىNumPy», welded at every boundary. So a boundary
    becomes a space unless one already sits there or the punctuation on either
    side is the hugging kind. A style change mid-word would gain a space it
    should not have; that is rarer than the welding, and far cheaper to read.

    An untranslated paragraph (nothing to parse) falls back to its own unicode,
    which never had scaffolding in it to begin with.
    """
    text = ""

    for composition in paragraph.pdf_paragraph_composition:
        part = _composition_text(composition)

        if part == "":
            continue

        if (text and not text[-1].isspace() and not part[0].isspace()
                and part[0] not in _HUGS_LEFT and text[-1] not in _HUGS_RIGHT):
            text += " "

        text += part

    return text.strip() or (paragraph.unicode or "").strip()


def snapshot_source(docs: il_version_1.Document) -> dict[int, dict[int, dict]]:
    """The SOURCE side of every paragraph, keyed page index → paragraph index.

    Called BEFORE ILTranslator, which overwrites `paragraph.unicode` AND its
    compositions in place with the translation. Positional keys (not
    `debug_id`, which is only populated in debug runs) are matched back in
    {@link build_sidecar}, and they hold because ILTranslator rewrites
    paragraphs IN PLACE: the only one it ever adds (the provider's
    content-filter notice) is appended past the end, so an index that existed
    before still names the same paragraph after. A paragraph with no entry —
    that notice — simply loses its source side, never its translation.
    """
    return {
        page_index: {
            para_index: {"text": paragraph.unicode,
                         "lines": _source_lines(paragraph)}
            for para_index, paragraph in enumerate(page.pdf_paragraph)
            if paragraph.unicode
        }
        for page_index, page in enumerate(docs.page)
    }


def build_sidecar(
    docs: il_version_1.Document,
    *,
    lang_in: str,
    lang_out: str,
    sources: dict[int, dict[int, dict]] | None = None,
) -> dict[str, Any]:
    """The sidecar document for a translated, NOT YET typeset, IL."""
    sources = sources or {}
    pages: list[dict[str, Any]] = []

    for page_index, page in enumerate(docs.page):
        mediabox = _box(page.mediabox.box) if page.mediabox is not None else None

        if mediabox is None:
            # Without the page's own frame a renderer cannot map anything on
            # it; emit the page as empty rather than dropping it, so page
            # indexes keep lining up with the PDF the caller pairs this with.
            logger.warning("sidecar: page %s has no usable mediabox", page_index)
            pages.append({"page_number": page_index, "mediabox": None,
                          "blocks": [], "obstacles": []})
            continue

        page_sources = sources.get(page_index, {})
        blocks: list[dict[str, Any]] = []

        for para_index, paragraph in enumerate(page.pdf_paragraph):
            box = _box(paragraph.box)
            target = _target_text(paragraph)

            if box is None or target == "":
                continue

            source = page_sources.get(para_index) or {}

            block = {
                "box": box,
                "source": (source.get("text") or "").strip() or None,
                "lines": source.get("lines") or [],
                "target": target,
                "font_size": (paragraph.pdf_style.font_size
                              if paragraph.pdf_style is not None else None),
                "label": paragraph.layout_label,
            }

            # Image-text lane (additive, version stays 1): a label that sits
            # ON an embedded raster image carries its image's placement box,
            # so the interlinear renderer knows the containing image's own
            # ink must not veto the gloss and can plate it for legibility.
            region = getattr(paragraph, "raster_region", None)
            if region and len(region) == 4 and None not in region:
                block["on_raster"] = True
                block["region"] = [float(value) for value in region]

            blocks.append(block)

        # Figures are not glossed, but they are the other thing that occupies
        # vertical space: a renderer looking for room above a paragraph has to
        # know an image is sitting there.
        obstacles = [b for b in (_box(figure.box) for figure in page.pdf_figure)
                     if b is not None]

        pages.append({
            "page_number": page_index,
            "mediabox": mediabox,
            "blocks": blocks,
            "obstacles": obstacles,
        })

    return {
        "version": SIDECAR_VERSION,
        "lang_in": lang_in,
        "lang_out": lang_out,
        "total_pages": docs.total_pages,
        "pages": pages,
    }


def write_sidecar(
    docs: il_version_1.Document,
    path: Path | str,
    *,
    lang_in: str,
    lang_out: str,
    sources: dict[int, dict[int, dict]] | None = None,
) -> None:
    """Write the sidecar beside a run's output.

    Best-effort by construction: the sidecar buys FUTURE layouts, and a run
    whose PDF is finished must not fail over one. The caller logs and carries
    on, which is why nothing here raises.
    """
    try:
        sidecar = build_sidecar(docs, lang_in=lang_in, lang_out=lang_out,
                                sources=sources)
        Path(path).write_text(json.dumps(sidecar, ensure_ascii=False),
                              encoding="utf-8")
    except Exception:  # noqa: BLE001 - a lost sidecar must never lose the run
        logger.exception("failed to write the translation sidecar to %s", path)
