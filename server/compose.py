"""Stateless composition of dual PDFs from an original + translated pair.

The job pipeline can produce alternating / side-by-side duals natively, but
only as part of a full (paid) translation run. The Laravel app stores both
the original PDF and the mono translation, so dual variants can be rebuilt
at download time with pure pypdf page shuffling — no LLM, no job state,
everything in memory.

Conventions:
- alternating: original page 1, translated page 1, original page 2, ...
- side_by_side: one double-width page per pair, translated half on the
  RIGHT (Arabic/RTL readers scan right-first).
- Original longer: the translated side is padded with blanks sized like the
  twin page, so every pair stays aligned.
- Translated longer: the extra TAIL pages are appended whole at the end
  rather than paired with blanks — since the mono result gained its
  «شرح المصطلحات» appendix (server/glossary_pages.py) that tail is glossary
  pages, and a glossary page facing a blank reads as a mistake.

GLOSSARY: a caller may also send the run's sidecar. When it carries the
"glossary" entries server/terms.py selected, the composed output gets the same
styled pages appended, rendered fresh at the composed page size — and the
translated input's own glossary tail (everything past the page count the
sidecar records) is IGNORED, so the appendix is never in the document twice.
Without a sidecar the tail-append rule above keeps a glossary-tailed mono
usable as-is.

PHRASE HIGHLIGHTS: when the sidecar's blocks carry "pairs" (the aligned phrase
segmentation phrase_pairs.py captured), both duals get matching colour chips —
each source phrase and its translation tinted alike. Compose itself only
shuffles pypdf pages, so the chips are a PyMuPDF PRE-PASS over the two INPUT
byte strings (server/phrase_highlights.py): s_rects drawn on the original,
t_rects on the translated, before either is opened for assembly. Everything
downstream is untouched — the pre-pass never changes a page count, so the
glossary-tail accounting above still holds, and side_by_side's placement
scaling carries the chips along with the page they sit on for free.
"""

import logging
from io import BytesIO

from pypdf import PageObject
from pypdf import PdfReader
from pypdf import PdfWriter
from pypdf import Transformation

from server import config
from server import phrase_highlights

logger = logging.getLogger("doctranslate.compose")

COMPOSE_FORMATS = ("alternating", "side_by_side")

# Compose runs synchronously in the request thread; page shuffling is cheap
# but not free, so bound each input (a big slide deck is ~200 pages).
MAX_PAGES_PER_INPUT = 2000


class ComposeError(ValueError):
    """Invalid compose input (maps to HTTP 422)."""


def _read(pdf_bytes: bytes, name: str) -> PdfReader:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_count = len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - pypdf raises many error types
        raise ComposeError(f"{name} is not a readable PDF: {exc}") from exc
    if page_count == 0:
        raise ComposeError(f"{name} has no pages")
    if page_count > MAX_PAGES_PER_INPUT:
        raise ComposeError(
            f"{name} has {page_count} pages (max {MAX_PAGES_PER_INPUT})")
    return reader


def _size(page: PageObject) -> tuple[float, float]:
    box = page.mediabox
    return float(box.width), float(box.height)


def _split(original: PdfReader,
           translated_pages: list[PageObject]) -> tuple[list, list]:
    """`translated_pages` as (body paired with the original, appended tail)."""
    count = len(original.pages)
    return list(translated_pages[:count]), list(translated_pages[count:])


def _alternating(original: PdfReader,
                 translated_pages: list[PageObject]) -> PdfWriter:
    writer = PdfWriter()
    body, tail = _split(original, translated_pages)
    for index, orig in enumerate(original.pages):
        trans = body[index] if index < len(body) else None
        writer.add_page(orig)
        # A missing translated page becomes a blank sized like its twin.
        if trans is not None:
            writer.add_page(trans)
        else:
            twin_w, twin_h = _size(orig)
            writer.add_page(PageObject.create_blank_page(width=twin_w,
                                                         height=twin_h))
    for page in tail:
        writer.add_page(page)
    return writer


def _merge_into_half(target: PageObject, source: PageObject,
                     x_offset: float, half_width: float, height: float) -> None:
    """Draw source onto target, scaled to fit [x_offset, x_offset+half_width),
    preserving aspect ratio, centered within the half."""
    box = source.mediabox
    w, h = float(box.width), float(box.height)
    scale = min(half_width / w, height / h)
    tx = x_offset - float(box.left) * scale + (half_width - w * scale) / 2
    ty = -float(box.bottom) * scale + (height - h * scale) / 2
    target.merge_transformed_page(
        source, Transformation().scale(scale).translate(tx, ty))


def _side_by_side(original: PdfReader,
                  translated_pages: list[PageObject]) -> PdfWriter:
    writer = PdfWriter()
    body, tail = _split(original, translated_pages)
    for index, orig in enumerate(original.pages):
        trans = body[index] if index < len(body) else None
        sizes = [_size(p) for p in (orig, trans) if p is not None]
        half_width = max(w for w, _ in sizes)
        height = max(h for _, h in sizes)
        page = writer.add_blank_page(width=2 * half_width, height=height)
        # Original on the left, translated on the right; a missing twin
        # simply leaves its half blank.
        _merge_into_half(page, orig, 0.0, half_width, height)
        if trans is not None:
            _merge_into_half(page, trans, half_width, half_width, height)
    # Tail pages carry no original twin; appended whole, at their own size.
    for page in tail:
        writer.add_page(page)
    return writer


def _glossary_entries(sidecar) -> list | None:
    """The sidecar's usable glossary, or None when there is nothing to render."""
    if not config.GLOSSARY_PAGES or not isinstance(sidecar, dict):
        return None

    entries = sidecar.get("glossary")

    if not isinstance(entries, list) or not entries:
        return None

    return entries


def _content_page_count(sidecar: dict) -> int | None:
    """How many pages the TRANSLATION itself has, per the sidecar's records.

    Everything past this count in the translated input is the mono result's
    own appended glossary — rendered fresh here instead, never copied.
    """
    count = sidecar.get("total_pages")

    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return count

    pages = sidecar.get("pages")

    return len(pages) if isinstance(pages, list) and pages else None


def _append_glossary(composed: bytes, entries: list) -> bytes:
    """`composed` with the glossary pages appended — or unchanged on failure.

    Best-effort on purpose: the dual the caller paid nothing for must not 422
    over its appendix.
    """
    import pymupdf

    from server import glossary_pages

    doc = pymupdf.open(stream=BytesIO(composed), filetype="pdf")

    try:
        if not glossary_pages.append_glossary_pages(doc, entries):
            return composed

        out = BytesIO()
        # garbage=4 folds the per-box copies of the subset card font into one.
        doc.save(out, garbage=4, deflate=True)

        return out.getvalue()
    finally:
        doc.close()


def compose_dual(original_bytes: bytes, translated_bytes: bytes,
                 fmt: str, sidecar: dict | None = None) -> bytes:
    """Build the requested dual variant; raises ComposeError on bad input.

    `sidecar` is optional and changes nothing when absent (or when it carries
    no "glossary", or when GLOSSARY_PAGES is off) — see the module docstring
    for what a glossary-bearing sidecar adds.
    """
    if fmt not in COMPOSE_FORMATS:
        raise ComposeError(f"format must be one of {COMPOSE_FORMATS}")

    if config.PHRASE_HIGHLIGHTS and isinstance(sidecar, dict):
        # The chips pre-pass (module docstring): draw on the input bytes,
        # assemble exactly as before. Best-effort inside — on any failure the
        # bytes come back untouched, never an exception.
        original_bytes = phrase_highlights.highlight_pairs(
            original_bytes, sidecar, "s_rects")
        translated_bytes = phrase_highlights.highlight_pairs(
            translated_bytes, sidecar, "t_rects")

    original = _read(original_bytes, "original")
    translated = _read(translated_bytes, "translated")

    translated_pages = list(translated.pages)
    entries = _glossary_entries(sidecar)

    if entries is not None:
        content = _content_page_count(sidecar)
        if content is not None and content < len(translated_pages):
            # Drop the mono result's baked-in glossary tail; the entries are
            # rendered fresh below, at the composed page size.
            translated_pages = translated_pages[:content]

    build = _alternating if fmt == "alternating" else _side_by_side
    out = BytesIO()
    build(original, translated_pages).write(out)
    composed = out.getvalue()

    if entries is not None:
        try:
            composed = _append_glossary(composed, entries)
        except Exception:  # noqa: BLE001 - the appendix is optional
            logger.exception("compose: appending glossary pages failed; "
                             "returning the dual without them")

    return composed
