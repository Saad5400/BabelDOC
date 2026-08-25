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
- Page-count mismatch: the shorter document is padded with blanks sized
  like the twin page, so every pair stays aligned.
"""

from io import BytesIO

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

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


def _pairs(original: PdfReader, translated: PdfReader):
    """Yield (original_page, translated_page) with None past the shorter end."""
    for i in range(max(len(original.pages), len(translated.pages))):
        yield (original.pages[i] if i < len(original.pages) else None,
               translated.pages[i] if i < len(translated.pages) else None)


def _alternating(original: PdfReader, translated: PdfReader) -> PdfWriter:
    writer = PdfWriter()
    for orig, trans in _pairs(original, translated):
        # A missing page becomes a blank sized like its twin (one of the two
        # always exists — both inputs are non-empty).
        twin_w, twin_h = _size(orig if orig is not None else trans)
        writer.add_page(orig if orig is not None else
                        PageObject.create_blank_page(width=twin_w, height=twin_h))
        writer.add_page(trans if trans is not None else
                        PageObject.create_blank_page(width=twin_w, height=twin_h))
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


def _side_by_side(original: PdfReader, translated: PdfReader) -> PdfWriter:
    writer = PdfWriter()
    for orig, trans in _pairs(original, translated):
        sizes = [_size(p) for p in (orig, trans) if p is not None]
        half_width = max(w for w, _ in sizes)
        height = max(h for _, h in sizes)
        page = writer.add_blank_page(width=2 * half_width, height=height)
        # Original on the left, translated on the right; a missing twin
        # simply leaves its half blank.
        if orig is not None:
            _merge_into_half(page, orig, 0.0, half_width, height)
        if trans is not None:
            _merge_into_half(page, trans, half_width, half_width, height)
    return writer


def compose_dual(original_bytes: bytes, translated_bytes: bytes,
                 fmt: str) -> bytes:
    """Build the requested dual variant; raises ComposeError on bad input."""
    if fmt not in COMPOSE_FORMATS:
        raise ComposeError(f"format must be one of {COMPOSE_FORMATS}")
    original = _read(original_bytes, "original")
    translated = _read(translated_bytes, "translated")
    build = _alternating if fmt == "alternating" else _side_by_side
    out = BytesIO()
    build(original, translated).write(out)
    return out.getvalue()
