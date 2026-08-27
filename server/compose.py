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
  rather than paired with blanks — a mono result may carry appended appendix
  pages of its own, and an appendix page facing a blank reads as a mistake.

SIDECAR: a caller may also send the run's sidecar. When it carries the
"vocab" entries server/vocab.py selected, each content page's «كلمات هذه
الصفحة» strip is rendered fresh at the bottom of that page's unit in the
dual — and the translated input's own baked-in vocab is taken back out
first: exactly, via the sidecar's "artifact_layout" (drop any inserted
fallback pages by "content_pages", crop the baked bottom strips back off by
"vocab_strips" — a strip lives entirely at negative PDF y, so restoring the
recorded height onto mediabox.y0 recovers the pristine page bit-for-bit),
else by trimming everything past the page count the sidecar records. Without
a sidecar the tail-append rule above keeps an appendix-tailed mono usable
as-is.
"""

import logging
from io import BytesIO

from pypdf import PageObject
from pypdf import PdfReader
from pypdf import PdfWriter
from pypdf import Transformation

from server import config

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


def _artifact_content_positions(sidecar, page_count: int) -> list[int] | None:
    """The baked mono's content-page positions, when the sidecar records them.

    A mono whose vocab pages were interleaved (server/pipeline.py) records
    `"artifact_layout": {"content_pages": [...]}` — index i is where content
    page i sits in the baked file. That generalizes the tail-trim below: the
    content pages are picked out EXACTLY, and everything else (interleaved
    vocab pages, any baked appendix tail) is dropped — the vocab is rendered
    fresh at the composed layout instead.

    None means "no usable layout" and the caller falls back to the
    total_pages tail-trim — which keeps every pre-feature sidecar working
    unchanged. The sidecar is uploader-supplied here, so the list is
    distrusted: anything but strictly increasing ints is refused whole, and a
    position past the file is skipped (the layout describes a longer file
    than the one sent; the pages that do exist still pair correctly).
    """
    if not isinstance(sidecar, dict):
        return None

    layout = sidecar.get("artifact_layout")

    if not isinstance(layout, dict):
        return None

    positions = layout.get("content_pages")

    if not isinstance(positions, list) or not positions:
        return None

    for index, position in enumerate(positions):
        if not isinstance(position, int) or isinstance(position, bool):
            return None
        if position < 0 or (index and position <= positions[index - 1]):
            return None

    kept = [position for position in positions if position < page_count]

    return kept or None


def _artifact_strip_heights(sidecar) -> dict[int, float]:
    """The baked mono's per-content-page vocab strip heights, distrusted.

    `"artifact_layout": {"vocab_strips": {"i": h}}` records that content page
    i (an index into content_pages order) carries an h-points bottom strip
    (server/vocab_pages.attach_vocab). Junk keys and junk heights are dropped
    item by item — a bad entry loses only its own page's restore.
    """
    if not isinstance(sidecar, dict):
        return {}

    layout = sidecar.get("artifact_layout")
    strips = layout.get("vocab_strips") if isinstance(layout, dict) else None

    if not isinstance(strips, dict):
        return {}

    out: dict[int, float] = {}

    for key, value in strips.items():
        try:
            index, height = int(key), float(value)
        except (TypeError, ValueError):
            continue

        if index >= 0 and 0 < height <= 14400:  # NaN fails both comparisons
            out[index] = height

    return out


def _crop_baked_strips(translated_bytes: bytes, positions: list[int],
                       strips: dict[int, float]) -> bytes:
    """The translated input with its baked bottom strips PHYSICALLY removed.

    Shrinking the mediabox alone would only hide the band: its bytes would
    still ride along and re-surface the moment the fresh strip re-extends the
    page (or a side-by-side merge re-centers the content). So the band is
    redacted out — text, drawings, the divider, everything strictly below the
    content's old bottom edge — and only then is the box restored. A height
    that would consume the whole page is refused (a lying sidecar must not
    blank the content); a page the layout doesn't cover is left alone.
    """
    import pymupdf

    doc = pymupdf.open(stream=BytesIO(translated_bytes), filetype="pdf")

    try:
        changed = False

        for content_index in sorted(strips):
            height = strips[content_index]

            if content_index >= len(positions):
                continue

            page = doc[positions[content_index]]
            rect = page.rect

            if height >= rect.height or page.rotation:
                continue

            # 0.25pt below the old bottom edge: the divider (stroked ON the
            # edge) intersects and goes; content that merely ENDS at the
            # edge does not intersect and stays.
            page.add_redact_annot(
                pymupdf.Rect(-2, rect.height - height + 0.25,
                             rect.width + 2, rect.height + 2))
            page.apply_redactions()
            media = page.mediabox
            page.set_mediabox(pymupdf.Rect(media.x0, media.y0 + height,
                                           media.x1, media.y1))
            changed = True

        if not changed:
            return translated_bytes

        out = BytesIO()
        doc.save(out, garbage=3, deflate=True)

        return out.getvalue()
    finally:
        doc.close()


def _content_page_count(sidecar: dict) -> int | None:
    """How many pages the TRANSLATION itself has, per the sidecar's records.

    Everything past this count in the translated input is a baked-in appendix
    tail the mono result carries for its own readers — dropped here rather
    than paired, so the dual never duplicates it.
    """
    count = sidecar.get("total_pages")

    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return count

    pages = sidecar.get("pages")

    return len(pages) if isinstance(pages, list) and pages else None


def _insert_vocab_pages(composed: bytes, sidecar: dict, fmt: str,
                        original_count: int, content_count: int) -> bytes:
    """`composed` with each page's vocab strip on its unit — or unchanged.

    Content page N's strip sits at the bottom of page N's unit in the dual:
    the translated page of the (original, translated) pair for `alternating`,
    the wide page for `side_by_side` (spanning both halves). A content page
    past the original (the translated-longer tail) sits whole at the end and
    carries its strip there. A unit the strip cannot serve falls back to the
    classic inserted vocab page right after it. Best-effort on purpose: the
    dual the caller paid nothing for must not 422 over its vocab layer.
    """
    vocab = sidecar.get("vocab")

    if not isinstance(vocab, dict) or not vocab:
        return composed

    import pymupdf

    from server import vocab_pages

    anchors: dict[int, int] = {}

    for key in vocab:
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue

        if number < 0 or (number >= original_count
                          and number >= content_count):
            continue  # no such content page in this dual

        if fmt == "alternating":
            anchors[number] = (2 * number + 1 if number < original_count
                               else original_count + number)
        else:
            anchors[number] = number

    if not anchors:
        return composed

    doc = pymupdf.open(stream=BytesIO(composed), filetype="pdf")

    try:
        if not vocab_pages.attach_vocab(doc, vocab, anchors):
            return composed

        out = BytesIO()
        # garbage=4 folds the per-box copies of the subset row font into one.
        doc.save(out, garbage=4, deflate=True)

        return out.getvalue()
    finally:
        doc.close()


def compose_dual(original_bytes: bytes, translated_bytes: bytes,
                 fmt: str, sidecar: dict | None = None) -> bytes:
    """Build the requested dual variant; raises ComposeError on bad input.

    `sidecar` is optional and changes nothing when absent — see the module
    docstring for what a sidecar adds.
    """
    if fmt not in COMPOSE_FORMATS:
        raise ComposeError(f"format must be one of {COMPOSE_FORMATS}")
    original = _read(original_bytes, "original")
    translated = _read(translated_bytes, "translated")

    positions = _artifact_content_positions(sidecar, len(translated.pages))
    strips = _artifact_strip_heights(sidecar) if positions is not None else {}
    vocab_ok = True

    if strips:
        try:
            translated = _read(
                _crop_baked_strips(translated_bytes, positions, strips),
                "translated")
        except Exception:  # noqa: BLE001 - best-effort, never 422 over vocab
            # The baked strips stay on the pages then — still correct
            # content. The fresh vocab layer is skipped so nothing doubles.
            logger.exception("compose: cropping baked strips failed; "
                             "keeping the baked layout")
            vocab_ok = False

    translated_pages = list(translated.pages)

    if positions is not None:
        # The baked mono's own vocab is undone here — inserted fallback pages
        # (and any appendix tail) dropped, baked bottom strips already
        # cropped back off above; the vocab is rendered fresh below, at the
        # composed layout.
        translated_pages = [translated_pages[i] for i in positions]
    elif isinstance(sidecar, dict):
        content = _content_page_count(sidecar)
        if content is not None and content < len(translated_pages):
            # Drop the mono result's baked-in appendix tail; whatever the
            # sidecar carries for those pages is rendered fresh below.
            translated_pages = translated_pages[:content]

    build = _alternating if fmt == "alternating" else _side_by_side
    out = BytesIO()
    build(original, translated_pages).write(out)
    composed = out.getvalue()

    # The vocab strips live in the body, on each pair's unit. Best-effort.
    if config.VOCAB_PAGES and vocab_ok and isinstance(sidecar, dict):
        try:
            composed = _insert_vocab_pages(composed, sidecar, fmt,
                                           len(original.pages),
                                           len(translated_pages))
        except Exception:  # noqa: BLE001 - the vocab layer is optional
            logger.exception("compose: inserting vocab pages failed; "
                             "returning the dual without them")

    return composed
