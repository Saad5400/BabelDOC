"""Stateless composition of dual PDFs from an original + translated pair.

The job pipeline can produce alternating / side-by-side duals natively, but
only as part of a full (paid) translation run. The Laravel app stores both
the original PDF and the mono translation, so dual variants can be rebuilt
at download time with pure pypdf page shuffling — no LLM, no job state,
everything in memory.

Conventions:
- alternating: original page 1, translated page 1, original page 2, ...
- side_by_side: one double-width page per pair, translated half on the
  RIGHT (Arabic/RTL readers scan right-first), with a hairline gutter down
  the seam so a full-bleed page cannot read as one double-width page.
- Original longer: the translated side is padded with blanks sized like the
  twin page, so every pair stays aligned.
- Translated longer: only reconcilable when the sidecar says which pages
  are content ("artifact_layout"). The extra pages are then a TAIL and are
  appended whole at the end rather than paired with blanks — a mono result
  may carry appended pages of its own, and one facing a blank reads as a
  mistake. Without that layout the pairing cannot be justified at all and
  the build is REFUSED; see {@link _reconcile}.

The dual also inherits what the mono says about itself — its title, /Lang
and /ViewerPreferences — and the original's outline, remapped onto the
composed pages. Nothing here renders «شرح المصطلحات» deep-terms pages:
compose has no terms code, and a sidecar's tail-trim below is what DROPS a
baked terms tail rather than what carries one over.

SIDECAR: a caller may also send the run's sidecar. When it carries the
"vocab" entries server/vocab.py selected, each content page's «كلمات هذه
الصفحة» strip is rendered fresh at the bottom of that page's unit in the
dual — and the translated input's own baked-in vocab is taken back out
first: exactly, via the sidecar's "artifact_layout" (drop any inserted
fallback pages by "content_pages", crop the baked bottom strips back off by
"vocab_strips" — a strip lives entirely at negative PDF y, so restoring the
recorded height onto mediabox.y0 recovers the pristine page bit-for-bit),
else by trimming everything past the page count the sidecar records.
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

# The side_by_side seam: a hairline in the vocab strip divider's own grey
# (#CBD5E1, one step darker than the strip's rule so it still reads over a
# full-bleed illustration), at the same weight — the two rules on a page
# should look like one hand.
_GUTTER_COLOR = (0xCB / 255, 0xD5 / 255, 0xE1 / 255)
_GUTTER_WIDTH = 0.75


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


def _reconcile(original: PdfReader, translated: PdfReader,
               positions: list[int] | None) -> None:
    """Refuse a pairing that cannot be justified, instead of guessing one.

    Runs on the translated INPUT, before any trimming — trimming is one of
    the guesses this is here to stop.

    {@link _split} pairs positionally: translated page i is original page
    i's twin, and anything past the original is a tail to append. That
    holds only when the translated input is content pages and nothing else.
    A mono whose «كلمات هذه الصفحة» pages were INSERTED between content
    pages — server/vocab_pages.attach_vocab falls back to draw_vocab_pages
    whenever a page is rotated, too narrow, or its rows too tall — is
    longer than its original for a completely different reason, and pairing
    it positionally drifts from the first inserted page onward and never
    recovers. Trimming to the sidecar's `total_pages` does not save it
    either: that drops pages off the END, and the pages in the way are in
    the MIDDLE. Measured on a 25-page production run rebuilt in that shape,
    composed with a sidecar that had no layout: 2 of 25 units paired, and
    11 units showed a vocab page where the reader's translation belongs.

    The sidecar's "artifact_layout" is the only thing that tells an
    inserted page from a tail page, so a longer translated input without
    one is refused (422). The caller degrades: the studio download reports
    the layout as unavailable and catodemy's ComposeBankArtifact::handle()
    banks the mono with a note. A layout the reader does not get is much
    cheaper than one whose pages lie about each other.
    """
    if positions is not None or len(translated.pages) <= len(original.pages):
        return

    raise ComposeError(
        f"translated has {len(translated.pages)} pages against "
        f"{len(original.pages)} original pages, and the sidecar does not "
        "record which of them are content pages (artifact_layout): the "
        "extra pages cannot be told from vocab pages inserted between "
        "content pages, so the pairing is refused rather than guessed")


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


def _upright(page: PageObject) -> PageObject:
    """`page` with its /Rotate baked into its content, so it MEASURES the way
    it displays.

    alternating keeps each page whole, so /Rotate rides along and a
    landscape page stays landscape. side_by_side draws pages into halves,
    and a merge carries the content but not the page attribute — so a page
    that displays landscape would be drawn upright inside a portrait half,
    i.e. sideways to the reader, and sized from the wrong rect. Baking the
    rotation into the content stream first makes both halves plain
    unrotated art whose mediabox is what the reader sees. A page with no
    /Rotate is untouched.
    """
    if not page.rotation % 360:
        return page

    # Baking rewrites the page's content stream, so it happens on a COPY
    # attached to a writer of its own: pypdf rewrites an unattached page
    # unreliably (and says so), and the reader's page belongs to the caller.
    attached = PdfWriter().add_page(page)
    attached.transfer_rotation_to_content()

    return attached


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
        orig = _upright(orig)
        trans = body[index] if index < len(body) else None
        trans = _upright(trans) if trans is not None else None
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


def _attach_vocab_layer(doc, sidecar: dict, fmt: str,
                        original_count: int, content_count: int) -> dict[int, int]:
    """Each page's vocab strip drawn onto its unit of the open `doc`.

    Content page N's strip sits at the bottom of page N's unit in the dual:
    the translated page of the (original, translated) pair for `alternating`,
    the wide page for `side_by_side` (spanning both halves). A content page
    past the original (the translated-longer tail) sits whole at the end and
    carries its strip there. A unit the strip cannot serve falls back to the
    classic inserted vocab page right after it — reported back as
    {composed page index: pages inserted after it} so a caller that indexes
    the finished document ({@link _carry_properties}' outline) can follow
    the shift.
    """
    vocab = sidecar.get("vocab")

    if not isinstance(vocab, dict) or not vocab:
        return {}

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
        return {}

    added = vocab_pages.attach_vocab(doc, vocab, anchors)

    return {anchors[number]: int(-value)
            for number, value in added.items() if value < 0}


def _draw_gutters(doc, unit_count: int) -> None:
    """A hairline down the seam of every paired side_by_side page.

    The two halves abut exactly, so a page whose art is full-bleed — a
    slide with a banner across the top, say — runs straight over the seam
    and the composed page reads as ONE double-width slide instead of a
    pair. (Rendered evidence: run38 p.29, run20 p.36.) Drawn here rather
    than in {@link _side_by_side} because it is one line per page on an
    open document, and drawn BEFORE the vocab layer so it spans the
    content only: the strip is the unit's, not either half's, and a rule
    through it would be wrong.
    """
    import pymupdf

    for index in range(min(unit_count, doc.page_count)):
        page = doc[index]
        rect = page.rect
        middle = rect.width / 2
        page.draw_line(pymupdf.Point(middle, 0),
                       pymupdf.Point(middle, rect.height),
                       color=_GUTTER_COLOR, width=_GUTTER_WIDTH)


def _resolved(value):
    """An indirect reference followed once; anything else as it is."""
    return value.get_object() if hasattr(value, "get_object") else value


def _document_properties(translated: PdfReader,
                         original: PdfReader) -> dict[str, str]:
    """What the dual should inherit from the documents it is built from.

    A dual is a rebuild of the mono, so the mono's own declarations belong
    on it: the title (a run whose mono was re-titled — catodemy's
    App\\Services\\DocTranslate\\PdfTitle repairs a REUSED translation that
    carries the DONOR user's /Title — hands the corrected one down here for
    free), /Lang, and /ViewerPreferences /Direction, so a translation that
    declares itself RTL does not open LTR just because the reader asked for
    both languages at once. Nothing is invented: a mono without a title
    hands the original's through, and a mono that declares neither /Lang
    nor a direction contributes neither.

    /Direction decides which side a viewer puts a page on in a two-up
    spread; it changes no page order in the file (the pages are written in
    the order {@link _alternating} / {@link _side_by_side} built them).
    """
    out: dict[str, str] = {}

    for reader in (translated, original):
        title = _resolved((reader.metadata or {}).get("/Title"))

        if isinstance(title, str) and title.strip():
            out["title"] = str(title)
            break

    for reader in (translated, original):
        root = reader.root_object
        lang = _resolved(root.get("/Lang"))

        if isinstance(lang, str) and lang.strip():
            out["lang"] = str(lang)
            break

    viewer = _resolved(translated.root_object.get("/ViewerPreferences"))
    direction = _resolved(viewer.get("/Direction")) if viewer else None

    if direction in ("/R2L", "/L2R"):
        out["direction"] = str(direction)

    return out


def _carry_properties(doc, properties: dict[str, str], outline: list,
                      fmt: str, unit_count: int,
                      insertions: dict[int, int]) -> None:
    """The inherited declarations and the remapped outline, onto the dual.

    Without this a dual is an anonymous, unnavigable stack of pages: the
    same paid translation delivered as a mono opens with 50 bookmarks and a
    title, delivered as a dual with none (measured: run30 50 -> 0 outline
    entries, run22 41 -> 0). The outline is the ORIGINAL's — its
    destinations are page indices this layout knows how to move (page i
    starts unit i), and its titles are the ones the uploaded document
    itself chose.
    """
    if properties.get("title"):
        # Only what is actually set: pymupdf writes every key it is handed,
        # so passing its own read-back would put a null /Author, /Subject
        # and /Trapped into a delivered file.
        kept = {key: value for key, value in (doc.metadata or {}).items()
                if isinstance(value, str) and value
                and key not in ("format", "encryption")}
        doc.set_metadata({**kept, "title": properties["title"]})

    catalog = doc.pdf_catalog()

    if properties.get("lang"):
        import pymupdf

        doc.xref_set_key(catalog, "Lang",
                         pymupdf.get_pdf_str(properties["lang"]))

    if properties.get("direction"):
        doc.xref_set_key(catalog, "ViewerPreferences/Direction",
                         properties["direction"])

    if not outline:
        return

    shifts = sorted(insertions.items())
    entries = []

    for level, title, page in outline:
        unit = page - 1  # get_toc is 1-based, and -1 when it cannot resolve

        if not 0 <= unit < unit_count:
            continue  # a destination this dual has no unit for

        # The unit's ORIGINAL page: where the reader starts reading it.
        index = 2 * unit if fmt == "alternating" else unit

        # Every vocab page the layer had to INSERT pushed the pages after
        # its anchor down by one; the outline is written last and has to
        # point at where they ended up.
        index += sum(count for anchor, count in shifts if anchor < index)
        # A dropped destination must not orphan its children: a level may
        # never jump by more than one below the level above it.
        level = min(int(level), (entries[-1][0] + 1) if entries else 1)
        entries.append([max(level, 1), str(title), index + 1])  # 1-based

    if entries:
        doc.set_toc(entries)


def _outline_of(pdf_bytes: bytes) -> list:
    """The document's outline as flat [level, title, 1-based page] rows."""
    import pymupdf

    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")

    try:
        return doc.get_toc(simple=True)
    finally:
        doc.close()


def _finalize(composed: bytes, *, fmt: str, sidecar: dict | None,
              vocab: bool, properties: dict[str, str], outline: list,
              unit_count: int, content_count: int) -> bytes:
    """The composed pages, finished — in ONE pass over the document.

    The vocab layer, the side_by_side gutter, the inherited declarations,
    the Arabic text layer, and the single compacting save all want the
    whole document, and the save is the expensive part, so they share one.

    That save is why this runs for every dual and not only the ones with a
    vocab layer. pypdf's writer neither deduplicates nor compresses, and
    `merge_transformed_page` gives every composed side_by_side page its own
    copy of both source pages' fonts and images; a `garbage=4, deflate=True`
    save folds them back together. Measured on the corpus: run14
    side_by_side without vocab 7.21 MB -> 2.48 MB, run20 10.43 MB ->
    3.13 MB, the prod-delivered run3 6.90 MB -> 2.06 MB. Only the duals
    whose vocab layer happened to re-serialize used to get it, so asking
    for NO word list made the download nearly three times bigger.
    """
    import pymupdf

    doc = pymupdf.open(stream=BytesIO(composed), filetype="pdf")

    try:
        if fmt == "side_by_side":
            _draw_gutters(doc, unit_count)

        insertions: dict[int, int] = {}

        if config.VOCAB_PAGES and vocab and isinstance(sidecar, dict):
            try:
                insertions = _attach_vocab_layer(doc, sidecar, fmt,
                                                 unit_count, content_count)
            except Exception:  # noqa: BLE001 - the vocab layer is optional
                logger.exception("compose: inserting vocab pages failed; "
                                 "finishing the dual without them")

        _carry_properties(doc, properties, outline, fmt, unit_count,
                          insertions)

        # AFTER every strip is on the page and BEFORE the save: the strips
        # are drawn with a subset font that shapes Arabic through its cmap,
        # so their text layer extracts as presentation forms (and worse)
        # until this rewrites what each glyph claims to be. The mono's own
        # body arrives already repaired; these strips are drawn fresh here
        # and would otherwise be the one broken layer in the file.
        from server import page_fonts

        page_fonts.repair_arabic_text_layer(doc)

        out = BytesIO()
        # garbage=4 folds the per-page copies of the merged sources' fonts
        # and images into one.
        doc.save(out, garbage=4, deflate=True)

        return out.getvalue()
    finally:
        doc.close()


def compose_dual(original_bytes: bytes, translated_bytes: bytes,
                 fmt: str, sidecar: dict | None = None,
                 vocab: bool = True) -> bytes:
    """Build the requested dual variant; raises ComposeError on bad input.

    `sidecar` is optional — see the module docstring for what one adds. The
    one thing its absence changes is a translated input LONGER than the
    original: only its "artifact_layout" can say whether those pages are an
    appended tail or vocab pages sitting between content pages, so without
    it that build is refused ({@link _reconcile}) rather than mis-paired.

    `vocab=False` is the caller opting out of the «كلمات هذه الصفحة» layer
    for THIS download: the baked-in vocab is still taken back out exactly as
    always (the sidecar's artifact_layout describes the input either way),
    but no fresh strips are drawn afterwards — the dual comes back clean.
    The config.VOCAB_PAGES kill switch keeps overriding everything to off.
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

    _reconcile(original, translated, positions)

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

    try:
        properties = _document_properties(translated, original)
        outline = _outline_of(original_bytes)
    except Exception:  # noqa: BLE001 - both inputs are uploader-supplied
        # What a document says about itself is a nicety; a malformed
        # catalog must not cost the reader the dual itself.
        logger.exception("compose: reading the inputs' own properties "
                         "failed; building the dual without them")
        properties, outline = {}, []

    build = _alternating if fmt == "alternating" else _side_by_side
    out = BytesIO()
    build(original, translated_pages).write(out)
    composed = out.getvalue()

    try:
        # The vocab strips live in the body, on each pair's unit.
        composed = _finalize(composed, fmt=fmt, sidecar=sidecar,
                             vocab=vocab and vocab_ok,
                             properties=properties, outline=outline,
                             unit_count=len(original.pages),
                             content_count=len(translated_pages))
    except Exception:  # noqa: BLE001 - never lose the dual over finishing it
        logger.exception("compose: finishing the dual failed; returning the "
                         "composed pages as they were built")

    return composed


def strip_vocab(translated_bytes: bytes, sidecar) -> bytes:
    """The stored mono with its baked «كلمات هذه الصفحة» layer taken back out.

    The un-bake half of what {@link compose_dual} does before it pairs pages,
    offered on its own: the sidecar's artifact_layout says exactly what the
    pipeline baked in, so inserted fallback vocab pages are dropped (keep the
    content_pages positions, nothing else) and baked bottom strips are
    PHYSICALLY redacted off with the mediabox restored afterwards
    ({@link _crop_baked_strips}). A sidecar without a usable layout means a
    pre-vocab mono — there is nothing to strip, and the input bytes come back
    byte-for-byte unchanged rather than pointlessly re-serialized.

    Unlike compose, where the vocab layer is best-effort garnish on a free
    dual, stripping IS this call's whole job — so a crop that fails raises
    ComposeError (HTTP 422) instead of quietly returning the vocab it was
    asked to remove.
    """
    translated = _read(translated_bytes, "translated")
    positions = _artifact_content_positions(sidecar, len(translated.pages))

    if positions is None:
        return translated_bytes

    strips = _artifact_strip_heights(sidecar)

    if strips:
        try:
            translated_bytes = _crop_baked_strips(translated_bytes,
                                                  positions, strips)
        except Exception as exc:  # noqa: BLE001 - pymupdf raises many types
            raise ComposeError(
                f"could not remove the baked vocab strips: {exc}") from exc

        translated = _read(translated_bytes, "translated")

    if positions == list(range(len(translated.pages))):
        # Strip-era mono: every page is a content page, and the strips (if
        # any) are already gone from the bytes above.
        return translated_bytes

    writer = PdfWriter()

    for position in positions:
        writer.add_page(translated.pages[position])

    out = BytesIO()
    writer.write(out)

    return out.getvalue()
