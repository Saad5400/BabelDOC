"""Tests for the stateless /v1/compose endpoint and server.compose.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_compose.py
"""

import json
from io import BytesIO

import pytest
from pypdf import PdfReader
from pypdf import PdfWriter

from server.conftest import TOKEN

# Distinguishable page sizes (points).
LETTER = (612.0, 792.0)   # original: 3 pages
A4 = (595.0, 842.0)       # translated: 2 pages


def _pdf(page_sizes: list[tuple[float, float]]) -> bytes:
    writer = PdfWriter()
    for w, h in page_sizes:
        writer.add_blank_page(width=w, height=h)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _text_pdf(marker: str, size: tuple[float, float]) -> bytes:
    """Minimal one-page PDF whose text layer contains `marker`."""
    w, h = size
    content = f"BT /F1 24 Tf 40 40 Td ({marker}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:g} {h:g}] "
         f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>").encode(),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref_pos))
    return bytes(out)


@pytest.fixture(autouse=True)
def arabic_text_layer_repair(monkeypatch):
    """`page_fonts.repair_arabic_text_layer` while its branch is unmerged.

    compose calls it on every finished dual (the fresh vocab strips are
    drawn here, so their broken text layer is this module's to repair). The
    function itself belongs to server/page_fonts.py and lands with the
    engine-text fix; until then a no-op stands in, so these tests exercise
    the composition rather than the absence of a dependency. Once it is
    there, this fixture does nothing and the real one runs.
    """
    from server import page_fonts

    if not hasattr(page_fonts, "repair_arabic_text_layer"):
        monkeypatch.setattr(page_fonts, "repair_arabic_text_layer",
                            lambda doc: 0, raising=False)


@pytest.fixture(scope="module")
def original_pdf() -> bytes:
    return _pdf([LETTER] * 3)


@pytest.fixture(scope="module")
def translated_pdf() -> bytes:
    return _pdf([A4] * 2)


def _post(client, original, translated, fmt, token=TOKEN):
    return client.post(
        "/v1/compose",
        headers={"X-Internal-Token": token} if token else {},
        files={"original": ("orig.pdf", original, "application/pdf"),
               "translated": ("trans.pdf", translated, "application/pdf")},
        data={"format": fmt},
    )


def _pages(response) -> list:
    assert response.headers["content-type"] == "application/pdf"
    return list(PdfReader(BytesIO(response.content)).pages)


def _size(page) -> tuple[float, float]:
    return float(page.mediabox.width), float(page.mediabox.height)


def test_alternating_interleaves_and_pads(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "alternating")
    assert resp.status_code == 200
    pages = _pages(resp)
    # 3 originals + 2 translated + 1 blank pad = 6, interleaved o,t,o,t,o,blank.
    assert len(pages) == 6
    assert [_size(p) for p in pages[:4]] == [LETTER, A4, LETTER, A4]
    assert _size(pages[4]) == LETTER
    assert _size(pages[5]) == LETTER  # pad blank sized like its original twin


def test_side_by_side_dimensions(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "side_by_side")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 3  # max(3, 2)
    half = max(LETTER[0], A4[0])
    # Pages 1-2 pair letter + A4: width = two halves, height = max height.
    for page in pages[:2]:
        assert _size(page) == (2 * half, max(LETTER[1], A4[1]))
    # Page 3 has no translated twin: sized from the original alone.
    assert _size(pages[2]) == (2 * LETTER[0], LETTER[1])


def test_side_by_side_keeps_both_text_layers(client):
    resp = _post(client, _text_pdf("ORIGMARK", LETTER),
                 _text_pdf("TRANSMARK", A4), "side_by_side")
    assert resp.status_code == 200
    text = _pages(resp)[0].extract_text()
    assert "ORIGMARK" in text
    assert "TRANSMARK" in text


def test_alternating_keeps_page_order_of_text(client):
    resp = _post(client, _text_pdf("ORIGMARK", LETTER),
                 _text_pdf("TRANSMARK", A4), "alternating")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert "ORIGMARK" in pages[0].extract_text()
    assert "TRANSMARK" in pages[1].extract_text()


def test_an_unexplained_longer_translated_is_refused(client, original_pdf,
                                                     translated_pdf):
    # Swap the inputs: original (2 pages) shorter than translated (3 pages),
    # with nothing saying why. The extra page could be an appended tail — or
    # a «كلمات هذه الصفحة» page the mono run had to INSERT between content
    # pages, in which case pairing positionally mis-pairs every unit after
    # it. Unknowable here, so it is refused rather than guessed.
    for fmt in ("alternating", "side_by_side"):
        resp = _post(client, translated_pdf, original_pdf, fmt)
        assert resp.status_code == 422
        assert "artifact_layout" in resp.json()["detail"]


def test_a_longer_translated_appends_its_tail_when_the_layout_explains_it(
        client, original_pdf, translated_pdf):
    # The same shape, with the sidecar's layout saying all three pages are
    # content: the extra one is then known to be a tail and is appended
    # whole at the end rather than interleaved against a blank.
    resp = _post_with_sidecar(
        client, translated_pdf, original_pdf, "alternating",
        _sidecar_json(3, artifact_layout={"content_pages": [0, 1, 2]}))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 5  # 2 pairs + 1 tail page
    assert [_size(p) for p in pages[:4]] == [A4, LETTER, A4, LETTER]
    assert _size(pages[4]) == LETTER  # the translated tail page itself


def test_side_by_side_appends_the_translated_tail_full_width(
        client, original_pdf, translated_pdf):
    resp = _post_with_sidecar(
        client, translated_pdf, original_pdf, "side_by_side",
        _sidecar_json(3, artifact_layout={"content_pages": [0, 1, 2]}))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 3  # 2 pairs + 1 tail page
    assert _size(pages[2]) == LETTER  # whole, at its own size


def test_bad_format_is_422(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "mono")
    assert resp.status_code == 422


def test_missing_file_is_422(client, original_pdf):
    resp = client.post(
        "/v1/compose",
        headers={"X-Internal-Token": TOKEN},
        files={"original": ("orig.pdf", original_pdf, "application/pdf")},
        data={"format": "alternating"},
    )
    assert resp.status_code == 422


def test_non_pdf_is_422(client, original_pdf):
    resp = _post(client, original_pdf, b"not a pdf", "alternating")
    assert resp.status_code == 422


def test_missing_token_is_401(client, original_pdf, translated_pdf):
    resp = _post(client, original_pdf, translated_pdf, "alternating", token=None)
    assert resp.status_code == 401


# --------------------------------------------------------------------------
# The optional sidecar part: «كلمات هذه الصفحة» strips drawn onto the dual
# --------------------------------------------------------------------------

VOCAB = {"0": [{"w": "declared", "ar": "يُصرَّح عنه"}],
         "1": [{"w": "evolved", "ar": "تطوَّر", "note": "تغيَّر مع الوقت"}]}


def _sidecar_json(total_pages, vocab=None, artifact_layout=None):
    data = {"version": 1, "lang_in": "en", "lang_out": "ar",
            "total_pages": total_pages,
            "pages": [{"page_number": i, "mediabox": [0, 0, *A4],
                       "blocks": [], "obstacles": []}
                      for i in range(total_pages)]}
    if vocab is not None:
        data["vocab"] = vocab
    if artifact_layout is not None:
        data["artifact_layout"] = artifact_layout
    return json.dumps(data)


def _post_with_sidecar(client, original, translated, fmt, sidecar, **data):
    return client.post(
        "/v1/compose",
        headers={"X-Internal-Token": TOKEN},
        files={"original": ("orig.pdf", original, "application/pdf"),
               "translated": ("trans.pdf", translated, "application/pdf"),
               "sidecar": ("sidecar.json", sidecar, "application/json")},
        data={"format": fmt, **data},
    )


def _page_text(response, index):
    """Text of one page of the response PDF, via pymupdf.

    pymupdf, not pypdf: the vocab pages carry a subset font whose spans
    pypdf's extractor reads only partially.
    """
    import pymupdf

    doc = pymupdf.open(stream=BytesIO(response.content), filetype="pdf")
    try:
        return doc[index].get_text()
    finally:
        doc.close()


def test_alternating_strips_each_pairs_translated_page(client):
    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2, vocab=VOCAB))
    assert resp.status_code == 200
    pages = _pages(resp)
    # o0 t0+strip o1 t1+strip — nothing inserted, the translated pages grew.
    assert len(pages) == 4
    sizes = [_size(p) for p in pages]
    assert sizes[0] == LETTER
    assert sizes[2] == LETTER
    for index in (1, 3):
        assert sizes[index][0] == A4[0]
        assert sizes[index][1] > A4[1]
    assert "declared" in _page_text(resp, 1)
    assert "evolved" in _page_text(resp, 3)


def test_side_by_side_strips_each_wide_page(client):
    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(client, original, translated, "side_by_side",
                              _sidecar_json(2, vocab=VOCAB))
    assert resp.status_code == 200
    pages = _pages(resp)
    # wide0+strip wide1+strip — the strip spans the whole wide page's width.
    assert len(pages) == 2
    wide = (2 * max(LETTER[0], A4[0]), max(LETTER[1], A4[1]))
    for size in [_size(p) for p in pages]:
        assert size[0] == wide[0]
        assert size[1] > wide[1]
    assert "declared" in _page_text(resp, 0)
    assert "evolved" in _page_text(resp, 1)


def test_a_sidecar_without_vocab_changes_nothing(client, original_pdf,
                                                 translated_pdf):
    plain = _post(client, original_pdf, translated_pdf, "alternating")
    with_sidecar = _post_with_sidecar(client, original_pdf, translated_pdf,
                                      "alternating", _sidecar_json(2))
    assert with_sidecar.status_code == 200
    assert len(_pages(with_sidecar)) == len(_pages(plain))


def test_an_empty_vocab_inserts_nothing(client, original_pdf, translated_pdf):
    resp = _post_with_sidecar(client, original_pdf, translated_pdf,
                              "alternating", _sidecar_json(2, vocab={}))
    assert resp.status_code == 200
    assert len(_pages(resp)) == 6  # exactly the plain alternating output


def test_artifact_layout_takes_the_content_pages_back_out_exactly(client):
    # A legacy baked mono (inserted vocab pages): content page 0, its vocab
    # page (odd size marks it), content page 1, plus a baked appendix tail
    # (another odd size). The sidecar records where the content pages sit;
    # the dual must rebuild from JUST those and render the vocab fresh.
    original = _pdf([LETTER] * 2)
    translated = _pdf([A4, (400.0, 400.0), A4, (500.0, 500.0)])
    resp = _post_with_sidecar(
        client, original, translated, "alternating",
        _sidecar_json(2, vocab=VOCAB,
                      artifact_layout={"content_pages": [0, 2]}))
    assert resp.status_code == 200
    pages = _pages(resp)
    sizes = [_size(p) for p in pages]
    assert (400.0, 400.0) not in sizes  # the baked vocab page was dropped
    assert (500.0, 500.0) not in sizes  # the baked appendix tail was dropped
    # o0 t0+strip o1 t1+strip — the vocab rendered fresh, as strips.
    assert len(pages) == 4
    assert sizes[0] == sizes[2] == LETTER
    assert sizes[1][1] > A4[1]
    assert "declared" in _page_text(resp, 1)
    assert "evolved" in _page_text(resp, 3)


def test_baked_strips_are_cropped_back_off_before_composing(client):
    # A strip-baked mono as the pipeline writes it now: each content page's
    # mediabox reaches below zero by the recorded height, with the baked
    # strip text living in that band. The dual must recover the pristine
    # pages (no doubled words from the baked band) and draw fresh strips.
    import pymupdf

    doc = pymupdf.open()
    for index in range(2):
        page = doc.new_page(width=A4[0], height=A4[1])
        page.insert_text((72, 72), f"CONTENT{index}", fontsize=24)
        media = page.mediabox
        page.set_mediabox(pymupdf.Rect(media.x0, media.y0 - 100.0,
                                       media.x1, media.y1))
        page.insert_text((72, A4[1] + 50), f"BAKEDSTRIP{index}", fontsize=18)
    translated = doc.tobytes()
    doc.close()

    original = _pdf([LETTER] * 2)
    resp = _post_with_sidecar(
        client, original, translated, "alternating",
        _sidecar_json(2, vocab=VOCAB,
                      artifact_layout={"content_pages": [0, 1],
                                       "vocab_strips": {"0": 100.0,
                                                        "1": 100.0}}))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 4
    text = " ".join(_page_text(resp, i) for i in range(4))
    assert "CONTENT0" in text and "CONTENT1" in text
    assert "BAKEDSTRIP" not in text  # the baked band is gone
    assert "declared" in _page_text(resp, 1)  # the fresh strip replaced it
    assert "evolved" in _page_text(resp, 3)


def test_a_pre_feature_sidecar_still_tail_trims_by_total_pages(client):
    # No "vocab", no "artifact_layout" — a sidecar from before this feature,
    # sent with a translated input that carries a baked tail past
    # total_pages. The tail is dropped, nothing is interleaved. (Three
    # originals, so the input is not LONGER than the original and the
    # trim is a trim rather than a guess — see the refusal test above.)
    original = _pdf([LETTER] * 3)
    translated = _pdf([A4] * 2 + [(500.0, 500.0)])
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert (500.0, 500.0) not in [_size(p) for p in pages]
    assert len(pages) == 6
    assert [_size(p) for p in pages[:4]] == [LETTER, A4, LETTER, A4]


def test_a_crafted_artifact_layout_is_refused_whole(client):
    # Duplicated / non-increasing positions would double pages; the layout is
    # ignored and the total_pages tail-trim takes over.
    original = _pdf([LETTER] * 3)
    translated = _pdf([A4] * 2 + [(500.0, 500.0)])
    resp = _post_with_sidecar(
        client, original, translated, "alternating",
        _sidecar_json(2, artifact_layout={"content_pages": [0, 0]}))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert (500.0, 500.0) not in [_size(p) for p in pages]
    assert [_size(p) for p in pages[:4]] == [LETTER, A4, LETTER, A4]


def test_the_vocab_kill_switch_disables_the_insert(client, monkeypatch):
    from server import config
    monkeypatch.setattr(config, "VOCAB_PAGES", False)

    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2, vocab=VOCAB))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 4  # pairs only
    assert [_size(p) for p in pages] == [LETTER, A4] * 2  # and no strips


def test_vocab_for_a_page_neither_side_has_is_skipped(client):
    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(
        client, original, translated, "alternating",
        _sidecar_json(2, vocab={"9": [{"w": "lost", "ar": "ضائع"}],
                                "junk": [{"w": "lost", "ar": "ضائع"}]}))
    assert resp.status_code == 200
    assert len(_pages(resp)) == 4


def test_an_unknown_sidecar_key_is_tolerated(client):
    # A sidecar from a pipeline with more passes than this one (a deep-terms
    # "glossary", say): unknown top-level keys are data for someone else,
    # never an error and never extra pages here.
    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    sidecar = json.loads(_sidecar_json(2, vocab=VOCAB))
    sidecar["glossary"] = [{"term": "Wrapping", "arabic": "التغليف",
                            "explanation": "شرح", "page": 1, "quote": None}]
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              json.dumps(sidecar))
    assert resp.status_code == 200
    assert len(_pages(resp)) == 4  # o0 t0+strip o1 t1+strip — nothing more


def test_a_malformed_sidecar_is_422(client, original_pdf, translated_pdf):
    resp = _post_with_sidecar(client, original_pdf, translated_pdf,
                              "alternating", "{not json")
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# vocab=0: the caller opts one download out of the vocab layer
# --------------------------------------------------------------------------

def _strip_baked_translated() -> bytes:
    """A strip-baked mono like the pipeline's: content pages whose mediabox
    reaches 100pt below zero, with the baked strip text in that band."""
    import pymupdf

    doc = pymupdf.open()
    for index in range(2):
        page = doc.new_page(width=A4[0], height=A4[1])
        page.insert_text((72, 72), f"CONTENT{index}", fontsize=24)
        media = page.mediabox
        page.set_mediabox(pymupdf.Rect(media.x0, media.y0 - 100.0,
                                       media.x1, media.y1))
        page.insert_text((72, A4[1] + 50), f"BAKEDSTRIP{index}", fontsize=18)
    try:
        return doc.tobytes()
    finally:
        doc.close()


def test_vocab_0_still_unbakes_but_draws_no_fresh_strips(client):
    # The opt-out only governs what goes IN: the baked strips still come
    # back out (undoing what the input carries), and then nothing replaces
    # them — the pages end at exactly their pristine sizes.
    original = _pdf([LETTER] * 2)
    resp = _post_with_sidecar(
        client, original, _strip_baked_translated(), "alternating",
        _sidecar_json(2, vocab=VOCAB,
                      artifact_layout={"content_pages": [0, 1],
                                       "vocab_strips": {"0": 100.0,
                                                        "1": 100.0}}),
        vocab="0")
    assert resp.status_code == 200
    pages = _pages(resp)
    assert [_size(p) for p in pages] == [LETTER, A4, LETTER, A4]  # no growth
    text = " ".join(_page_text(resp, i) for i in range(4))
    assert "CONTENT0" in text and "CONTENT1" in text
    assert "BAKEDSTRIP" not in text  # the baked band is still removed
    assert "declared" not in text    # and nothing fresh went in
    assert "evolved" not in text


def test_vocab_false_is_tolerated_as_a_spelling(client):
    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2, vocab=VOCAB), vocab="false")
    assert resp.status_code == 200
    assert [_size(p) for p in _pages(resp)] == [LETTER, A4] * 2


def test_vocab_1_is_the_default_and_explicit_1_matches_it(client):
    original, translated = _pdf([LETTER] * 2), _pdf([A4] * 2)
    resp = _post_with_sidecar(client, original, translated, "alternating",
                              _sidecar_json(2, vocab=VOCAB), vocab="1")
    assert resp.status_code == 200
    sizes = [_size(p) for p in _pages(resp)]
    assert sizes[1][1] > A4[1]  # the strips still land when asked for


def test_a_junk_vocab_value_is_422(client, original_pdf, translated_pdf):
    resp = _post_with_sidecar(client, original_pdf, translated_pdf,
                              "alternating", _sidecar_json(2, vocab=VOCAB),
                              vocab="maybe")
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# What the finished document carries: one compacting save, the inputs' own
# declarations, the original's outline, the seam, and /Rotate
# --------------------------------------------------------------------------

def _imaged_pdf(pages: int, size: tuple[float, float] = A4) -> bytes:
    """`pages` pages all showing the SAME image.

    A composed page merges its two sources' resources into itself, so a
    shared image is exactly what a dual duplicates if nothing folds it back
    together afterwards.
    """
    import pymupdf

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 160, 160))
    pix.set_rect(pix.irect, (255, 255, 255))
    for x in range(0, 160, 2):  # stripes: cheap to build, poor to compress
        pix.set_rect(pymupdf.IRect(x, 0, x + 1, 160), (17 * x % 256,) * 3)
    image = pix.tobytes("png")

    doc = pymupdf.open()
    try:
        for _ in range(pages):
            page = doc.new_page(width=size[0], height=size[1])
            page.insert_image(pymupdf.Rect(20, 20, size[0] - 20, size[1] - 20),
                              stream=image)
        return doc.tobytes(garbage=4, deflate=True)
    finally:
        doc.close()


def _declaring_pdf(pages: int, *, title: str | None = None,
                   lang: str | None = None, direction: str | None = None,
                   outline: list | None = None,
                   rotation: dict[int, int] | None = None) -> bytes:
    """A PDF that declares things about itself, for the carry-over tests."""
    import pymupdf

    doc = pymupdf.open()
    try:
        for index in range(pages):
            page = doc.new_page(width=A4[0], height=A4[1])
            page.insert_text((72, 72), f"PAGE{index}", fontsize=24)
            if rotation and index in rotation:
                page.set_rotation(rotation[index])
        if title:
            doc.set_metadata({**(doc.metadata or {}), "title": title})
        if lang:
            doc.xref_set_key(doc.pdf_catalog(), "Lang",
                             pymupdf.get_pdf_str(lang))
        if direction:
            doc.xref_set_key(doc.pdf_catalog(), "ViewerPreferences/Direction",
                             direction)
        if outline:
            doc.set_toc(outline)
        return doc.tobytes()
    finally:
        doc.close()


def _opened(response):
    import pymupdf

    return pymupdf.open(stream=BytesIO(response.content), filetype="pdf")


def test_an_interleaved_mono_without_its_layout_is_refused_not_mis_paired(
        client):
    # The shape the engine's own vocab fallback produces: content, vocab,
    # content, vocab. Told which pages are content, the dual pairs them
    # exactly; not told, the same file would pair unit 2 with a VOCAB page
    # — so it is refused instead.
    from pypdf import PdfWriter as _Writer

    merged = _Writer()
    for marker in ("CONTENTA", "VOCABA", "CONTENTB", "VOCABB"):
        merged.append(BytesIO(_text_pdf(marker, A4)))
    buf = BytesIO()
    merged.write(buf)
    translated = buf.getvalue()
    original = _pdf([LETTER] * 2)

    refused = _post_with_sidecar(client, original, translated, "alternating",
                                 _sidecar_json(2))
    assert refused.status_code == 422

    resp = _post_with_sidecar(
        client, original, translated, "alternating",
        _sidecar_json(2, artifact_layout={"content_pages": [0, 2]}))
    assert resp.status_code == 200
    pages = _pages(resp)
    assert len(pages) == 4
    assert "CONTENTA" in pages[1].extract_text()
    assert "CONTENTB" in pages[3].extract_text()
    joined = " ".join(page.extract_text() for page in pages)
    assert "VOCAB" not in joined


def test_the_dual_is_compacted_whether_or_not_the_vocab_layer_runs(client):
    # The vocab path used to be the only one that re-serialized the
    # document, so it was the only one that ever got a compacting save:
    # asking for NO word list handed the reader a file up to 3.4x larger
    # than the same pages with one.
    original, translated = _imaged_pdf(4), _imaged_pdf(4)
    sidecar = _sidecar_json(4, vocab=VOCAB)
    with_vocab = _post_with_sidecar(client, original, translated,
                                    "side_by_side", sidecar)
    without = _post_with_sidecar(client, original, translated, "side_by_side",
                                 sidecar, vocab="0")
    assert with_vocab.status_code == without.status_code == 200
    # Dropping a layer must never grow the file.
    assert len(without.content) <= len(with_vocab.content)
    # And the one image the eight merged halves share is stored once.
    doc = _opened(without)
    try:
        xrefs = {image[0] for index in range(doc.page_count)
                 for image in doc.get_page_images(index)}
    finally:
        doc.close()
    assert len(xrefs) == 1


def test_the_dual_inherits_what_the_mono_declares_about_itself(client):
    original = _declaring_pdf(2, title="the user's own upload", lang="en")
    translated = _declaring_pdf(2, title="الترجمة", lang="ar",
                                direction="/R2L")
    resp = _post(client, original, translated, "alternating")
    assert resp.status_code == 200
    doc = _opened(resp)
    try:
        assert doc.metadata["title"] == "الترجمة"
        assert "ar" in doc.xref_get_key(doc.pdf_catalog(), "Lang")[1]
        assert doc.xref_get_key(doc.pdf_catalog(),
                                "ViewerPreferences/Direction")[1] == "/R2L"
    finally:
        doc.close()


def test_a_mono_without_a_title_hands_the_originals_through(client):
    original = _declaring_pdf(2, title="the user's own upload")
    resp = _post(client, original, _declaring_pdf(2), "alternating")
    assert resp.status_code == 200
    doc = _opened(resp)
    try:
        assert doc.metadata["title"] == "the user's own upload"
    finally:
        doc.close()


@pytest.mark.parametrize(("fmt", "expected"), [("alternating", [1, 5]),
                                               ("side_by_side", [1, 3])])
def test_the_originals_outline_is_remapped_onto_the_units(client, fmt,
                                                          expected):
    # Original pages 1 and 3 are bookmarked; in the dual those units start
    # at pages 1 and 5 (alternating) / 1 and 3 (side_by_side) — the unit's
    # ORIGINAL page, where the reader starts reading it.
    original = _declaring_pdf(3, outline=[[1, "Start", 1],
                                          [2, "Middle", 3]])
    resp = _post(client, original, _declaring_pdf(3), fmt)
    assert resp.status_code == 200
    doc = _opened(resp)
    try:
        toc = doc.get_toc(simple=True)
    finally:
        doc.close()
    assert [title for _level, title, _page in toc] == ["Start", "Middle"]
    assert [page for _level, _title, page in toc] == expected


def test_side_by_side_draws_a_gutter_at_the_seam(client):
    resp = _post(client, _pdf([LETTER] * 2), _pdf([LETTER] * 2),
                 "side_by_side")
    assert resp.status_code == 200
    doc = _opened(resp)
    try:
        page = doc[0]
        seams = [drawing for drawing in page.get_drawings()
                 if abs(drawing["rect"].x0 - page.rect.width / 2) < 1
                 and abs(drawing["rect"].x1 - page.rect.width / 2) < 1]
    finally:
        doc.close()
    assert len(seams) == 1  # one hairline, full height, on the midline
    assert seams[0]["rect"].height == pytest.approx(LETTER[1])


def test_alternating_draws_no_gutter(client):
    resp = _post(client, _pdf([LETTER] * 2), _pdf([LETTER] * 2),
                 "alternating")
    assert resp.status_code == 200
    doc = _opened(resp)
    try:
        assert [len(page.get_drawings()) for page in doc] == [0] * 4
    finally:
        doc.close()


def test_side_by_side_keeps_a_rotated_page_the_way_it_displays(client):
    # A /Rotate 90 page DISPLAYS landscape; merged content does not carry
    # the attribute, so without baking it the page would be drawn upright
    # inside a portrait half — sideways to the reader.
    original = _declaring_pdf(2, rotation={1: 90})
    resp = _post(client, original, _declaring_pdf(2), "side_by_side")
    assert resp.status_code == 200
    pages = _pages(resp)
    # Page 2's original half is A4 landscape, so the half is A4's height
    # wide and the page twice that.
    assert _size(pages[0]) == (2 * A4[0], A4[1])
    assert _size(pages[1]) == (2 * A4[1], A4[1])


def test_the_arabic_text_layer_is_repaired_after_the_strips_and_before_saving(
        client, monkeypatch):
    # The dual inherits the mono's repaired body, but its «كلمات هذه
    # الصفحة» strips are drawn FRESH here — so the repair has to run on
    # this document, once the strips are on it and while it can still be
    # saved.
    from server import page_fonts

    seen: dict[str, object] = {}

    def spy(doc):
        seen["pages"] = doc.page_count
        seen["strips"] = "declared" in doc[1].get_text()
        doc[0].insert_text((72, 200), "REPAIRRAN", fontsize=18)
        return 1

    monkeypatch.setattr(page_fonts, "repair_arabic_text_layer", spy,
                        raising=False)
    resp = _post_with_sidecar(client, _pdf([LETTER] * 2), _pdf([A4] * 2),
                              "alternating", _sidecar_json(2, vocab=VOCAB))
    assert resp.status_code == 200
    assert seen["pages"] == 4
    assert seen["strips"] is True          # after the strips were drawn
    assert "REPAIRRAN" in _page_text(resp, 0)  # and before the save
