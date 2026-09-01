"""Tests for the stateless /v1/strip-vocab endpoint and compose.strip_vocab.

Run from the repo root (pyproject's testpaths only covers tests/, so pass
the path explicitly):

    pytest server/test_strip_vocab.py
"""

import json
import unicodedata
from io import BytesIO

import pymupdf
import pytest
from pypdf import PdfWriter

from server import page_fonts
from server import vocab_pages
from server.conftest import TOKEN

A4 = (595.0, 842.0)

VOCAB = {"0": [{"w": "declared", "ar": "يُصرَّح عنه"}],
         "1": [{"w": "evolved", "ar": "تطوَّر", "note": "تغيَّر مع الوقت"}]}


@pytest.fixture(scope="module", autouse=True)
def fonts_cached():
    """Fetch the faces once — the fixture bakes real strips with them."""
    page_fonts._font_path(page_fonts.FONT_FILE)
    page_fonts._font_path(page_fonts.BOLD_FONT_FILE)


def _blank_pdf(page_sizes: list[tuple[float, float]]) -> bytes:
    writer = PdfWriter()
    for w, h in page_sizes:
        writer.add_blank_page(width=w, height=h)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _strip_baked_mono() -> tuple[bytes, dict]:
    """A mono exactly as the pipeline bakes it: two content pages whose
    strips were attached by the REAL renderer, plus the artifact_layout the
    job sidecar would record for them."""
    doc = pymupdf.open()
    for index in range(2):
        page = doc.new_page(width=A4[0], height=A4[1])
        page.insert_text((72, 72), f"CONTENT{index}", fontsize=24)

    added = vocab_pages.attach_vocab(doc, VOCAB, {0: 0, 1: 1})
    assert all(height > 0 for height in added.values())  # strips, not pages

    out = BytesIO()
    doc.save(out, garbage=4, deflate=True)
    doc.close()

    layout = {"content_pages": [0, 1],
              "vocab_strips": {str(number): height
                               for number, height in added.items()}}
    return out.getvalue(), layout


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


def _post(client, translated, sidecar, token=TOKEN):
    return client.post(
        "/v1/strip-vocab",
        headers={"X-Internal-Token": token} if token else {},
        files={"translated": ("trans.pdf", translated, "application/pdf"),
               "sidecar": ("sidecar.json", sidecar, "application/json")},
    )


def _doc_facts(pdf_bytes: bytes) -> tuple[list[float], str]:
    """(page heights, all text NFKC-normalized) of a PDF."""
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        heights = [doc[i].rect.height for i in range(doc.page_count)]
        text = unicodedata.normalize(
            "NFKC", " ".join(doc[i].get_text() for i in range(doc.page_count)))
        return heights, text
    finally:
        doc.close()


def test_a_strip_baked_mono_comes_back_pristine(client):
    translated, layout = _strip_baked_mono()
    heights, text = _doc_facts(translated)
    assert all(height > A4[1] for height in heights)  # the fixture DID grow
    assert "declared" in text

    resp = _post(client, translated,
                 _sidecar_json(2, vocab=VOCAB, artifact_layout=layout))

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    heights, text = _doc_facts(resp.content)
    assert len(heights) == 2
    assert heights == [pytest.approx(A4[1]), pytest.approx(A4[1])]
    assert "CONTENT0" in text and "CONTENT1" in text  # the content survives
    assert "declared" not in text and "evolved" not in text
    assert "كلمات" not in text  # the strip title went with its rows


def test_an_inserted_pages_mono_drops_the_vocab_pages(client):
    # The inserted-pages era: content page, its vocab page (odd size marks
    # it), content page, a baked appendix tail. Only the content survives.
    translated = _blank_pdf([A4, (400.0, 400.0), A4, (500.0, 500.0)])

    resp = _post(client, translated,
                 _sidecar_json(2, vocab=VOCAB,
                               artifact_layout={"content_pages": [0, 2]}))

    assert resp.status_code == 200
    heights, _text = _doc_facts(resp.content)
    assert heights == [pytest.approx(A4[1]), pytest.approx(A4[1])]


def test_a_legacy_sidecar_returns_the_bytes_unchanged(client):
    # No artifact_layout — a pre-vocab mono has nothing to strip, and the
    # caller gets back byte-for-byte what it sent, not a re-serialization.
    translated = _blank_pdf([A4] * 2)

    resp = _post(client, translated, _sidecar_json(2))

    assert resp.status_code == 200
    assert resp.content == translated


def test_a_crafted_layout_is_refused_whole_and_treated_as_legacy(client):
    # Non-increasing content_pages: the layout is junk, so the mono is
    # treated as having nothing recorded to strip.
    translated = _blank_pdf([A4] * 2)

    resp = _post(client, translated,
                 _sidecar_json(2, artifact_layout={"content_pages": [1, 0]}))

    assert resp.status_code == 200
    assert resp.content == translated


def test_a_malformed_sidecar_is_422(client):
    resp = _post(client, _blank_pdf([A4]), "{not json")
    assert resp.status_code == 422
    assert "valid JSON" in resp.json()["detail"]


def test_an_unreadable_pdf_is_422(client):
    resp = _post(client, b"%PDF-1.4 but junk after the magic",
                 _sidecar_json(1))
    assert resp.status_code == 422


def test_a_non_pdf_is_422(client):
    resp = _post(client, b"not a pdf", _sidecar_json(1))
    assert resp.status_code == 422


def test_missing_token_is_401(client):
    resp = _post(client, _blank_pdf([A4]), _sidecar_json(1), token=None)
    assert resp.status_code == 401


# --------------------------------------------------------------------------
# the text layer of a mono this engine did not draw
# --------------------------------------------------------------------------

def _arabic_classes(pdf_bytes: bytes) -> tuple[int, int]:
    """(base letters, presentation forms) in a PDF's extracted text.

    Deliberately NOT NFKC-normalized, unlike `_doc_facts`: that fold turns
    presentation forms into the base letters this counts, which is exactly
    the difference under test.
    """
    doc = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        text = "".join(page.get_text() for page in doc)
    finally:
        doc.close()

    base = sum(1 for ch in text
               if "ؠ" <= ch <= "ي" or "ٱ" <= ch <= "ۓ")
    presentation = sum(1 for ch in text
                       if "ﭐ" <= ch <= "﷿"
                       or "ﹰ" <= ch <= "﻿")

    return base, presentation


AR_BODY = "تغيير البرمجيات أمر لا مفر منه، وهو ما تفترضه هذه الصفحة"


def _mono_as_banked() -> bytes:
    """Two A4 pages of Arabic body text, drawn and saved the way a run banked
    before the CMap fix left it.

    The same `insert_htmlbox` + subset-font path the pipeline draws Arabic
    through, and then a plain save with NO text-layer repair — so the file
    renders correctly and extracts as glyph shapes, which is precisely the
    state of all 87 translations already sitting in production.
    """
    fonts = page_fonts.PageFonts([AR_BODY])
    doc = pymupdf.open()

    for _ in range(2):
        page = doc.new_page(width=A4[0], height=A4[1])
        page.insert_htmlbox(pymupdf.Rect(60, 60, A4[0] - 60, 400),
                            f"<div>{AR_BODY}</div>",
                            css=fonts.css, archive=fonts.archive)

    out = BytesIO()
    doc.save(out, garbage=4, deflate=True)
    doc.close()

    return out.getvalue()


def test_a_mono_banked_before_the_cmap_fix_comes_back_searchable(client):
    """The reason this endpoint repairs a text layer it did not draw.

    Every run banked before the CMap fix stores its Arabic as presentation
    forms, and those are the runs readers reach for this endpoint with — it
    exists to re-cut an ALREADY finished translation. A legacy sidecar is the
    sharpest case: there is nothing to strip, so before the repair these bytes
    came back exactly as banked, glyph soup and all.
    """
    translated = _mono_as_banked()
    base_in, presentation_in = _arabic_classes(translated)
    # The fixture really is an old-engine artifact: its shaped Arabic extracts
    # as glyph shapes rather than as letters.
    assert presentation_in > 0
    assert presentation_in > base_in

    resp = _post(client, translated, _sidecar_json(2))

    assert resp.status_code == 200
    base_out, presentation_out = _arabic_classes(resp.content)
    # Every shape is now a letter, and no Arabic was lost turning it into one.
    assert presentation_out == 0
    assert base_out >= base_in + presentation_in
    # ...and the reader can actually find it.
    doc = pymupdf.open(stream=BytesIO(resp.content), filetype="pdf")
    try:
        assert any(page.search_for("البرمجيات") for page in doc)
    finally:
        doc.close()


def test_a_mono_with_no_arabic_is_returned_byte_for_byte(client):
    """The repair must not cost the no-op its no-op.

    A document with nothing to fix is handed back unserialized — which is
    what keeps a legacy strip cheap and a current-engine mono free.
    """
    translated = _blank_pdf([A4] * 2)

    resp = _post(client, translated, _sidecar_json(2))

    assert resp.status_code == 200
    assert resp.content == translated
